"""NSGA-III 튜닝 콜백 — HV 조기종료 + 적응 변이율.

[책임]
  - hv_early_stop_callback : 세대별 hypervolume 추세 정체 시 study.stop()
  - adaptive_mutation_callback : HV 개선 추세에 따라 child 변이율 자동 조절

둘 다 sampler 진행 상태만 보고 움직이는 일반 튜닝 기계장치다 — 학교 목적/졸업
판정과 무관하므로 objectives/graduate에서 분리했다.
"""
import numpy as np
import optuna
from optuna._hypervolume import compute_hypervolume  # optuna 내장 HV (private — 인터페이스 변동 가능)

from app.academy.training.classroom.nsga3.objectives import DIRECTIONS


def hv_early_stop_callback(population_size: int, window: int = 5,
                           stop: bool = True, scale_warmup: int = 6,
                           min_rel_improve: float = 1e-3):
    """세대별 hypervolume 추세로 Pareto 수렴을 감지 → 정체하면 study.stop().

    HV는 "front가 더 커지나"만 보는 단조 증가 추세 지수다 — 옛 1.0 캡은 없다.
      - 첫 scale_warmup 세대: 목적값 모으기만 (스케일 박스 확보), HV 미계산.
      - 이후: 6세대 누적 min/max로 min-max 스케일(축 단위 통일: 잔고 ~백만 vs turnover ~0.04)
        → optuna 내장 HV. 좋은 쪽은 무제한이고, ref보다 나쁜 쪽만 HV 유효성을 위해 컷한다.
      - window 이동평균이 min_rel_improve 미만으로 정체하면 수렴으로 보고 멈춘다.
    """
    sign = np.array([-1.0 if d == "maximize" else 1.0 for d in DIRECTIONS])
    ref = np.ones(len(DIRECTIONS))   # 스케일 nadir(6세대 worst 코너) = HV 기준점
    st = {"lo": None, "hi": None, "hv": [], "trend": [],
          "best_ma": -1.0, "stale": 0, "n": 0, "stopped": False}

    def cb(study: optuna.Study, trial) -> None:
        st["n"] += 1
        n = st["n"]
        if st["lo"] is None:
            # 워밍업: scale_warmup 세대(=population_size×scale_warmup trial) 누적 min/max로 스케일 고정
            if n < scale_warmup * population_size:
                return
            allv = np.array([t.values for t in study.trials
                             if t.values is not None]) * sign
            if len(allv) == 0:
                return
            st["lo"], hi = allv.min(0), allv.max(0)
            st["hi"] = np.where(hi > st["lo"], hi, st["lo"] + 1e-9)
            return
        if n % population_size:          # 세대 경계에서만 측정·기록 (현업 관행: 세대 단위)
            return
        front = np.array([t.values for t in study.best_trials
                          if t.values is not None])
        if len(front) == 0:
            return
        scaled = (front * sign - st["lo"]) / (st["hi"] - st["lo"])  # 좋은 쪽 무제한(no clip)
        scaled = np.minimum(scaled, ref)                # 나쁜 쪽만 ref로 컷(HV 유효성)
        hv = float(compute_hypervolume(scaled, ref))    # 1.0 캡 없음 — front 좋아질수록 증가
        st["hv"].append(hv)
        # MA(window) 평활: 세대별 노이즈를 눌러 수렴(평탄)이 눈에 보이게 한다.
        # 정체 판정·기록 트렌드 모두 이 평활값을 쓴다 (raw는 함께 보존).
        ma = float(np.mean(st["hv"][-window:]))         # window 덜 차면 부분 평균
        st["trend"].append([trial.number, round(hv, 6), round(ma, 6)])
        study.set_user_attr("hv_trend", st["trend"])    # DB 영속: 세대별 [trial, raw, ma]
        cb.hv = list(st["hv"])
        if len(st["hv"]) < window:
            return
        if ma > st["best_ma"] * (1 + min_rel_improve):  # 평활 HV가 의미있게 오르면 정체 카운터 리셋
            st["best_ma"] = ma
            st["stale"] = 0
            return
        st["stale"] += 1                                 # 변화 없는 세대 누적
        if stop and st["stale"] >= window:               # 5세대 연속 변화 없으면 stop
            print(f"  [early-stop] HV MA({window}) {window}세대 연속 변화 없음 → stop "
                  f"at {st['n']} trials (HV_ma {ma:.4f})")
            st["stopped"] = True
            cb.stopped = True
            study.stop()

    cb.hv = []
    cb.trend = st["trend"]   # 동일 리스트 참조 — engine이 마지막에 한 번 더 flush
    cb.stopped = False
    return cb


def adaptive_mutation_callback(sampler, hv_cb, n_params: int,
                               population_size: int, window: int = 3,
                               up: float = 1.5, down: float = 0.85,
                               hi: float = 0.5):
    lo = 1.0 / n_params
    state = {"best_ma": -1.0, "current": lo * 2, "n": 0}
    try:
        sampler._child_generation_strategy._mutation_prob = state["current"]
    except AttributeError:
        print("  [adaptive-mut] sampler path changed; disabled")
        cb_noop = lambda *args, **kwargs: None
        cb_noop.history = []
        return cb_noop

    def cb(_study, _trial):
        state["n"] += 1
        if state["n"] % population_size:
            return
        hv = list(hv_cb.hv)
        if len(hv) < window:
            return
        ma = sum(hv[-window:]) / window
        improved = ma > state["best_ma"] + 1e-9
        factor = down if improved else up
        new_val = max(lo, min(hi, state["current"] * factor))
        state["current"] = new_val
        if improved:
            state["best_ma"] = ma
        sampler._child_generation_strategy._mutation_prob = new_val
        cb.history.append({"gen": len(hv), "hv": round(hv[-1], 4),
                           "ma": round(ma, 4), "improved": improved,
                           "mut_prob": round(new_val, 4)})

    cb.history = []
    return cb
