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

    HV는 "front가 더 커지나"만 보는 단조 증가 추세 지수다 — 천장(1.0) 없다.
      - 첫 scale_warmup 세대: 목적값 모으기만 (스케일 박스 확보), HV 미계산.
      - 이후: 6세대 누적 min/max로 min-max 스케일(축 단위 통일: 잔고 ~백만 vs turnover ~0.04)
        → optuna 내장 HV. 개선이 스케일 0 미만으로 가도 clip하지 않아 HV가 계속 큰다.
      - window 이동평균이 min_rel_improve 미만으로 정체하면 수렴으로 보고 멈춘다.
    """
    sign = np.array([-1.0 if d == "maximize" else 1.0 for d in DIRECTIONS])
    ref = np.ones(len(DIRECTIONS))   # 스케일 nadir(6세대 worst 코너) = HV 기준점
    st = {"lo": None, "hi": None, "hv": [], "best_ma": -1.0, "n": 0, "stopped": False}

    def cb(study: optuna.Study, _trial) -> None:
        st["n"] += 1
        if st["n"] % population_size:
            return
        gen = st["n"] // population_size
        if gen < scale_warmup:
            return
        if st["lo"] is None:
            # 6세대 누적 전체(=population_size×scale_warmup trial)의 min/max로 스케일 고정
            allv = np.array([t.values for t in study.trials
                             if t.values is not None]) * sign
            if len(allv) == 0:
                return
            st["lo"], hi = allv.min(0), allv.max(0)
            st["hi"] = np.where(hi > st["lo"], hi, st["lo"] + 1e-9)
            return
        front = np.array([t.values for t in study.best_trials
                          if t.values is not None])
        if len(front) == 0:
            return
        scaled = (front * sign - st["lo"]) / (st["hi"] - st["lo"])  # 좋은 쪽 무제한(no clip)
        scaled = np.minimum(scaled, ref)                # 나쁜 쪽만 ref로 컷(HV 유효성)
        hv = float(compute_hypervolume(scaled, ref))    # 천장 없음 — front 좋아질수록 증가
        st["hv"].append(hv)
        cb.hv = list(st["hv"])
        if len(st["hv"]) < window:
            return
        ma = float(np.mean(st["hv"][-window:]))
        if ma > st["best_ma"] * (1 + min_rel_improve):  # 추세가 더 오르면 계속
            st["best_ma"] = ma
            return
        if stop:
            print(f"  [early-stop] HV MA({window}) 정체 at "
                  f"{st['n']} trials (HV {st['hv'][-1]:.4f})")
            st["stopped"] = True
            cb.stopped = True
            study.stop()

    cb.hv = []
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
