"""
nsga3.py - Optuna NSGA-III 다목적 최적화 (설계: OPTIMIZATION.md 4절)

[문제 정식화]
  maximize  [ bear, rebound, crash_v, bull, chop ]   # 국면별 라이벌(DCA)전 점수
  minimize  turnover                                  # 일평균 매매 비율
     over   X = 시그널 가중치 6개 + 시그널 파라미터 7개

  bear = min(닷컴, 금융위기) — 하락 2체육관을 min으로 압축해 6목적 유지
  (7목적은 front가 너무 넓어짐 — 코덱스 제안 채택)

[결정변수 X]
  가중치 w_i ∈ [0,1]: 결합은 '기권 제외 가중평균'(combine_positions weights).
    분모에 Σw가 있어 비율만 의미 = 예산 제약 내장, "전부 최대" 퇴화 없음.
  파라미터: DD_LIMIT / MA_WINDOW / MOM_LOOKBACK / RSI_OVERSOLD / BB_K /
    VOL_CALM / VOL_SPREAD(STRESSED = CALM + SPREAD, 순서 보장).

[주의 — 돼지저금통와 front의 극단점]
  turnover minimize 목적이 있으므로 "아무것도 안 하기"(전 가중치≈0)가
  front의 한쪽 극단(턴오버 0)으로 반드시 살아남는다. 이건 다목적의 정상
  거동이고, 배포 후보는 summarize_front의 하드 필터 3종(전 국면 ≥ -tol ·
  턴오버 cap · 최악 MDD ≤ DCA)으로 거른다.

실행은 `app.service.run_nsga3(...)` 직접 호출 (tools/e2e.py가 한 예).
"""
import numpy as np
import optuna

from app.backend.genes.signals import ALL_GENES, combine_positions, positions_with_params
from app.backend.data_io.data import LoadedGym, load_gyms
from app.backend.market.gym import all_gyms
from app.backend.engine.battle import _score_position, fight_dca, score_vs_dca, terminal_balance

# 체육관 이름 → 목적함수 키 (이름이 바뀌면 여기만 맞추면 됨)
GYM_KEYS = {
    "닷컴": "dotcom", "금융위기": "gfc", "회복장": "rebound",
    "코로나": "crash_v", "상승장": "bull", "횡보장": "chop",
}
OBJECTIVE_NAMES = ["bear", "rebound", "crash_v", "bull", "chop", "turnover"]
DIRECTIONS = ["maximize"] * 5 + ["minimize"]


def _gym_key(gym_name: str) -> str:
    for token, key in GYM_KEYS.items():
        if token in gym_name:
            return key
    raise KeyError(f"[nsga3] 목적 키를 모르는 체육관: {gym_name!r} — GYM_KEYS에 추가 필요")


def evaluate_candidate(weights: list[float], params: dict,
                       loaded_gyms: list[LoadedGym], dca: dict,
                       base_positions: dict | None = None) -> dict:
    """후보 1개(가중치+파라미터)를 전 체육관에서 채점해
    {체육관키: score_vs_dca, "turnover": 일평균} 을 돌려준다.

    base_positions: {체육관이름: 포지션목록} — 가중치 전용 리그(v2)에선 시그널이
    트라이얼마다 동일하므로 미리 계산해 넘기면 가중 결합+채점만 남는다(대폭 가속)."""
    out, turnovers = {}, []
    for lg in loaded_gyms:
        positions = (base_positions[lg.gym.name] if base_positions is not None
                     else positions_with_params(lg.prices, params))
        position = combine_positions(positions, weights)
        result = _score_position(position, lg)          # 전략과 동일 실행 모델(0.1% 과금)
        out[_gym_key(lg.gym.name)] = score_vs_dca(result, dca[lg.gym.name])
        turnovers.append(result.turnover)
    out["turnover"] = sum(turnovers) / len(turnovers)
    return out


def suggest_candidate(trial: optuna.Trial,
                      tune_params: bool = False) -> tuple[list[float], dict]:
    """탐색공간 정의.

    [v2 리그 = 가중치 전용이 기본 (A안, 2026-06-11 사용자 결정)]
    v1 리그(가중치+파라미터 13차원)는 챔피언로드 관문 ①에서 전멸했다 —
    인샘플↔OOS 상관 -0.21, 유일 생존자는 무튜닝 기본값. 과적합 벡터가
    파라미터 탐색이었으므로 v2는 파라미터를 기본값에 고정하고 가중치 6개만
    탐색한다. tune_params=True는 나중에 고도화할 때를 위해 보존.
    """
    weights = [trial.suggest_float(f"w_{g}", 0.0, 1.0) for g in ALL_GENES]
    if not tune_params:
        return weights, {}                       # 시그널 파라미터 = 모듈 기본값
    vol_calm = trial.suggest_float("VOL_CALM", 0.005, 0.015)
    params = {
        "DD_LIMIT": trial.suggest_float("DD_LIMIT", 0.05, 0.25),
        "MA_WINDOW": trial.suggest_int("MA_WINDOW", 50, 250),
        "MOM_LOOKBACK": trial.suggest_int("MOM_LOOKBACK", 20, 120),
        "RSI_OVERSOLD": trial.suggest_int("RSI_OVERSOLD", 20, 40),
        "BB_K": trial.suggest_float("BB_K", 1.5, 2.5),
        "VOL_CALM": vol_calm,
        # STRESSED = CALM + SPREAD 로 샘플링해 calm < stressed 를 항상 보장
        "VOL_STRESSED": vol_calm + trial.suggest_float("VOL_SPREAD", 0.003, 0.020),
    }
    return weights, params


def evaluate_balances(weights: list[float], params: dict,
                      loaded_gyms: list[LoadedGym], dca: dict,
                      seed_krw: int = 1_000_000) -> dict:
    """후보의 체육관별 (전략 잔고, 성실이 잔고) — 표시·판정용 (옵티마이저 아님).

    같은 _score_position을 거치므로 score_vs_dca와 동일 실행 모델 (0.1% 과금 등).
    내부 결과는 단순 dict {체육관이름: {"strat": 원, "dca": 원}}."""
    out = {}
    for lg in loaded_gyms:
        positions = positions_with_params(lg.prices, params)
        position = combine_positions(positions, weights)
        result = _score_position(position, lg)
        out[lg.gym.name] = {"strat": terminal_balance(result, seed_krw),
                            "dca": terminal_balance(dca[lg.gym.name], seed_krw)}
    return out


def decode_params(params: dict) -> tuple[list[float], dict]:
    """Optuna trial.params → (가중치, 시그널 파라미터). suggest_candidate의 역함수.
    가중치 전용 리그(v2) 트라이얼엔 w_* 만 있다 → 시그널 파라미터는 기본값."""
    weights = [params[f"w_{g}"] for g in ALL_GENES]
    if "VOL_CALM" not in params:
        return weights, {}
    sig = {k: params[k] for k in
           ("DD_LIMIT", "MA_WINDOW", "MOM_LOOKBACK", "RSI_OVERSOLD", "BB_K", "VOL_CALM")}
    sig["VOL_STRESSED"] = params["VOL_CALM"] + params["VOL_SPREAD"]
    return weights, sig


def make_objective(loaded_gyms: list[LoadedGym], dca: dict, tune_params: bool = False):
    # 가중치 전용 리그: 시그널 포지션은 전 트라이얼 공통 → 체육관당 1번만 계산
    base_positions = (None if tune_params else
                      {lg.gym.name: positions_with_params(lg.prices) for lg in loaded_gyms})

    def objective(trial: optuna.Trial):
        weights, params = suggest_candidate(trial, tune_params)
        s = evaluate_candidate(weights, params, loaded_gyms, dca, base_positions)
        return (min(s["dotcom"], s["gfc"]),     # bear (압축)
                s["rebound"], s["crash_v"], s["bull"], s["chop"],
                s["turnover"])
    return objective


# ── 수렴 모니터링: 하이퍼볼륨 MA 얼리스탑 (TunePilotAI에서 이식, 06-13) ──
def hv_early_stop_callback(population_size: int, window: int = 5,
                           n_mc: int = 4096, seed: int = 0):
    """하이퍼볼륨 MA 기반 얼리스탑 콜백 — Optuna study에 붙이면 정체 시 self stop.

    스펙 (TunePilotAI/flicker에서 이식, 부호 처리만 추가):
      ① 1세대(첫 population) = 목적별 스케일 캘리브레이션 — min~max로 [0,1] 정규화.
      ② 세대 경계마다 파레토 프론트의 HV를 MC 추정 (고정 샘플 = 세대 간 비교 일관).
      ③ HV는 측정 노이즈로 들쭉날쭉 → window MA로 평활, 신고점 미갱신 시 study.stop().

    부호 처리: PocketQuant는 5목적 maximize + turnover minimize → maximize는 -1배해
    "낮을수록 좋다" 표준형으로 통일 (TunePilot 원본은 전부 minimize였음).

    리포트용으로 cb.hv 리스트(세대별 HV)와 cb.stopped 플래그를 노출한다.
    """
    sign = np.array([-1.0 if d == "maximize" else 1.0 for d in DIRECTIONS])
    rng = np.random.default_rng(seed)
    n_obj = len(DIRECTIONS)
    st = {"lo": None, "hi": None, "mc": rng.random((n_mc, n_obj)),
          "hv": [], "best_ma": -1.0, "n": 0, "stopped": False}

    def cb(study: optuna.Study, _trial) -> None:
        st["n"] += 1
        if st["n"] % population_size:           # 세대 경계에서만 평가
            return
        raw = np.array([t.values for t in study.trials if t.values])
        if len(raw) == 0:
            return
        vals = raw * sign                       # minimize 표준형
        if st["lo"] is None:                    # 1세대 = 스케일 캘리브레이션
            st["lo"], hi = vals.min(0), vals.max(0)
            st["hi"] = np.where(hi > st["lo"], hi, st["lo"] + 1e-9)
            return
        norm = np.clip((vals - st["lo"]) / (st["hi"] - st["lo"]), 0.0, 1.0)
        keep = []                               # 비지배 점 (파레토)
        for i in range(len(norm)):
            dom = (norm <= norm[i]).all(1) & (norm < norm[i]).any(1)
            dom[i] = False
            if not dom.any():
                keep.append(i)
        dominated = np.zeros(len(st["mc"]), dtype=bool)
        for p in norm[keep]:                    # MC 추정: 프론트가 지배하는 부피
            dominated |= (st["mc"] >= p).all(1)
        st["hv"].append(float(dominated.mean()))
        cb.hv = list(st["hv"])
        if len(st["hv"]) < window:
            return
        ma = float(np.mean(st["hv"][-window:]))
        if ma > st["best_ma"] + 1e-12:
            st["best_ma"] = ma
        else:
            print(f"  [early-stop] HV MA({window}) 정체 — "
                  f"{st['n']} trial에서 중단 (HV {st['hv'][-1]:.4f})")
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
    """HV MA 신호를 받아 NSGA-III의 mutation_prob을 자동 조정하는 콜백 (06-13).

    설계 (사용자 안: "적응형 mutation은 옵튜나 콜백으로 구현"):
      - 매 세대(population_size 단위) 경계마다 hv_cb의 HV 곡선을 본다.
      - HV MA(window)가 신고점이면 잘 가고 있음 → mutation 좁혀 수렴 가속 (× down).
      - 정체면 다양성 부족 → mutation 넓혀 탐색 강화 (× up).
      - 클램프: [1/n_params (Optuna 자동 기본값), hi] — hi 위는 무작위에 가까움.

    sampler._child_generation_strategy._mutation_prob 직접 갱신 (Optuna 4.x 경로).
    의존: hv_cb (hv_early_stop_callback의 인스턴스) — callbacks 리스트에서 hv_cb를
    먼저 등록해 cb.hv가 갱신된 상태로 이 콜백이 본다.
    리포트용 cb.history 노출 — (세대, HV, MA, 신고점 여부, mutation_prob)."""
    lo = 1.0 / n_params
    state = {"best_ma": -1.0, "current": lo * 2, "n": 0}    # 초기값 = 자동의 2배
    try:
        sampler._child_generation_strategy._mutation_prob = state["current"]
    except AttributeError:
        print("  [adaptive-mut] sampler 내부 경로 변경 — 적응형 mutation 비활성")
        cb_noop = lambda *args, **kwargs: None
        cb_noop.history = []
        return cb_noop

    def cb(study, _trial):
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


def _guard_search_space(study, tune_params: bool) -> None:
    """같은 study_name에 다른 탐색공간이 섞이는 사고 방지 (코덱스 리뷰 P2, 06-11).

    config.json에서 tune_params만 바꿔 같은 스터디를 재개하면 v1/v2 후보가
    한 front에 섞인다 — v1 과적합 전멸 전례가 있어 운영상 치명적. 새 스터디면
    현재 탐색공간을 user_attrs로 도장 찍고, 기존 스터디면 대조해 다르면 중단.
    도장 없는 구버전 스터디는 trial 파라미터 키로 공간을 추정한다."""
    expected = {"tune_params": tune_params, "genes": list(ALL_GENES),
                "objectives": OBJECTIVE_NAMES}
    stamped = study.user_attrs.get("search_space")
    if stamped is None and study.trials:
        stamped = {**expected,
                   "tune_params": "VOL_CALM" in study.trials[0].params}
    if stamped is not None and stamped != expected:
        raise RuntimeError(
            f"[nsga3] 스터디 {study.study_name!r}의 탐색공간이 현재 설정과 다름 — "
            f"섞이면 front가 오염된다.\n  스터디: {stamped}\n  현재  : {expected}\n"
            "  → config의 study_name을 새로 짓거나 tune_params를 스터디와 맞출 것.")
    if study.user_attrs.get("search_space") != expected:
        study.set_user_attr("search_space", expected)


def run_study(n_trials: int, seed: int | None = 42, storage: str | None = None,
              study_name: str = "nsga3_v2_weights", tune_params: bool = False,
              on_progress=None, population_size: int = 50,
              early_stop_window: int | None = None,
              adaptive_mutation: bool = False):
    """스터디 1회 실행. storage(sqlite URL)를 주면 중단/재개 가능.
    n_trials = '총 목표 trial 수' — 재개 시 모자란 만큼만 추가 실행한다
    (Optuna 원래 의미는 '추가 실행 수'라 예산 관리가 흔들렸음. 코덱스 리뷰 P2).
    on_progress(완료수, 목표수, front크기) — 진행 콜백 훅.
    population_size: NSGA-III 한 세대 크기 — 6목적엔 50~100 권장 (das-dennis
      ref point 수 대비). 기본 50 = 사용자 본업(5목적) 검증치.
    early_stop_window: None=끔, 정수면 HV MA(window) 정체 시 self stop
      (hv_early_stop_callback).
    adaptive_mutation: True면 HV 정체/개선 신호로 mutation_prob 자동 조정.
      hv_cb 필요 → early_stop_window=None이면 자동 생성(window=3).
    같은 study_name에 다른 탐색공간을 섞으면 _guard_search_space가 중단시킨다.
    반환: (study, loaded_gyms, dca, hv_cb, mut_cb) — 마지막 둘은 None 가능."""
    loaded_gyms = load_gyms(all_gyms())
    dca = {lg.gym.name: fight_dca(lg) for lg in loaded_gyms}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.NSGAIIISampler(
        seed=seed, population_size=population_size)
    study = optuna.create_study(
        directions=DIRECTIONS, sampler=sampler,
        storage=storage, study_name=study_name if storage else None,
        # 2026-06-13 운영 규칙: load_if_exists=False 고정 — 같은 study_name이 이미
        # 있으면 즉시 에러로 차단. storage는 시즌 임시 작업 영역, hall_of_fame.md에
        # 결과 흡수 후 sqlite db 폐기. study_name은 매 시즌/실험마다 새로.
        load_if_exists=False,
    )
    study.set_metric_names(OBJECTIVE_NAMES)
    if storage:
        _guard_search_space(study, tune_params)

    done = len(study.trials)
    remaining = max(0, n_trials - done)
    if done:
        print(f"  스터디 재개: 기존 {done} trial → 목표 {n_trials}까지 {remaining}개 추가")

    callbacks = []
    if on_progress:
        def _cb(st, _trial):
            n = len(st.trials)
            if n % 200 == 0 or n >= n_trials:
                on_progress(n, n_trials, len(st.best_trials))
        callbacks.append(_cb)

    # 적응형 mutation은 hv_cb 신호가 필요 → 미설정 시 내부용 hv_cb 자동 생성
    hv_cb = None
    if early_stop_window or adaptive_mutation:
        w = early_stop_window or 3
        hv_cb = hv_early_stop_callback(population_size, window=w, seed=seed or 0)
        # early_stop이 꺼져 있으면 stop은 무력화 (HV 곡선만 수집)
        if not early_stop_window:
            hv_cb._stop_disabled = True
            # 콜백 안의 study.stop 호출을 방어 — wrap
            inner = hv_cb
            def _hv_silent(study, trial, _orig=inner):
                # study.stop을 일시 차단
                _real_stop = study.stop
                study.stop = lambda: None
                try:
                    _orig(study, trial)
                finally:
                    study.stop = _real_stop
            _hv_silent.hv = inner.hv
            _hv_silent.stopped = inner.stopped
            callbacks.append(_hv_silent)
            # mut_cb는 inner의 cb.hv를 직접 참조해야 → wrap의 hv가 실시간 갱신되게 함
            class _Proxy:
                @property
                def hv(self_): return inner.hv
            hv_cb = _Proxy()
        else:
            callbacks.append(hv_cb)

    mut_cb = None
    if adaptive_mutation:
        n_params = len(ALL_GENES) + (7 if tune_params else 0)
        mut_cb = adaptive_mutation_callback(sampler, hv_cb, n_params,
                                             population_size, window=3)
        callbacks.append(mut_cb)

    if remaining:
        study.optimize(make_objective(loaded_gyms, dca, tune_params),
                       n_trials=remaining, callbacks=callbacks)
    return study, loaded_gyms, dca, hv_cb, mut_cb


# ── Pareto 후처리: 하드 필터 + 라벨 (OPTIMIZATION.md 4-5) ──────────
def reference_vector(loaded_gyms: list[LoadedGym], dca: dict) -> dict:
    """비교 기준: 현 단일목적 챔피언(VOL+REV_RSI+REV_BB, 동일가중, 기본 파라미터)."""
    weights = [1.0 if g in ("VOL", "REV_RSI", "REV_BB") else 0.0 for g in ALL_GENES]
    return evaluate_candidate(weights, {}, loaded_gyms, dca)


def summarize_front(study, tolerance: float = 0.05, turnover_cap: float = 0.10,
                    loaded_gyms: list[LoadedGym] | None = None,
                    dca: dict | None = None) -> dict:
    """front를 배포 후보로 거른다.

    하드 필터 2종:
      ① 전 국면 score ≥ -tolerance (실측: 전 국면 양수 후보는 0개라 tolerance 필수)
      ② 턴오버 ≤ cap (비용 민감도 0.2% FAIL 실측 근거)

    ※ MDD 하드필터는 06-13 사용자 결정으로 제거:
      "어차피 깨져도 안 팔면 그만이야". B&H 정신 — 낙폭 그 자체가 페널티
      되는 건 score_vs_dca의 0.4×낙폭개선 항에 이미 들어가 있고,
      체육관 6개 다목적이 위험을 자연 분담한다.

    라벨: Defensive(bear 최고) / Balanced(5국면 평균 최고) /
          Aggressive(rebound+bull 최고) / Low-turnover(필터 내 턴오버 최소).

    loaded_gyms/dca: 호출처가 이미 로드했으면 전달(재로드 방지), 없으면 여기서.
    """
    if loaded_gyms is None:
        loaded_gyms = load_gyms(all_gyms())
    if dca is None:
        dca = {lg.gym.name: fight_dca(lg) for lg in loaded_gyms}

    front = [{"number": t.number, "values": list(t.values), "params": dict(t.params)}
             for t in study.best_trials]
    for row in front:
        row["mean5"] = sum(row["values"][:5]) / 5
        row["min5"] = min(row["values"][:5])

    passed = [r for r in front
              if r["min5"] >= -tolerance and r["values"][5] <= turnover_cap]

    labels = {}
    if passed:
        labels["Defensive"] = max(passed, key=lambda r: r["values"][0])
        labels["Balanced"] = max(passed, key=lambda r: r["mean5"])
        labels["Aggressive"] = max(passed, key=lambda r: r["values"][1] + r["values"][3])
        labels["Low-turnover"] = min(passed, key=lambda r: r["values"][5])

    return {"front_size": len(front), "passed": passed, "labels": labels,
            "tolerance": tolerance, "turnover_cap": turnover_cap}
