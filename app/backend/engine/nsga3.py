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

실행 진입점은 service.run_nsga3 (config.json: mode="nsga3").
"""
import optuna

from ..genes.signals import ALL_GENES, combine_positions, positions_with_params
from ..market.data import LoadedGym, load_gyms
from ..market.gym import all_gyms
from .battle import _score_position, fight_dca, score_vs_dca

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
              on_progress=None):
    """스터디 1회 실행. storage(sqlite URL)를 주면 중단/재개 가능.
    n_trials = '총 목표 trial 수' — 재개 시 모자란 만큼만 추가 실행한다
    (Optuna 원래 의미는 '추가 실행 수'라 예산 관리가 흔들렸음. 코덱스 리뷰 P2).
    on_progress(완료수, 목표수, front크기) — 진행 콜백 훅.
    같은 study_name에 다른 탐색공간을 섞으면 _guard_search_space가 중단시킨다."""
    loaded_gyms = load_gyms(all_gyms())
    dca = {lg.gym.name: fight_dca(lg) for lg in loaded_gyms}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        directions=DIRECTIONS,
        sampler=optuna.samplers.NSGAIIISampler(seed=seed),
        storage=storage, study_name=study_name if storage else None,
        load_if_exists=bool(storage),
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

    if remaining:
        study.optimize(make_objective(loaded_gyms, dca, tune_params),
                       n_trials=remaining, callbacks=callbacks)
    return study, loaded_gyms, dca


# ── Pareto 후처리: 하드 필터 + 라벨 (OPTIMIZATION.md 4-5) ──────────
def reference_vector(loaded_gyms: list[LoadedGym], dca: dict) -> dict:
    """비교 기준: 현 단일목적 챔피언(VOL+REV_RSI+REV_BB, 동일가중, 기본 파라미터)."""
    weights = [1.0 if g in ("VOL", "REV_RSI", "REV_BB") else 0.0 for g in ALL_GENES]
    return evaluate_candidate(weights, {}, loaded_gyms, dca)


def _worst_mdd(weights: list[float], params: dict,
               loaded_gyms: list[LoadedGym]) -> float:
    """후보의 전 체육관 최악 MDD (음수, 가장 깊은 값)."""
    return min(
        _score_position(
            combine_positions(positions_with_params(lg.prices, params), weights),
            lg).max_drawdown
        for lg in loaded_gyms)


def summarize_front(study, tolerance: float = 0.05, turnover_cap: float = 0.10,
                    loaded_gyms: list[LoadedGym] | None = None,
                    dca: dict | None = None) -> dict:
    """front를 배포 후보로 거른다.

    하드 필터 3종 (OPTIMIZATION.md 4-5와 일치):
      ① 전 국면 score ≥ -tolerance (실측: 전 국면 양수 후보는 0개라 tolerance 필수)
      ② 턴오버 ≤ cap (비용 민감도 0.2% FAIL 실측 근거)
      ③ 최악 MDD ≤ DCA 최악 MDD — 문서에만 있다가 06-12 구현 (코덱스 리뷰 P2).
        score_vs_dca에 MDD가 40% 들어가도 수익/샤프 큰 후보가 깊은 낙폭을
        상쇄하고 통과할 수 있어서, "방어 오버레이" 해석을 게이트로 강제한다.
    라벨: Defensive(bear 최고) / Balanced(5국면 평균 최고) /
          Aggressive(rebound+bull 최고) / Low-turnover(필터 내 턴오버 최소).

    loaded_gyms/dca: 호출처가 이미 로드했으면 전달(재로드 방지), 없으면 여기서
    로드한다. MDD는 목적값에 없어서 후보별 재계산이 필요한데, front가 크면
    비싸므로 결과를 study 객체에 캐시한다 (같은 프로세스 내 반복 호출 대비).
    """
    if loaded_gyms is None:
        loaded_gyms = load_gyms(all_gyms())
    if dca is None:
        dca = {lg.gym.name: fight_dca(lg) for lg in loaded_gyms}
    dca_worst = min(r.max_drawdown for r in dca.values())

    mdd_cache = getattr(study, "_pq_mdd_cache", None)
    if mdd_cache is None:
        mdd_cache = {}
        study._pq_mdd_cache = mdd_cache

    front = [{"number": t.number, "values": list(t.values), "params": dict(t.params)}
             for t in study.best_trials]
    for row in front:
        row["mean5"] = sum(row["values"][:5]) / 5
        row["min5"] = min(row["values"][:5])
        if row["number"] not in mdd_cache:
            mdd_cache[row["number"]] = _worst_mdd(*decode_params(row["params"]),
                                                  loaded_gyms)
        row["worst_mdd"] = mdd_cache[row["number"]]

    passed = [r for r in front
              if r["min5"] >= -tolerance and r["values"][5] <= turnover_cap
              and r["worst_mdd"] >= dca_worst]   # MDD는 음수 — 클수록(얕을수록) 좋음

    labels = {}
    if passed:
        labels["Defensive"] = max(passed, key=lambda r: r["values"][0])
        labels["Balanced"] = max(passed, key=lambda r: r["mean5"])
        labels["Aggressive"] = max(passed, key=lambda r: r["values"][1] + r["values"][3])
        labels["Low-turnover"] = min(passed, key=lambda r: r["values"][5])

    return {"front_size": len(front), "passed": passed, "labels": labels,
            "tolerance": tolerance, "turnover_cap": turnover_cap,
            "dca_worst_mdd": dca_worst}
