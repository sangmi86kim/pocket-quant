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

[주의 — 잠만보와 front의 극단점]
  turnover minimize 목적이 있으므로 "아무것도 안 하기"(전 가중치≈0)가
  front의 한쪽 극단(턴오버 0)으로 반드시 살아남는다. 이건 다목적의 정상
  거동이고, 배포 후보는 summarize_front의 하드 필터(전 국면 ≥ -tol)로 거른다.

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
                       loaded_gyms: list[LoadedGym], dca: dict) -> dict:
    """후보 1개(가중치+파라미터)를 전 체육관에서 채점해
    {체육관키: score_vs_dca, "turnover": 일평균} 을 돌려준다."""
    out, turnovers = {}, []
    for lg in loaded_gyms:
        positions = positions_with_params(lg.prices, params)
        position = combine_positions(positions, weights)
        result = _score_position(position, lg)          # 전략과 동일 실행 모델(0.1% 과금)
        out[_gym_key(lg.gym.name)] = score_vs_dca(result, dca[lg.gym.name])
        turnovers.append(result.turnover)
    out["turnover"] = sum(turnovers) / len(turnovers)
    return out


def suggest_candidate(trial: optuna.Trial) -> tuple[list[float], dict]:
    """탐색공간 정의 — 첫 버전은 보수적으로 (OPTIMIZATION.md 4-3)."""
    weights = [trial.suggest_float(f"w_{g}", 0.0, 1.0) for g in ALL_GENES]
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


def make_objective(loaded_gyms: list[LoadedGym], dca: dict):
    def objective(trial: optuna.Trial):
        weights, params = suggest_candidate(trial)
        s = evaluate_candidate(weights, params, loaded_gyms, dca)
        return (min(s["dotcom"], s["gfc"]),     # bear (압축)
                s["rebound"], s["crash_v"], s["bull"], s["chop"],
                s["turnover"])
    return objective


def run_study(n_trials: int, seed: int | None = 42, storage: str | None = None,
              study_name: str = "nsga3_v1", on_progress=None):
    """스터디 1회 실행. storage(sqlite URL)를 주면 중단/재개 가능.
    on_progress(완료수, 전체수, front크기) — 진행 콜백 훅."""
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

    callbacks = []
    if on_progress:
        def _cb(st, _trial):
            n = len(st.trials)
            if n % 200 == 0 or n == n_trials:
                on_progress(n, n_trials, len(st.best_trials))
        callbacks.append(_cb)

    study.optimize(make_objective(loaded_gyms, dca), n_trials=n_trials,
                   callbacks=callbacks)
    return study, loaded_gyms, dca


# ── Pareto 후처리: 하드 필터 + 라벨 (OPTIMIZATION.md 4-5) ──────────
def reference_vector(loaded_gyms: list[LoadedGym], dca: dict) -> dict:
    """비교 기준: 현 단일목적 챔피언(VOL+REV_RSI+REV_BB, 동일가중, 기본 파라미터)."""
    weights = [1.0 if g in ("VOL", "REV_RSI", "REV_BB") else 0.0 for g in ALL_GENES]
    return evaluate_candidate(weights, {}, loaded_gyms, dca)


def summarize_front(study, tolerance: float = 0.05, turnover_cap: float = 0.10) -> dict:
    """front를 배포 후보로 거른다.

    하드 필터: 전 국면 score ≥ -tolerance (실측: 전 국면 양수 후보는 0개라
    tolerance 필수) + 턴오버 ≤ cap (비용 민감도 0.2% FAIL 실측 근거).
    라벨: Defensive(bear 최고) / Balanced(5국면 평균 최고) /
          Aggressive(rebound+bull 최고) / Low-turnover(필터 내 턴오버 최소).
    """
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
