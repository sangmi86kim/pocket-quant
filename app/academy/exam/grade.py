"""시험 채점 — 수험생 1명(가중치+시그널 파라미터)을 6과목에 응시시키고 점수·잔고로 환산.

[책임]
- `decode_params`     : Optuna trial.params → (가중치, 시그널 파라미터). 인코딩 역함수.
- `evaluate_candidate`: 다목적 채점 — {국면키: score_vs_dca, "turnover": 일평균}.
- `evaluate_balances` : 표시·판정용 잔고 — {체육관이름: {"strat": 원, "dca": 원}}.

[왜 exam에 있나]
이전엔 `training/nsga3.py`에 같이 있었지만, sampler(NSGA-III/TPE/CMA-ES/GP)와
무관한 "시험 치는 행위"다. tpe/cma_es/gp가 `from ...nsga3 import evaluate_balances`
하던 어색한 의존선이 여기로 모이면서 사라진다.

[실행 모델]
모두 같은 `_score_position`을 거친다 → 0.1% 과금 등 채점 규칙 일관. score_vs_dca와
evaluate_balances가 동일 경로라야 "가중치 → 점수"와 "가중치 → 잔고"가 모순 없음.
"""
from app.academy.exam import gym_key
from app.pocket.battle import _score_position, score_vs_dca, terminal_balance
from app.pocket.signals import SIGNAL_NAMES, combine_positions, positions_with_params
from app.world.data_loader import LoadedGym


def decode_params(params: dict) -> tuple[list[float], dict]:
    """Optuna trial.params → (가중치, 시그널 파라미터). suggest_candidate의 역함수.

    가중치 전용 리그(v2) 트라이얼엔 w_* 만 있다 → 시그널 파라미터는 기본값({})."""
    weights = [params[f"w_{g}"] for g in SIGNAL_NAMES]
    if "VOL_CALM" not in params:
        return weights, {}
    sig = {k: params[k] for k in
           ("DD_LIMIT", "MA_WINDOW", "MOM_LOOKBACK", "RSI_OVERSOLD", "BB_K", "VOL_CALM")}
    sig["VOL_STRESSED"] = params["VOL_CALM"] + params["VOL_SPREAD"]
    return weights, sig


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
        out[gym_key(lg.gym.name)] = score_vs_dca(result, dca[lg.gym.name])
        turnovers.append(result.turnover)
    out["turnover"] = sum(turnovers) / len(turnovers)
    return out


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
