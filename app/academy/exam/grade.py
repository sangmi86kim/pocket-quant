"""시험 채점 — 수험생 1명(가중치+시그널 파라미터)을 체육관에 응시시키고 잔고로 환산.

[책임]
- `evaluate_balances` : 표시·판정용 잔고 — {체육관이름: {"strat": 원, "dca": 원}}.

트라이얼 파라미터 디코딩(decode_params)은 채점이 아니라 코덱이라 `training/candidate.py`로
분리했다 — 채점 함수는 이미 디코드된 (weights, params)를 받는다.

[실행 모델]
`_score_position`을 거친다 → 0.1% 과금 등 채점 규칙 일관.
"""
from app.pocket.battle import _score_position, terminal_balance
from app.pocket.signals import combine_positions, positions_with_params
from app.world.data_loader import LoadedGym


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
