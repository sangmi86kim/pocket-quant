"""NSGA-III 학교 목적함수 — 가중치 후보 → 2목적 raw 지표 + turnover 진단.

[책임]
  - 목적 상수(OBJECTIVE_NAMES/DIRECTIONS)와 학기 주사위(roll_seed)
  - 후보 제시(suggest_candidate) → 체육관 백테스트(_candidate_results)
  - 2목적 환산(evaluate_objectives) + 졸업생 메타(academy_metrics)
  - Optuna objective 클로저(make_objective)
  - 학교 합성장 준비(prepare_data)

체육관 이름/국면 키에 의존하지 않는다 — 학교는 합성장만 본다.
"""
import secrets

import numpy as np
import optuna

from app.academy.curriculum import prepare_academy_data
from app.academy.training.candidate import suggest_weights
from app.pocket.battle import _score_position, terminal_balance
from app.pocket.signals import combine_positions, positions_with_params
from app.world.data_loader import LoadedGym

OBJECTIVE_NAMES = ["median_balance", "worst_balance"]
DIRECTIONS = ["maximize", "maximize"]
SEED_KRW = 1_000_000


def roll_seed() -> int:
    """학교 주사위 1개. 결과는 study.user_attrs에 기록해 재현한다."""
    return secrets.randbelow(1_000_000_000)


def suggest_candidate(trial: optuna.Trial) -> tuple[list[float], dict]:
    """학교 NSGA-III는 v2 기본처럼 가중치만 탐색한다 (공용 코덱에 위임)."""
    return suggest_weights(trial)


def _candidate_results(weights: list[float], params: dict,
                       loaded_gyms: list[LoadedGym],
                       base_positions: dict | None = None) -> list:
    results = []
    for lg in loaded_gyms:
        positions = (base_positions[lg.gym.name] if base_positions is not None
                     else positions_with_params(lg.prices, params))
        pos = combine_positions(positions, weights)
        results.append(_score_position(pos, lg))
    return results


def evaluate_objectives(weights: list[float], params: dict,
                        loaded_gyms: list[LoadedGym],
                        base_positions: dict | None = None,
                        seed_krw: int = SEED_KRW) -> dict:
    """학교용 raw 지표. 체육관 이름/국면 키에 의존하지 않는다.

    1목적은 평균이 아니라 **중앙값(median)**이다 — 평균은 한 합성장 대박이 통째로
    끌어올려 '전형적 시장 실력'을 부풀린다(평균의 함정). median은 그 한 방에 안 휘둘린다.
    turnover는 돈 버는 목적이 아니라 운용 스펙이므로 목적함수에서 빼고 진단/필터로만 쓴다.
    """
    results = _candidate_results(weights, params, loaded_gyms, base_positions)
    balances = [terminal_balance(r, seed_krw) for r in results]
    return {
        "median_balance": float(np.median(balances)),
        "worst_balance": min(balances),
        "turnover": sum(r.turnover for r in results) / len(results),
        "balances": balances,
    }


def academy_metrics(values: list[float], seed_krw: int = SEED_KRW) -> dict:
    """OBJECTIVE_NAMES와 values를 묶어 졸업생 메타데이터로 변환한다."""
    obj = dict(zip(OBJECTIVE_NAMES, values))
    return {
        "median_balance": obj["median_balance"],
        "worst_balance": obj["worst_balance"],
        "score": obj["median_balance"] / seed_krw - 1.0,
    }


def make_objective(loaded_gyms: list[LoadedGym]):
    base_positions = {lg.gym.name: positions_with_params(lg.prices)
                      for lg in loaded_gyms}

    def objective(trial: optuna.Trial):
        weights, params = suggest_candidate(trial)
        obj = evaluate_objectives(weights, params, loaded_gyms, base_positions)
        trial.set_user_attr("academy_metrics", {
            "median_balance": obj["median_balance"],
            "worst_balance": obj["worst_balance"],
            "turnover": obj["turnover"],
            "score": obj["median_balance"] / SEED_KRW - 1.0,
            "balances": obj["balances"],
        })
        return obj["median_balance"], obj["worst_balance"]
    return objective


def prepare_data(n_gyms: int = 20, seed: int | None = None
                 ) -> tuple[list[LoadedGym], dict]:
    """학교 합성장 + 성실이 기준선. (gyms, dca) 생성은 curriculum에 위임."""
    if seed is None:
        seed = 42
    return prepare_academy_data(n_gyms=n_gyms, seed=seed)
