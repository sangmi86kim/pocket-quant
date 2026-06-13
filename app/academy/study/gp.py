"""GP(Gaussian Process) 단일목적 엔진 — TPE/CMA-ES와 같은 인터페이스.

[샘플러 3종 정리]
- `tpe`    : Tree-structured Parzen Estimator (Bayesian, 시드 안정성·빠른 수렴)
- `cma_es` : Covariance Matrix Adaptation ES (연속 공간 강함, multi-modal 약함)
- `gp`     : Gaussian Process Bayesian Optimization (적은 trial로 효율 — sample 1개당
             O(n³) 커널 비용 때문에 trials 200~300이 sweet spot, 그 이상은 후반부 폭주)

같은 목적함수(잔고 합 max) + 같은 결정변수(가중치 ALL_GENES 차원). decode/evaluate는
nsga3에서 재사용. service에서 갈아끼울 수 있게 시그니처는 tpe/cma_es와 통일.

[v1.x] CMA-ES sweep이 ±0.53% 수렴 양호 직전 + 코사인 0.951(multi-modal 가능성 잔존).
GP는 surrogate 모델로 적은 trial에서 답을 짚어내는 데 강해 비교 의의 있음.
"""
from __future__ import annotations

from typing import Callable

import optuna

from app.backend.data_io.data import LoadedGym, load_gyms
from app.backend.genes.signals import ALL_GENES
from app.backend.market.gym import all_gyms
from app.backend.engine.battle import fight_dca
from app.academy.study.nsga3 import decode_params, evaluate_balances

SEED_KRW = 1_000_000


def _objective(trial: optuna.Trial, loaded_gyms: list[LoadedGym], dca: dict) -> float:
    """tpe._objective와 동일 — sampler만 다르고 평가 경로는 같다 (공정한 비교)."""
    for g in ALL_GENES:
        trial.suggest_float(f"w_{g}", 0.0, 1.0)
    weights, sig_params = decode_params(trial.params)
    bals = evaluate_balances(weights, sig_params, loaded_gyms, dca, seed_krw=SEED_KRW)
    return sum(b["strat"] for b in bals.values())


def prepare_data() -> tuple[list[LoadedGym], dict]:
    """tpe/cma_es.prepare_data와 동일 — sweep용 한 번 준비/N회 재사용."""
    loaded_gyms = load_gyms(all_gyms())
    dca = {lg.gym.name: fight_dca(lg) for lg in loaded_gyms}
    return loaded_gyms, dca


def run_study(
    trials: int,
    seed: int | None = None,
    storage: str | None = None,
    study_name: str = "gp_single_obj",
    on_progress: Callable[[int, int, float], None] | None = None,
    loaded_gyms: list[LoadedGym] | None = None,
    dca: dict | None = None,
    extra_callbacks: list | None = None,
) -> tuple[optuna.Study, list[LoadedGym], dict]:
    """GP 단일목적 탐색. tpe/cma_es와 같은 시그니처."""
    if loaded_gyms is None or dca is None:
        loaded_gyms, dca = prepare_data()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # deterministic_objective=True — yfinance 캐시 백테스트는 X 같으면 Y 같음
    # (실데이터+결정론적 채점). GP가 노이즈 없는 가정으로 더 단단한 surrogate를 만든다.
    sampler = optuna.samplers.GPSampler(seed=seed, deterministic_objective=True)
    if storage is None:
        study = optuna.create_study(direction="maximize", sampler=sampler)
    else:
        study = optuna.create_study(
            direction="maximize", sampler=sampler,
            storage=storage, study_name=study_name, load_if_exists=False,
        )

    done = len(study.trials)
    remaining = max(0, trials - done)

    callbacks: list = []
    if on_progress is not None:
        def _cb(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            on_progress(trial.number + 1, trials, study_.best_value)
        callbacks.append(_cb)
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    study.optimize(
        lambda t: _objective(t, loaded_gyms, dca),
        n_trials=remaining, callbacks=callbacks or None,
    )
    return study, loaded_gyms, dca


def champion_balances(
    study: optuna.Study, loaded_gyms: list[LoadedGym], dca: dict,
) -> tuple[list[float], dict, dict]:
    """tpe/cma_es.champion_balances와 동일. 1등 trial의 (weights, 체육관별 잔고, 요약)."""
    best = study.best_trial
    weights, sig_params = decode_params(best.params)
    bals = evaluate_balances(weights, sig_params, loaded_gyms, dca, seed_krw=SEED_KRW)
    summary = {
        "trial": best.number,
        "balance_sum": best.value,
        "weights": weights,
        "per_gym": bals,
    }
    return weights, bals, summary
