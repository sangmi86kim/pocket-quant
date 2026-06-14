"""CMA-ES 단일목적 엔진 — sampler 한 줄만 다르고 평가 경로는 _single_obj와 공유.

[메모]
- CMA-ES는 워밍업 단계(boundary 학습)에 N >= n_dim 트라이얼이 필요. n_startup_trials
  기본 = n_dim. 가중치 13차원이면 startup 13개 정도.
- NSGA-III 계열과 친숙 (사용자 본업 sampler 정신).
"""
import optuna

from app.academy.training._single_obj import (
    SEED_KRW,
    champion_balances,
    prepare_data,
    run_single_obj_study,
)

__all__ = ["SEED_KRW", "champion_balances", "prepare_data", "run_study"]


def _make_sampler(seed: int | None) -> optuna.samplers.BaseSampler:
    return optuna.samplers.CmaEsSampler(seed=seed, warn_independent_sampling=False)


def run_study(trials, seed=None, storage=None, study_name="cma_es_single_obj",
              on_progress=None, loaded_gyms=None, dca=None,
              extra_callbacks=None):
    """CMA-ES 단일목적 탐색. _single_obj.run_single_obj_study에 위임."""
    return run_single_obj_study(
        _make_sampler, trials, seed=seed, storage=storage, study_name=study_name,
        on_progress=on_progress, loaded_gyms=loaded_gyms, dca=dca,
        extra_callbacks=extra_callbacks,
    )
