"""GP(Gaussian Process) 단일목적 엔진 — sampler 한 줄만 다르고 평가 경로는 _single_obj와 공유.

[메모]
GP는 sample 1개당 O(n³) 커널 비용 — trials 200~300이 sweet spot.
deterministic_objective=True: yfinance 캐시 백테스트는 X 같으면 Y 같다 (실데이터
+ 결정론적 채점). GP가 노이즈 없는 가정으로 더 단단한 surrogate를 만든다.
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
    return optuna.samplers.GPSampler(seed=seed, deterministic_objective=True)


def run_study(trials, seed=None, storage=None, study_name="gp_single_obj",
              on_progress=None, loaded_gyms=None, dca=None,
              extra_callbacks=None, early_stop=True,
              patience=None, min_delta_pct=None):
    """GP 단일목적 탐색. _single_obj.run_single_obj_study에 위임."""
    return run_single_obj_study(
        _make_sampler, trials, seed=seed, storage=storage, study_name=study_name,
        on_progress=on_progress, loaded_gyms=loaded_gyms, dca=dca,
        extra_callbacks=extra_callbacks, early_stop=early_stop,
        patience=patience, min_delta_pct=min_delta_pct,
    )
