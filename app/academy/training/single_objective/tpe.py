"""TPE 단일목적 엔진 — 시그널 가중치 → 6체육관 잔고 합 max.

[샘플러 3종 정리]
- `tpe`    : Tree-structured Parzen Estimator (Bayesian, 시드 안정성·빠른 수렴)
- `cma_es` : Covariance Matrix Adaptation ES (연속 공간 강함, multi-modal 약함)
- `gp`     : Gaussian Process Bayesian Optimization (적은 trial 효율)

세 엔진의 평가 경로(목적/데이터/스터디)는 `single_objective.engine`에 모여 있고, 여기선
sampler 한 줄만 정의한다. 공개 API(`run_study`/`champion_balances`/`prepare_data`)는
세 엔진 동일 시그니처라 service에서 갈아끼울 수 있다.

[주의 — 단일목적 함정]
worst-case가 안 보인다. "한 체육관에서 처참한데 합산 1위" 후보가 챔피언으로 부상할
수 있다 (다목적이 막아주던 함정 2·3 부활). 채택 전 챔피언로드 ② 평행세계 토탈로
OOS 검증 필요.
"""
import optuna

from app.academy.training.single_objective.engine import (
    SEED_KRW,
    champion_balances,
    prepare_data,
    run_single_obj_study,
)

__all__ = ["SEED_KRW", "champion_balances", "prepare_data", "run_study"]


def _make_sampler(seed: int | None) -> optuna.samplers.BaseSampler:
    return optuna.samplers.TPESampler(seed=seed)


def run_study(trials, seed=None, storage=None, study_name="tpe_single_obj",
              on_progress=None, loaded_gyms=None, dca=None,
              extra_callbacks=None, early_stop=True,
              patience=None, min_delta_pct=None):
    """TPE 단일목적 탐색. single_objective.engine에 위임."""
    return run_single_obj_study(
        _make_sampler, trials, seed=seed, storage=storage, study_name=study_name,
        on_progress=on_progress, loaded_gyms=loaded_gyms, dca=dca,
        extra_callbacks=extra_callbacks, early_stop=early_stop,
        patience=patience, min_delta_pct=min_delta_pct,
    )
