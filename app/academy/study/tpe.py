"""TPE 단일목적 엔진 — 6체육관 × 시드 100만원 → 종료 잔고 합 max.

[nsga3.py와의 관계]
- `nsga3` : 다목적(6목적 score_vs_dca + turnover) → Pareto front 라인업
- `tpe`   : 단일목적(잔고 합)                       → 단일 챔피언

가설(experiment/single-obj 브랜치): 평행세계 ②가 사실상 "잔고 합 단일목적"이었고
거기서 TOP06이 1위였다. 인샘플 잔고 합 단일목적으로 직접 탐색하면 챔피언이 바뀌는지
확인 + 좋으면 nsga3 대신 갈아끼울 수 있게 정식 엔진으로 둔다.

[설계]
- 결정변수: 시그널 가중치 len(ALL_GENES)차원 (시그널 파라미터 동결 — v1 과적합 회피 정신 동일).
- 목적   : Σ_체육관 evaluate_balances[g]["strat"]  (100만원 시드 → 종료 잔고 합).
- Sampler: Optuna TPE (Bayesian — 단일목적 표준).
- decode/evaluate 헬퍼는 `nsga3`에서 재사용 (가중치 → 잔고 환산 동일 경로).

[주의 — 단일목적 함정]
worst-case가 안 보인다. "한 체육관에서 처참한데 합산 1위" 후보가 챔피언으로 부상할
수 있다 (다목적이 막아주던 함정 2·3 부활). 본 엔진 결과를 챔피언으로 채택하기 전
챔피언로드 ② 평행세계 토탈로 OOS 검증 필요.
"""

from typing import Callable

import optuna

from app.backend.data_io.data import LoadedGym, load_gyms
from app.backend.genes.signals import ALL_GENES
from app.backend.market.gym import all_gyms
from app.backend.engine.battle import fight_dca
from app.academy.study.nsga3 import decode_params, evaluate_balances

# 100만원 시드 — sweep_seeds·hall_of_fame과 동일 단위(만원 환산은 표시 층에서).
SEED_KRW = 1_000_000


def _objective(trial: optuna.Trial, loaded_gyms: list[LoadedGym], dca: dict) -> float:
    """시그널 가중치 제시 → 6체육관 잔고 합 (만원 환산은 표시 층 책임)."""
    for g in ALL_GENES:
        trial.suggest_float(f"w_{g}", 0.0, 1.0)
    weights, sig_params = decode_params(trial.params)
    bals = evaluate_balances(weights, sig_params, loaded_gyms, dca, seed_krw=SEED_KRW)
    return sum(b["strat"] for b in bals.values())


def prepare_data() -> tuple[list[LoadedGym], dict]:
    """체육관 가격 로딩 + 성실이(DCA) 기준선 산출. 시드와 무관 — 5시드 sweep 등에선
    바깥에서 한 번만 만들고 `run_study`에 넘기면 yfinance/fight_dca 중복 호출 제거."""
    loaded_gyms = load_gyms(all_gyms())
    dca = {lg.gym.name: fight_dca(lg) for lg in loaded_gyms}
    return loaded_gyms, dca


def run_study(
    trials: int,
    seed: int | None = None,
    storage: str | None = None,
    study_name: str = "tpe_single_obj",
    on_progress: Callable[[int, int, float], None] | None = None,
    loaded_gyms: list[LoadedGym] | None = None,
    dca: dict | None = None,
    extra_callbacks: list | None = None,
) -> tuple[optuna.Study, list[LoadedGym], dict]:
    """TPE 단일목적 탐색.

    `nsga3.run_study`와 같은 패턴 — main path에서 갈아끼우기 쉽게 시그니처 통일.
    storage 사용 시 중단/재개 가능, trials는 총 목표 수 (재개 시 모자란 만큼만).
    on_progress(done, total, best_value)는 매 트라이얼 후 호출 (콘솔 진행 출력 훅).
    loaded_gyms/dca를 미리 만들어 주입하면 캐시 hit 데이터를 재호출하지 않는다.
    """
    if loaded_gyms is None or dca is None:
        loaded_gyms, dca = prepare_data()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    if storage is None:
        study = optuna.create_study(direction="maximize", sampler=sampler)
    else:
        study = optuna.create_study(
            direction="maximize", sampler=sampler,
            # load_if_exists=False — 같은 study_name 충돌 시 즉시 에러로 차단.
            # storage는 시즌 임시 영역, hall_of_fame.md 흡수 후 db 폐기.
            storage=storage, study_name=study_name, load_if_exists=False,
        )

    # 재개 시 추가 분만 실행 — nsga3.run_study와 동일 의미론.
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
    """1등 trial의 (가중치, 체육관별 잔고 dict, 합계 dict) 추출.

    표시 층(어댑터)이 그대로 출력에 쓸 수 있는 형태."""
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
