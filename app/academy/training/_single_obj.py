"""단일목적 엔진 공통 — TPE/CMA-ES/GP 공유 헬퍼.

[책임]
세 엔진은 sampler 인스턴스 1줄만 다르고, 나머지(목적 함수·데이터 준비·스터디
오케스트레이션·챔피언 잔고 환산)는 동일하다. 이 파일이 그 동일 경로를 한 곳에
모은다. 각 엔진(tpe/cma_es/gp)은 `_make_sampler(seed)`만 정의하고 `run_study`/
`champion_balances`/`prepare_data`를 그대로 위임한다.

[설계 — 사용자 안 "기능별로 분배"]
- 시험 채점 = `exam.grade.evaluate_balances` (sampler 무관)
- 스터디 실행 = 이 파일 (`_run_single_obj_study`)
- sampler 선택 = 각 엔진 파일 (10줄 안짝)

[왜 _ prefix]
모듈 외부에서 직접 import하지 말라는 신호 — 공개 API는 tpe/cma_es/gp.run_study 등.
"""
from typing import Callable

import optuna

from app.academy.exam import all_gyms
from app.academy.exam.grade import decode_params, evaluate_balances
from app.pocket.battle import fight_dca
from app.pocket.signals import ALL_GENES
from app.world.data_loader import LoadedGym, load_gyms

# 100만원 시드 — sweep_seeds·hall_of_fame과 동일 단위(만원 환산은 표시 층에서).
SEED_KRW = 1_000_000


def _objective(trial: optuna.Trial, loaded_gyms: list[LoadedGym], dca: dict) -> float:
    """시그널 가중치 제시 → 6체육관 잔고 합. sampler 무관(공정 비교)."""
    for g in ALL_GENES:
        trial.suggest_float(f"w_{g}", 0.0, 1.0)
    weights, sig_params = decode_params(trial.params)
    bals = evaluate_balances(weights, sig_params, loaded_gyms, dca, seed_krw=SEED_KRW)
    return sum(b["strat"] for b in bals.values())


def prepare_data() -> tuple[list[LoadedGym], dict]:
    """체육관 가격 로딩 + 성실이(DCA) 기준선 — 시드 sweep 등에서 한 번만 만들고 재사용."""
    loaded_gyms = load_gyms(all_gyms())
    dca = {lg.gym.name: fight_dca(lg) for lg in loaded_gyms}
    return loaded_gyms, dca


def run_single_obj_study(
    make_sampler: Callable[[int | None], optuna.samplers.BaseSampler],
    trials: int,
    seed: int | None = None,
    storage: str | None = None,
    study_name: str = "single_obj",
    on_progress: Callable[[int, int, float], None] | None = None,
    loaded_gyms: list[LoadedGym] | None = None,
    dca: dict | None = None,
    extra_callbacks: list | None = None,
) -> tuple[optuna.Study, list[LoadedGym], dict]:
    """단일목적 탐색 공통 경로.

    `make_sampler(seed)`는 sampler 인스턴스 1개를 반환하는 팩토리. 각 엔진 파일이
    이걸로 자기 sampler(TPE/CMA-ES/GP)를 꽂는다. storage 사용 시 중단/재개 가능,
    trials는 총 목표 수(재개 시 모자란 만큼만 추가 실행). on_progress(done,total,
    best_value)는 매 트라이얼 후 호출. loaded_gyms/dca를 미리 만들어 주입하면
    yfinance/fight_dca 중복 호출을 막는다.
    """
    if loaded_gyms is None or dca is None:
        loaded_gyms, dca = prepare_data()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = make_sampler(seed)
    if storage is None:
        study = optuna.create_study(direction="maximize", sampler=sampler)
    else:
        study = optuna.create_study(
            direction="maximize", sampler=sampler,
            # AGENTS.md §11 운영 규칙: 같은 study_name 충돌 시 즉시 에러로 차단.
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
    """1등 trial의 (가중치, 체육관별 잔고 dict, 요약 dict) — 표시·판정용."""
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
