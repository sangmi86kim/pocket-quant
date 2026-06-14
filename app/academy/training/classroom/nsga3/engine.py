"""학교 NSGA-III 실행기 — sampler 구성 + 콜백 배선 + study.optimize.

objective는 objectives.make_objective, 졸업 요약은 graduate.summarize_front,
튜닝 콜백은 callbacks가 맡는다. 이 파일은 그것들을 엮어 한 학기를 돌린다.
"""
import optuna

from app.academy.training.classroom.nsga3.callbacks import (
    adaptive_mutation_callback,
    hv_early_stop_callback,
)
from app.academy.training.classroom.nsga3.objectives import (
    DIRECTIONS,
    OBJECTIVE_NAMES,
    make_objective,
    prepare_data,
    roll_seed,
)
from app.pocket.signals import SIGNAL_NAMES
from app.world.data_loader import LoadedGym


def run_study(n_trials: int, seed: int | None = None,
              storage: str | None = None,
              study_name: str = "academy_nsga3",
              loaded_gyms: list[LoadedGym] | None = None,
              dca: dict | None = None,
              n_gyms: int = 20,
              academy_seed: int | None = None,
              population_size: int = 50,
              on_progress=None,
              tune_params: bool = False,
              early_stop_window: int | None = None,
              adaptive_mutation: bool = False):
    """학교 NSGA-III 실행.

    seed는 sampler 주사위, academy_seed는 합성장 주사위다. 둘 다 None이면 매번
    새로 던진다. 결과를 재현하려면 호출자가 두 seed를 기록해야 한다.

    early_stop_window는 HV 정체 시 조기 종료, adaptive_mutation은 변이율 자동 조절을
    켠다 — 둘 다 학교 엔진이 실제로 사용한다(study/sweep_seeds가 켬). tune_params만
    구 6체육관 NSGA 호출과의 시그니처 호환을 위해 받기만 하고 쓰지 않는다.
    """
    if seed is None:
        seed = roll_seed()
    if academy_seed is None:
        academy_seed = roll_seed()

    if loaded_gyms is None or dca is None:
        loaded_gyms, dca = prepare_data(n_gyms=n_gyms, seed=academy_seed)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.NSGAIIISampler(
        seed=seed, population_size=population_size)
    study = optuna.create_study(
        directions=DIRECTIONS, sampler=sampler,
        storage=storage, study_name=study_name if storage else None,
        load_if_exists=False,
    )
    study.set_metric_names(OBJECTIVE_NAMES)
    study.set_user_attr("academy_seed", academy_seed)
    study.set_user_attr("sampler_seed", seed)
    study.set_user_attr("objectives", OBJECTIVE_NAMES)

    callbacks = []
    if on_progress:
        def _cb(st, _trial):
            n = len(st.trials)
            if n % 200 == 0 or n >= n_trials:
                on_progress(n, n_trials, len(st.best_trials))
        callbacks.append(_cb)

    hv_cb = None
    if early_stop_window or adaptive_mutation:
        hv_cb = hv_early_stop_callback(
            population_size, window=early_stop_window or 3,
            stop=bool(early_stop_window))
        callbacks.append(hv_cb)

    mut_cb = None
    if adaptive_mutation:
        mut_cb = adaptive_mutation_callback(
            sampler, hv_cb, len(SIGNAL_NAMES), population_size, window=3)
        callbacks.append(mut_cb)

    study.optimize(make_objective(loaded_gyms), n_trials=n_trials,
                   callbacks=callbacks or None)
    return study, loaded_gyms, dca, hv_cb, mut_cb
