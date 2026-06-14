"""NSGA-III 학교 엔진 — 합성장 3목적 최적화.

[왜 학교 전용인가]
학교 밖에서 따로 NSGA-III를 돌리지 않는다. 6체육관/5국면 NSGA는 폐기했고,
이 파일이 학교 2차 교육의 다목적 엔진이다.

[학교 목적]
  maximize  평균 누적자산
  maximize  최악 누적자산
  minimize  turnover

단일목적(TPE/CMA-ES/GP)은 누적자산 합 1등을 찾고, 이 모듈은 평균도 좋고 최악장도
버티며 매매도 덜 하는 트레이더를 찾는다. 성실이/어플삭제맨 비교는 objective가
아니라 졸업 필터에서 본다.
"""
import secrets

import numpy as np
import optuna

from app.academy.curriculum import prepare_academy_data
from app.academy.exam.grade import decode_params
from app.pocket.battle import _score_position, fight_dca, terminal_balance
from app.pocket.signals import SIGNAL_NAMES, combine_positions, positions_with_params
from app.world.data_loader import LoadedGym

OBJECTIVE_NAMES = ["mean_balance", "worst_balance", "turnover"]
DIRECTIONS = ["maximize", "maximize", "minimize"]
SEED_KRW = 1_000_000


def roll_seed() -> int:
    """학교 주사위 1개. 결과는 study.user_attrs에 기록해 재현한다."""
    return secrets.randbelow(1_000_000_000)


def suggest_candidate(trial: optuna.Trial) -> tuple[list[float], dict]:
    """학교 NSGA-III는 v2 기본처럼 가중치만 탐색한다."""
    params = {f"w_{g}": trial.suggest_float(f"w_{g}", 0.0, 1.0)
              for g in SIGNAL_NAMES}
    return decode_params(params)


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
    """학교용 3목적 raw 지표. 체육관 이름/국면 키에 의존하지 않는다."""
    results = _candidate_results(weights, params, loaded_gyms, base_positions)
    balances = [terminal_balance(r, seed_krw) for r in results]
    return {
        "mean_balance": sum(balances) / len(balances),
        "worst_balance": min(balances),
        "turnover": sum(r.turnover for r in results) / len(results),
        "balances": balances,
    }


def academy_metrics(values: list[float], seed_krw: int = SEED_KRW) -> dict:
    """OBJECTIVE_NAMES와 values를 묶어 졸업생 메타데이터로 변환한다."""
    obj = dict(zip(OBJECTIVE_NAMES, values))
    return {
        "mean_balance": obj["mean_balance"],
        "worst_balance": obj["worst_balance"],
        "turnover": obj["turnover"],
        "score": obj["mean_balance"] / seed_krw - 1.0,
    }


def _buy_hold_balance(loaded: LoadedGym, seed_krw: int = SEED_KRW) -> int:
    prices = loaded.prices.loc[loaded.gym.start:loaded.gym.end]
    rets = prices.pct_change().dropna()
    if len(rets) == 0:
        return seed_krw
    return int(seed_krw * float((1 + rets).cumprod().iloc[-1]))


def _baseline_summary(loaded_gyms: list[LoadedGym], dca: dict,
                      seed_krw: int = SEED_KRW) -> dict:
    dca_balances = [terminal_balance(dca[lg.gym.name], seed_krw)
                    for lg in loaded_gyms]
    bh_balances = [_buy_hold_balance(lg, seed_krw) for lg in loaded_gyms]
    return {
        "dca_mean": sum(dca_balances) / len(dca_balances),
        "dca_worst": min(dca_balances),
        "bh_mean": sum(bh_balances) / len(bh_balances),
        "bh_worst": min(bh_balances),
    }


def make_objective(loaded_gyms: list[LoadedGym]):
    base_positions = {lg.gym.name: positions_with_params(lg.prices)
                      for lg in loaded_gyms}

    def objective(trial: optuna.Trial):
        weights, params = suggest_candidate(trial)
        obj = evaluate_objectives(weights, params, loaded_gyms, base_positions)
        return obj["mean_balance"], obj["worst_balance"], obj["turnover"]
    return objective


def prepare_data(n_gyms: int = 20, seed: int | None = None
                 ) -> tuple[list[LoadedGym], dict]:
    """학교 합성장 + 성실이 기준선. seed=None이면 매번 다른 학기."""
    loaded_gyms = prepare_academy_data(n_gyms=n_gyms, seed=seed)[0]
    dca = {lg.gym.name: fight_dca(lg) for lg in loaded_gyms}
    return loaded_gyms, dca


def hv_early_stop_callback(population_size: int, window: int = 5,
                           n_mc: int = 4096, seed: int = 0,
                           stop: bool = True):
    sign = np.array([-1.0 if d == "maximize" else 1.0 for d in DIRECTIONS])
    rng = np.random.default_rng(seed)
    n_obj = len(DIRECTIONS)
    st = {"lo": None, "hi": None, "mc": rng.random((n_mc, n_obj)),
          "hv": [], "best_ma": -1.0, "n": 0, "stopped": False}

    def cb(study: optuna.Study, _trial) -> None:
        st["n"] += 1
        if st["n"] % population_size:
            return
        raw = np.array([t.values for t in study.trials
                        if t.values is not None])
        if len(raw) == 0:
            return
        vals = raw * sign
        if st["lo"] is None:
            st["lo"], hi = vals.min(0), vals.max(0)
            st["hi"] = np.where(hi > st["lo"], hi, st["lo"] + 1e-9)
            return
        norm = np.clip((vals - st["lo"]) / (st["hi"] - st["lo"]),
                       0.0, 1.0)
        keep = []
        for i in range(len(norm)):
            dom = (norm <= norm[i]).all(1) & (norm < norm[i]).any(1)
            dom[i] = False
            if not dom.any():
                keep.append(i)
        dominated = np.zeros(len(st["mc"]), dtype=bool)
        for p in norm[keep]:
            dominated |= (st["mc"] >= p).all(1)
        st["hv"].append(float(dominated.mean()))
        cb.hv = list(st["hv"])
        if len(st["hv"]) < window:
            return
        ma = float(np.mean(st["hv"][-window:]))
        if ma > st["best_ma"] + 1e-12:
            st["best_ma"] = ma
            return
        if stop:
            print(f"  [early-stop] HV MA({window}) stale at "
                  f"{st['n']} trials (HV {st['hv'][-1]:.4f})")
            st["stopped"] = True
            cb.stopped = True
            study.stop()

    cb.hv = []
    cb.stopped = False
    return cb


def adaptive_mutation_callback(sampler, hv_cb, n_params: int,
                               population_size: int, window: int = 3,
                               up: float = 1.5, down: float = 0.85,
                               hi: float = 0.5):
    lo = 1.0 / n_params
    state = {"best_ma": -1.0, "current": lo * 2, "n": 0}
    try:
        sampler._child_generation_strategy._mutation_prob = state["current"]
    except AttributeError:
        print("  [adaptive-mut] sampler path changed; disabled")
        cb_noop = lambda *args, **kwargs: None
        cb_noop.history = []
        return cb_noop

    def cb(_study, _trial):
        state["n"] += 1
        if state["n"] % population_size:
            return
        hv = list(hv_cb.hv)
        if len(hv) < window:
            return
        ma = sum(hv[-window:]) / window
        improved = ma > state["best_ma"] + 1e-9
        factor = down if improved else up
        new_val = max(lo, min(hi, state["current"] * factor))
        state["current"] = new_val
        if improved:
            state["best_ma"] = ma
        sampler._child_generation_strategy._mutation_prob = new_val
        cb.history.append({"gen": len(hv), "hv": round(hv[-1], 4),
                           "ma": round(ma, 4), "improved": improved,
                           "mut_prob": round(new_val, 4)})

    cb.history = []
    return cb


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
    tune_params/early_stop_window/adaptive_mutation은 구 6체육관 NSGA 호출 호환용으로
    받기만 하고 학교 엔진에서는 쓰지 않는다.
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
            seed=seed or 0, stop=bool(early_stop_window))
        callbacks.append(hv_cb)

    mut_cb = None
    if adaptive_mutation:
        mut_cb = adaptive_mutation_callback(
            sampler, hv_cb, len(SIGNAL_NAMES), population_size, window=3)
        callbacks.append(mut_cb)

    study.optimize(make_objective(loaded_gyms), n_trials=n_trials,
                   callbacks=callbacks or None)
    return study, loaded_gyms, dca, hv_cb, mut_cb


def summarize_front(study, loaded_gyms: list[LoadedGym] | None = None,
                    dca: dict | None = None,
                    tolerance: float | None = None,
                    turnover_cap: float = 0.10,
                    bh_mean_floor: float = 0.90,
                    seed_krw: int = SEED_KRW) -> dict:
    """학교 front 졸업 후보 요약.

    tolerance는 구 6체육관 NSGA 호출 호환용으로 받기만 한다.

    졸업 필터:
      ① 평균 잔고가 성실이 평균보다 큼
      ② 최악 잔고가 돼지저금통보다 큼
      ③ 평균 잔고가 어플삭제맨 평균의 bh_mean_floor 이상
      ④ turnover cap 이하
    """
    if loaded_gyms is None or dca is None:
        loaded_gyms, dca = prepare_data(
            seed=study.user_attrs.get("academy_seed"))
    baselines = _baseline_summary(loaded_gyms, dca, seed_krw)
    front = []
    for t in study.best_trials:
        row = {"number": t.number, "values": list(t.values),
               "params": dict(t.params)}
        row["academy"] = academy_metrics(row["values"], seed_krw)
        row["graduated"] = (
            row["academy"]["mean_balance"] > baselines["dca_mean"]
            and row["academy"]["worst_balance"] > seed_krw
            and row["academy"]["mean_balance"] >= baselines["bh_mean"] * bh_mean_floor
            and row["academy"]["turnover"] <= turnover_cap
        )
        front.append(row)

    passed = [r for r in front if r["graduated"]]

    labels = {}
    if passed:
        labels["Rich"] = max(passed, key=lambda r: r["academy"]["mean_balance"])
        labels["Sturdy"] = max(passed, key=lambda r: r["academy"]["worst_balance"])
        labels["Low-turnover"] = min(passed, key=lambda r: r["academy"]["turnover"])

    return {
        "front_size": len(front),
        "front": front,
        "passed": passed,
        "labels": labels,
        "baselines": baselines,
        "turnover_cap": turnover_cap,
        "bh_mean_floor": bh_mean_floor,
    }
