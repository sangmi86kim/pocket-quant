"""NSGA-III 튜닝 콜백 — HV 조기종료 + 적응 변이율.

[책임]
  - hv_early_stop_callback : 세대별 hypervolume(MC 근사) 정체 시 study.stop()
  - adaptive_mutation_callback : HV 개선 추세에 따라 child 변이율 자동 조절

둘 다 sampler 진행 상태만 보고 움직이는 일반 튜닝 기계장치다 — 학교 목적/졸업
판정과 무관하므로 objectives/graduate에서 분리했다.
"""
import numpy as np
import optuna

from app.academy.training.classroom.nsga3.objectives import DIRECTIONS


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
