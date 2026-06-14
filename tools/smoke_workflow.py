"""
smoke_workflow.py - 짧은 전체 워크플로우 동작 확인

긴 e2e 대신 "연결이 살아 있는지"만 보는 스모크다.

순서:
  ① 아카데미: 합성 train/validation 생성 → TPE/CMA-ES/GP 짧은 trial
  ② 체육관: 실데이터 6체육관 → TPE/CMA-ES/GP 짧은 trial
  ③ 리그: NSGA-III 짧은 trial

목적:
  - 아카데미 합성장 attrs가 실제 study 엔진까지 들어가는지
  - 기존 실체육관 학습 경로가 깨지지 않았는지
  - 리그 다목적 NSGA-III가 최소 실행되는지

실행: 프로젝트 루트에서  python tools/smoke_workflow.py
"""
import sys
import time
from importlib.util import find_spec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")    # type: ignore[union-attr]
    except Exception:
        pass

from app.academy.academy import prepare_academy_split
from app.academy.study import cma_es, gp, nsga3, tpe
from app.academy.study.nsga3 import evaluate_balances
from app.backend.data_io.data import load_gyms
from app.backend.engine.battle import fight_dca
from app.backend.genes.signals import ALL_GENES
from app.backend.market.gym import all_gyms


SEED = 42
ACADEMY_N_TRAIN = 2
ACADEMY_N_VALIDATION = 2
SINGLE_OBJ_TRIALS = 2
NSGA3_TRIALS = 5
NSGA3_POPULATION = 5


def _balance_sum(bals: dict) -> int:
    return sum(b["strat"] for b in bals.values())


def _run_single_obj(label: str, engine, loaded_gyms: list, dca: dict):
    """단일목적 study 1개를 아주 짧게 실행한다."""
    t0 = time.perf_counter()
    study, gyms, dca_out = engine.run_study(
        trials=SINGLE_OBJ_TRIALS, seed=SEED, loaded_gyms=loaded_gyms, dca=dca)
    elapsed = time.perf_counter() - t0
    weights, bals, summary = engine.champion_balances(study, gyms, dca_out)
    ok = len(study.trials) == SINGLE_OBJ_TRIALS and summary["balance_sum"] > 0
    if not ok:
        raise RuntimeError(f"{label} 단일목적 smoke 실패")
    main = _format_main_weights(weights)
    print(f"  [PASS] {label:<14} trials {len(study.trials)} · "
          f"best {summary['balance_sum']/10000:7.1f}만 · {elapsed:4.1f}s · {main}")
    return study, elapsed, _balance_sum(bals)


def _run_optional_single_obj(label: str, engine, loaded_gyms: list, dca: dict,
                             required_modules: list[str]):
    """선택 sampler는 의존성 없으면 실패가 아니라 SKIP으로 기록한다."""
    missing = [m for m in required_modules if find_spec(m) is None]
    if missing:
        print(f"  [SKIP] {label:<14} missing: {', '.join(missing)}")
        return None, 0.0, 0
    return _run_single_obj(label, engine, loaded_gyms, dca)


def _format_main_weights(weights: list[float]) -> str:
    total = sum(weights) or 1.0
    pairs = [(g, w / total * 100) for g, w in zip(ALL_GENES, weights)]
    pairs = [(g, p) for g, p in pairs if p >= 10.0]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return " · ".join(f"{g} {p:.0f}%" for g, p in pairs) or "분산"


def _check_academy() -> float:
    """아카데미 합성장 → 단일목적 study → validation 재평가."""
    print("\n=== ① 아카데미 smoke ===")
    t0 = time.perf_counter()
    (train_gyms, train_dca), (val_gyms, val_dca) = prepare_academy_split(
        n_train=ACADEMY_N_TRAIN,
        n_validation=ACADEMY_N_VALIDATION,
        train_seed=SEED,
        validation_seed=SEED + 10_000,
    )
    if not all(g.prices.attrs.get("synthetic") for g in train_gyms + val_gyms):
        raise RuntimeError("아카데미 합성장 synthetic attrs 누락")
    if train_gyms[0].prices.equals(val_gyms[0].prices):
        raise RuntimeError("아카데미 train/validation split이 같은 세계")

    for name, engine in (("Academy TPE", tpe), ("Academy CMA-ES", cma_es),
                         ("Academy GP", gp)):
        if "CMA-ES" in name:
            study, _elapsed, _ = _run_optional_single_obj(
                name, engine, train_gyms, train_dca, ["cmaes"])
        elif "GP" in name:
            study, _elapsed, _ = _run_optional_single_obj(
                name, engine, train_gyms, train_dca, [])
        else:
            study, _elapsed, _ = _run_single_obj(name, engine, train_gyms, train_dca)
        if study is None:
            continue
        # 같은 best trial을 validation에 다시 평가할 수 있어야 한다.
        weights, _params = nsga3.decode_params(study.best_trial.params)
        val_bals = evaluate_balances(weights, {}, val_gyms, val_dca)
        if _balance_sum(val_bals) <= 0:
            raise RuntimeError(f"{name} validation 평가 실패")
        print(f"         validation 재평가 { _balance_sum(val_bals)/10000:7.1f}만")
    return time.perf_counter() - t0


def _check_real_gyms() -> float:
    """실데이터 6체육관 → 단일목적 study."""
    print("\n=== ② 체육관 smoke ===")
    t0 = time.perf_counter()
    gyms = load_gyms(all_gyms())
    dca = {lg.gym.name: fight_dca(lg) for lg in gyms}
    for name, engine in (("Gym TPE", tpe), ("Gym CMA-ES", cma_es), ("Gym GP", gp)):
        if "CMA-ES" in name:
            _run_optional_single_obj(name, engine, gyms, dca, ["cmaes"])
        elif "GP" in name:
            _run_optional_single_obj(name, engine, gyms, dca, [])
        else:
            _run_single_obj(name, engine, gyms, dca)
    return time.perf_counter() - t0


def _check_league() -> float:
    """리그 다목적 NSGA-III 최소 실행."""
    print("\n=== ③ 리그 smoke ===")
    t0 = time.perf_counter()
    study, gyms, dca, _hv, _mut = nsga3.run_study(
        NSGA3_TRIALS, seed=SEED, population_size=NSGA3_POPULATION)
    if len(study.trials) != NSGA3_TRIALS:
        raise RuntimeError("NSGA-III trial 수 불일치")
    summary = nsga3.summarize_front(study, loaded_gyms=gyms, dca=dca)
    elapsed = time.perf_counter() - t0
    print(f"  [PASS] NSGA-III      trials {len(study.trials)} · "
          f"front {summary['front_size']} · {elapsed:4.1f}s")
    return elapsed


def main() -> int:
    print("=== PocketQuant workflow smoke ===")
    print(f"seed {SEED} · single trials {SINGLE_OBJ_TRIALS} · "
          f"nsga3 trials {NSGA3_TRIALS}\n")
    rows = []
    try:
        rows.append(("아카데미", _check_academy()))
        rows.append(("체육관", _check_real_gyms()))
        rows.append(("리그", _check_league()))
    except Exception as exc:
        print(f"\n=== 판정: FAIL ===\n{exc}")
        return 1

    print("\n=== 결과 ===")
    for name, elapsed in rows:
        print(f"  {name:<8} PASS  {elapsed:5.1f}s")
    print(f"\n=== 판정: PASS · 총 {sum(e for _, e in rows):.1f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
