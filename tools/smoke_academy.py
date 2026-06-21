# ruff: noqa: E402
"""smoke_academy.py — 학교(academy) 워크플로우 스모크.

"애들 키우고 졸업시키기"가 끊김 없이 도는지만 본다(연결 확인용, 성능 아님).

순서:
  ① 합성장 학습: train/validation split → TPE/CMA-ES/GP/NSGA-III 짧은 trial
  ② 졸업시험   : 방금 키운 졸업생을 실QQQ 6체육관에 응시 → 졸업 성적표 조립
                 (graduate.build_payload, 진단 전용 · median 잣대)

목적:
  - 합성장 attrs가 실제 study 엔진까지 들어가는지 / train≠validation 인지
  - NSGA payload 조립(selected·select_score)이 안 깨졌는지
  - 졸업 러너 조립(디코드 → 6체육관 채점 → median)이 현재 14신호로 도는지

실행: 프로젝트 루트에서  python tools/smoke_academy.py
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

from app.academy.curriculum import prepare_academy_split
from app.academy.exam import all_gyms, graduate
from app.academy.exam.grade import evaluate_balances
from app.academy.training import study
from app.academy.training.candidate import decode_params
from app.academy.training.multi_objective import nsga3
from app.academy.training.single_objective import cma_es, gp, tpe
from app.pocket.battle import fight_dca
from app.pocket.signals import SIGNAL_NAMES
from app.world.data_loader import load_gyms

SEED = 42
ACADEMY_N_TRAIN = 2
ACADEMY_N_VALIDATION = 2
SINGLE_OBJ_TRIALS = 2
NSGA3_TRIALS = 5
NSGA3_POPULATION = 5
TOPK_SMOKE = 3        # 반별로 졸업생 몇 명만 졸업시험에 태운다(스모크라 소수)


def _balance_sum(bals: dict) -> int:
    return sum(b["strat"] for b in bals.values())


def _format_main_weights(weights: list[float]) -> str:
    total = sum(weights) or 1.0
    pairs = [(g, w / total * 100) for g, w in zip(SIGNAL_NAMES, weights)]
    pairs = [(g, p) for g, p in pairs if p >= 10.0]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return " · ".join(f"{g} {p:.0f}%" for g, p in pairs) or "분산"


def _run_single_obj(label: str, engine, loaded_gyms: list, dca: dict):
    t0 = time.perf_counter()
    study_obj, gyms, dca_out = engine.run_study(
        trials=SINGLE_OBJ_TRIALS, seed=SEED, loaded_gyms=loaded_gyms, dca=dca)
    elapsed = time.perf_counter() - t0
    weights, _bals, summary = engine.champion_balances(study_obj, gyms, dca_out)
    ok = len(study_obj.trials) == SINGLE_OBJ_TRIALS and summary["balance_median"] > 0
    if not ok:
        raise RuntimeError(f"{label} 단일목적 smoke 실패")
    print(f"  [PASS] {label:<14} trials {len(study_obj.trials)} · "
          f"best {summary['balance_median']/10000:7.1f}만 · {elapsed:4.1f}s · "
          f"{_format_main_weights(weights)}")
    return study_obj


def _run_optional_single_obj(label, engine, loaded_gyms, dca, required_modules):
    missing = [m for m in required_modules if find_spec(m) is None]
    if missing:
        print(f"  [SKIP] {label:<14} missing: {', '.join(missing)}")
        return None
    return _run_single_obj(label, engine, loaded_gyms, dca)


def _topk_items(study_obj) -> list[dict]:
    """완료 trial을 점수순 상위 몇 명 → top30 dict의 topk 형식(trial·params)."""
    done = [t for t in study_obj.trials if t.value is not None]
    done.sort(key=lambda t: t.value, reverse=True)
    return [{"trial": t.number, "params": dict(t.params)} for t in done[:TOPK_SMOKE]]


def _check_training():
    """① 합성장 학습 — 4반 짧은 trial. 졸업시험에 넘길 반별 졸업생을 모은다."""
    print("\n=== ① 합성장 학습 smoke ===")
    t0 = time.perf_counter()
    (train_gyms, train_dca), (val_gyms, val_dca) = prepare_academy_split(
        n_train=ACADEMY_N_TRAIN, n_validation=ACADEMY_N_VALIDATION,
        train_seed=SEED, validation_seed=SEED + 10_000)
    if not all(g.prices.attrs.get("synthetic") for g in train_gyms + val_gyms):
        raise RuntimeError("합성장 synthetic attrs 누락")
    if train_gyms[0].prices.equals(val_gyms[0].prices):
        raise RuntimeError("train/validation split이 같은 세계")

    classrooms = []
    for name, engine, req in (("TPE", tpe, None), ("CMA-ES", cma_es, ["cmaes"]),
                              ("GP", gp, [])):
        study_obj = (_run_single_obj(name, engine, train_gyms, train_dca)
                     if req is None else
                     _run_optional_single_obj(name, engine, train_gyms, train_dca, req))
        if study_obj is None:
            continue
        weights, _params = decode_params(study_obj.best_trial.params)
        val_bals = evaluate_balances(weights, {}, val_gyms, val_dca)
        if _balance_sum(val_bals) <= 0:
            raise RuntimeError(f"{name} validation 평가 실패")
        print(f"         validation 재평가 {_balance_sum(val_bals)/10000:7.1f}만")
        classrooms.append({"name": name, "topk": _topk_items(study_obj)})

    # NSGA-III — 다목적 반. payload 조립(selected·select_score)도 여기서 점검.
    nsga_study, *_ = nsga3.run_study(NSGA3_TRIALS, seed=SEED,
                                     population_size=NSGA3_POPULATION)
    summary = nsga3.summarize_front(nsga_study)
    items = study.nsga_items(summary)
    if not items or any("select_score" not in it for it in items):
        raise RuntimeError("NSGA payload items/select_score 누락 — payload 저장 깨짐")
    print(f"  [PASS] NSGA-III       trials {len(nsga_study.trials)} · "
          f"front {summary['front_size']} · selected {len(items)}")
    classrooms.append({"name": "NSGA-III",
                       "topk": [{"trial": it["trial"], "params": it["params"]}
                                for it in items[:TOPK_SMOKE]]})

    top30 = {"stamp": "smoke", "classrooms": classrooms}
    return time.perf_counter() - t0, top30


def _check_graduation(top30: dict):
    """② 졸업시험 — 방금 키운 졸업생을 실QQQ 6체육관에 응시(진단 전용)."""
    print("\n=== ② 졸업시험 smoke (실QQQ 6체육관) ===")
    t0 = time.perf_counter()
    gyms = load_gyms(all_gyms())
    dca = {lg.gym.name: fight_dca(lg) for lg in gyms}
    payload = graduate.build_payload(top30, gyms, dca)
    if payload["dca_score"] <= 0:
        raise RuntimeError("성실이(DCA) 졸업 기준선 0")
    for c in payload["classrooms"]:
        if not c["members"]:
            raise RuntimeError(f"{c['group']} 졸업생 없음")
        if any(m["score"] <= 0 for m in c["members"]):
            raise RuntimeError(f"{c['group']} 졸업 점수 0 — 디코드/채점 깨짐")
        med = sorted(m["score"] for m in c["members"])[len(c["members"]) // 2]
        print(f"  [PASS] {c['group']:<14} 졸업생 {len(c['members'])} · "
              f"6체육관 median {med/10000:7.1f}만")
    print(f"  성실이(DCA) 기준선 {payload['dca_score']/10000:7.1f}만")
    return time.perf_counter() - t0


def main() -> int:
    print("=== PocketQuant 학교(academy) smoke ===")
    print(f"seed {SEED} · single trials {SINGLE_OBJ_TRIALS} · "
          f"nsga3 trials {NSGA3_TRIALS}\n")
    rows = []
    try:
        train_elapsed, top30 = _check_training()
        rows.append(("합성장 학습", train_elapsed))
        rows.append(("졸업시험", _check_graduation(top30)))
    except Exception as exc:
        print(f"\n=== 판정: FAIL ===\n{exc}")
        return 1

    print("\n=== 결과 ===")
    for name, elapsed in rows:
        print(f"  {name:<10} PASS  {elapsed:5.1f}s")
    print(f"\n=== 판정: PASS · 총 {sum(e for _, e in rows):.1f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
