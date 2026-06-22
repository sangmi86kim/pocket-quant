# ruff: noqa: E402
"""smoke_academy.py — 학교(academy) 워크플로우 스모크.

"애들 키우고 졸업시키기"가 끊김 없이 도는지만 본다(연결 확인용, 성능 아님).

순서:
  ① 합성장 학습: RS 1차 → 약점 진단 → 70/30 보충 2차
  ② 졸업시험   : 방금 키운 졸업생(1차/보충)을 실QQQ 6체육관에 응시 → 성적표 조립
                 (graduate.build_payload, 진단 전용 · median 잣대)

목적:
  - 합성장 attrs가 실제 study 엔진까지 들어가는지
  - NSGA payload 조립(selected·select_score)이 안 깨졌는지
  - 졸업 러너 조립(디코드 → 6체육관 채점 → median)이 현재 14신호로 도는지

실행: 프로젝트 루트에서  python tools/smoke_academy.py
"""
import sys
import time
from importlib.util import find_spec
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")    # type: ignore[union-attr]
    except Exception:
        pass

from app.academy.exam import all_gyms, graduate
from app.academy.training import study
from app.academy.training.single_objective import cma_es, gp
from app.pocket.battle import fight_dca
from app.pocket.signals import SIGNAL_NAMES
from app.world.data_loader import load_gyms

SEED = 42
ACADEMY_N_TRAIN = 3
SINGLE_OBJ_TRIALS = 2
NSGA3_TRIALS = 5
TOPK_SMOKE = 3        # 반별로 졸업생 몇 명만 졸업시험에 태운다(스모크라 소수)


def _format_main_weights(weights: list[float]) -> str:
    total = sum(weights) or 1.0
    pairs = [(g, w / total * 100) for g, w in zip(SIGNAL_NAMES, weights)]
    pairs = [(g, p) for g, p in pairs if p >= 10.0]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return " · ".join(f"{g} {p:.0f}%" for g, p in pairs) or "분산"


def _trim_classroom(classroom: dict) -> dict:
    """졸업시험 스모크용으로 topk 수만 줄인다."""
    out = dict(classroom)
    out["topk"] = classroom["topk"][:TOPK_SMOKE]
    if "phase1" in classroom:
        out["phase1"] = dict(classroom["phase1"])
        out["phase1"]["topk"] = classroom["phase1"]["topk"][:TOPK_SMOKE]
    return out


def _run_single_obj(label: str, engine, loaded_gyms: list, dca: dict,
                    diagnostic: dict):
    t0 = time.perf_counter()
    result = study.run_single_classroom_2phase(
        label, engine, loaded_gyms, dca, diagnostic, trials=SINGLE_OBJ_TRIALS)
    elapsed = time.perf_counter() - t0
    if not result["topk"] or not result["phase1"]["topk"]:
        raise RuntimeError(f"{label} 단일목적 smoke 실패")
    weights, _params = decode_first(result["topk"])
    print(f"  [PASS] {label:<14} trials {result['trials']} · "
          f"weak {result['weak_regime']} · {elapsed:4.1f}s · "
          f"{_format_main_weights(weights)}")
    return _trim_classroom(result)


def decode_first(topk: list[dict]):
    from app.academy.training.candidate import decode_params
    return decode_params(topk[0]["params"])


def _run_optional_single_obj(label, engine, loaded_gyms, dca, diagnostic,
                             required_modules):
    missing = [m for m in required_modules if find_spec(m) is None]
    if missing:
        print(f"  [SKIP] {label:<14} missing: {', '.join(missing)}")
        return None
    return _run_single_obj(label, engine, loaded_gyms, dca, diagnostic)


def _check_training():
    """① 합성장 학습 — 3반 짧은 trial. 졸업시험에 넘길 반별 졸업생을 모은다."""
    print("\n=== ① 합성장 학습 smoke ===")
    t0 = time.perf_counter()
    train_gyms, train_dca = study.prepare_school_data(n_gyms=ACADEMY_N_TRAIN, seed=SEED)
    diagnostic = study.remedial.make_diagnostic_gyms(seed=SEED + 10_000, n_per_regime=1)
    diag_gyms = [lg for gyms, _dca in diagnostic.values() for lg in gyms]
    if not all(g.prices.attrs.get("synthetic") for g in train_gyms + diag_gyms):
        raise RuntimeError("합성장 synthetic attrs 누락")

    classrooms = []
    for name, engine, req in (("CMA-ES", cma_es, ["cmaes"]), ("GP", gp, [])):
        if name == "GP":
            result = study.run_gp_seedleague_2phase(
                train_gyms, train_dca, diagnostic,
                trials=SINGLE_OBJ_TRIALS, n_seeds=2)
            result = _trim_classroom(result)
            print(f"  [PASS] {'GP':<14} trials {result['trials']} · "
                  f"weak {result['weak_regime']} · selected {len(result['topk'])}")
        else:
            result = (_run_single_obj(name, engine, train_gyms, train_dca, diagnostic)
                      if req is None else
                      _run_optional_single_obj(name, engine, train_gyms, train_dca,
                                               diagnostic, req))
        if result is None:
            continue
        classrooms.append(result)

    nsga_result = study.run_nsga_classroom_2phase(
        f"smoke_{int(time.time() * 1000)}", train_gyms, train_dca, diagnostic,
        academy_seed=SEED, trials=NSGA3_TRIALS, use_storage=False)
    if not nsga_result["topk"] or any("select_score" not in it
                                      for it in nsga_result["topk"]):
        raise RuntimeError("NSGA payload topk/select_score 누락 — payload 저장 깨짐")
    print(f"  [PASS] NSGA-III       trials {nsga_result['trials']} · "
          f"weak {nsga_result['weak_regime']} · selected {len(nsga_result['topk'])}")
    classrooms.append(_trim_classroom(nsga_result))

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
        med = median(m["score"] for m in c["members"])
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
