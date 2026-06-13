"""v1.x 챔피언로드 라인업 — 4 sampler × 5시드 → 챔피언로드 관문 ① OOS 출전.

[라인업 구성]
  단일목적 sampler 3종 (TPE/CMA-ES/GP) × 5시드 = 15명 (시드당 1등 1명씩)
  NSGA-III × 5시드 = 떼거지(시드별 front 통과 + 스페셜리스트 5명) — ~100~250명
  기준: 현 챔피언 (VOL+REV_RSI+REV_BB 동일가중)

[중요]
  단일목적은 plateau 얼리스토핑(saturation까지). NSGA-III는 HV-MA 정체 self-stop.
  prepare_data 1번 — 모든 sampler 공유 (yfinance/fight_dca 중복 차단).
  결과는 victory_road.run_gate1(graduates) 그대로 호출 — OOS 11년 성실이전.

  ⚠️ 시간 약 60~90분 백그라운드. 시드별 1등 잔고는 즉시 출력 → 끊겨도 진척 보임.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")    # type: ignore[union-attr]
    except Exception:
        pass

from app.academy.study import tpe, cma_es, gp, nsga3
from app.backend.genes.signals import ALL_GENES
from app.league.single_obj_compare_gp import PlateauStopCallback, PATIENCE, MIN_DELTA_PCT
from app.league.victory_road import run_gate1

SEEDS = [42, 7, 11, 19, 23]
CAP_FAST = 5000      # TPE/CMA-ES
CAP_GP = 1500        # GP (trial당 O(n³) — 별도 cap)
NSGA3_TRIALS = 1200  # 시드당 (사용자 본업 친화 영역 + HV early-stop으로 정체 시 멈춤)
NSGA3_EARLY_STOP_WINDOW = 5
NSGA3_POP = 50
NSGA3_TOLERANCE = 0.10
NSGA3_STORAGE_DB = _ROOT / "optuna_pocketquant_v1x_lineup.db"


def _genes_str(weights: list[float], thr: float = 0.10) -> str:
    total = sum(weights) or 1.0
    main = sorted(
        [(g, w / total) for g, w in zip(ALL_GENES, weights) if w / total > thr],
        key=lambda x: x[1], reverse=True,
    )
    return " · ".join(f"{g} {p*100:.0f}%" for g, p in main) or "분산"


def sweep_single_obj(engine, name: str, cap: int,
                     loaded_gyms, dca) -> list[dict]:
    """단일목적 sampler 5시드 saturation sweep → graduates 5명."""
    print(f"\n┏━━ {name} × {len(SEEDS)}시드 saturation (cap {cap}) ━━━━━━━━━━━━━━━━")
    out = []
    for seed in SEEDS:
        stop_cb = PlateauStopCallback(PATIENCE, MIN_DELTA_PCT)
        t0 = time.perf_counter()
        study, _, _ = engine.run_study(
            trials=cap, seed=seed, loaded_gyms=loaded_gyms, dca=dca,
            extra_callbacks=[stop_cb],
        )
        weights, _, summary = engine.champion_balances(study, loaded_gyms, dca)
        elapsed = time.perf_counter() - t0
        n = len(study.trials)
        sat = stop_cb.saturate_trial
        print(f"  s{seed:>2} #{summary['trial']:<5} 잔고 합 "
              f"{summary['balance_sum']/10000:6.1f}만 (trials {n}, sat@{sat}, "
              f"{elapsed:.1f}s)  {_genes_str(weights)}")
        out.append({
            "name": f"{name}-s{seed}",
            "label": "단일목적",
            "weights": weights,
            "params": {},
            "mean5": None,
            "specialist": False,
        })
    return out


def sweep_nsga3(loaded_gyms, dca) -> list[dict]:
    """NSGA-III × 5시드 → 시드별 (front 통과 + 스페셜리스트 5명) 떼거지 입장."""
    print(f"\n┏━━ NSGA-III × {len(SEEDS)}시드 (trials {NSGA3_TRIALS}, "
          f"HV-MA{NSGA3_EARLY_STOP_WINDOW} early-stop) ━━")
    storage = f"sqlite:///{NSGA3_STORAGE_DB.as_posix()}"
    # 깨끗한 db로 시작 — 이전 시즌 study 충돌 방지
    if NSGA3_STORAGE_DB.exists():
        NSGA3_STORAGE_DB.unlink()
    out = []
    for seed in SEEDS:
        study_name = f"v1x_lineup_s{seed}"
        t0 = time.perf_counter()
        # nsga3.run_study는 자기 loaded_gyms를 새로 만든다 — 캐시 hit이라 빠름
        study, lg, _dca, hv_cb, _ = nsga3.run_study(
            n_trials=NSGA3_TRIALS, seed=seed,
            storage=storage, study_name=study_name,
            tune_params=False, population_size=NSGA3_POP,
            early_stop_window=NSGA3_EARLY_STOP_WINDOW,
        )
        summary = nsga3.summarize_front(study, tolerance=NSGA3_TOLERANCE)
        label_of = {row["number"]: lbl for lbl, row in summary["labels"].items()}
        elapsed = time.perf_counter() - t0
        hv_note = f"HV {len(hv_cb.hv)}gen · stopped {getattr(hv_cb, 'stopped', False)}" \
                  if hv_cb else "no HV"
        print(f"  s{seed:>2} front {summary['front_size']}개 · 필터 통과 "
              f"{len(summary['passed'])}명 · {hv_note} ({elapsed:.1f}s)")

        # 올라운더 (필터 통과)
        for r in summary["passed"]:
            w, sig = nsga3.decode_params(r["params"])
            label = label_of.get(r["number"], "통과")
            out.append({
                "name": f"NSGA3-s{seed}-#{r['number']}",
                "label": f"{label}",
                "weights": w, "params": sig,
                "mean5": r["mean5"], "specialist": False,
            })

        # 스페셜리스트 (목적별 1위 5명, 필터 무시 — 이미 있으면 생략)
        seen = {f"NSGA3-s{seed}-#{r['number']}" for r in summary["passed"]}
        front = [{"number": t.number, "values": list(t.values),
                  "params": dict(t.params)} for t in study.best_trials]
        for i in range(5):
            spec = max(front, key=lambda r: r["values"][i])
            name = f"NSGA3-s{seed}-#{spec['number']}"
            if name in seen:
                continue
            seen.add(name)
            w, sig = nsga3.decode_params(spec["params"])
            out.append({
                "name": name,
                "label": f"★{nsga3.OBJECTIVE_NAMES[i]}",
                "weights": w, "params": sig,
                "mean5": sum(spec["values"][:5]) / 5, "specialist": True,
            })
        print(f"        → 입장 {sum(1 for g in out if g['name'].startswith(f'NSGA3-s{seed}'))}명 "
              f"(올라운더 {len(summary['passed'])} + 스페셜리스트 누적)")
    return out


def build_lineup(loaded_gyms, dca) -> list[dict]:
    """4 sampler × 5시드 라인업 — 기준 + 단일목적 15 + NSGA-III 떼거지."""
    graduates: list[dict] = []

    # 기준 — 현 챔피언 (동일가중 VOL+REV_RSI+REV_BB)
    graduates.append({
        "name": "현챔피언(동일가중)", "label": "기준",
        "weights": [1.0 if g in ("VOL", "REV_RSI", "REV_BB") else 0.0 for g in ALL_GENES],
        "params": {}, "mean5": None, "specialist": False,
    })

    # 단일목적 3종
    graduates += sweep_single_obj(tpe,    "TPE",    CAP_FAST, loaded_gyms, dca)
    graduates += sweep_single_obj(cma_es, "CMA-ES", CAP_FAST, loaded_gyms, dca)
    graduates += sweep_single_obj(gp,     "GP",     CAP_GP,   loaded_gyms, dca)

    # 다목적 — 떼거지
    graduates += sweep_nsga3(loaded_gyms, dca)
    return graduates


def main() -> None:
    print("=== v1.x 챔피언로드 라인업 — 4 sampler × 5시드 → 관문 ①·② ===")
    print(f"시드 {SEEDS} · 단일목적 saturation · NSGA-III HV-MA early-stop")
    print(f"NSGA-III storage(임시): {NSGA3_STORAGE_DB.relative_to(_ROOT)}\n")

    t0 = time.perf_counter()
    loaded_gyms, dca = tpe.prepare_data()

    graduates = build_lineup(loaded_gyms, dca)
    elapsed = time.perf_counter() - t0
    print(f"\n=== 라인업 완성: 총 {len(graduates)}명 ({elapsed/60:.1f}분 소요) ===\n")

    # ── 챔피언로드 관문 ① 출전 ────────────────────────
    print("\n" + "=" * 70)
    print("관문 ① OOS 11년 시험 — victory_road.run_gate1")
    print("=" * 70 + "\n")
    survived = run_gate1(graduates=graduates)
    print(f"\n=== 관문 ① 결과: 도전권 {'있음' if survived else '없음'} ===")

    # ── 챔피언로드 관문 ② 출전 (배틀 프론티어) ───────
    print("\n" + "=" * 70)
    print("관문 ② 배틀 프론티어 — battle_frontier_v1x.main(graduates)")
    print("=" * 70 + "\n")
    from app.league.battle_frontier_v1x import main as run_gate2
    run_gate2(graduates)


if __name__ == "__main__":
    main()
