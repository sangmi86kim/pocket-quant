"""단일목적 1시드 — `search.tpe.run_study` 결과 콘솔 표시.

엔진(`app/academy/study/tpe.py`)이 탐색·1등 추출까지 다 한다 — 본 어댑터는
인쇄·비교만. 5시드 안정성 검증은 `single_obj_sweep.py`.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.academy.study import tpe
from app.academy.study.nsga3 import evaluate_balances
from app.backend.genes.signals import ALL_GENES

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")    # type: ignore[union-attr]
    except Exception:
        pass

SEED = 42
TRIALS = 2000


def _format_genes(weights: list[float]) -> str:
    total = sum(weights) or 1.0
    main = sorted(
        [(g, w / total * 100) for g, w in zip(ALL_GENES, weights) if w / total > 0.1],
        key=lambda x: x[1], reverse=True,
    )
    return " · ".join(f"{g} {p:.0f}%" for g, p in main) or "분산"


def main() -> None:
    print("=== 단일목적 TPE — 6체육관 100만원 시드, 잔고 합 max ===")
    print(f"시드 {SEED} · trials {TRIALS} · sampler TPE · 가중치 {len(ALL_GENES)}차원\n")

    t0 = time.perf_counter()

    def _progress(done: int, total: int, best_value: float) -> None:
        if done % 500 == 0:
            print(f"  [{done:>5}/{total}] 1등 {best_value/10000:6.1f}만")

    study, loaded_gyms, dca = tpe.run_study(
        trials=TRIALS, seed=SEED, on_progress=_progress,
    )
    elapsed = time.perf_counter() - t0

    weights, bals, summary = tpe.champion_balances(study, loaded_gyms, dca)

    # 기준점 — 현 챔피언 동일가중 VOL+REV_RSI+REV_BB
    champ_w = [1.0 if g in ("VOL", "REV_RSI", "REV_BB") else 0.0 for g in ALL_GENES]
    champ_bals = evaluate_balances(champ_w, {}, loaded_gyms, dca, seed_krw=tpe.SEED_KRW)
    champ_sum = sum(b["strat"] for b in champ_bals.values())
    dca_sum = sum(b["dca"] for b in champ_bals.values())

    print(f"\n=== 1등 — trial #{summary['trial']} ({elapsed:.1f}s) ===")
    diff = summary["balance_sum"] - champ_sum
    sign = "+" if diff >= 0 else ""
    print(f"  잔고 합: {summary['balance_sum']/10000:.1f}만 "
          f"(현 챔피언 {champ_sum/10000:.1f}만 대비 {sign}{diff/10000:.1f}만, "
          f"성실이 {dca_sum/10000:.1f}만)")
    print(f"  주력 : {_format_genes(weights)}")
    print(f"\n  체육관별 (전략 / 성실이):")
    for gym_name, b in bals.items():
        print(f"    {gym_name:<22} {b['strat']/10000:6.1f}만 / {b['dca']/10000:6.1f}만")


if __name__ == "__main__":
    main()
