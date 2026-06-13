"""GP 단독 saturation 탐색 — `search.gp.run_study` + plateau 얼리스토핑.

`compare_gp.py`의 3-way 중 GP만 떼어낸 단독 진입점. scipy/torch가 빠지면 GP 한
부분이 죽는데 3-way 어댑터는 거기서 통째 crash하므로, GP만 다시 돌릴 때 쓴다.
TPE/CMA-ES 결과는 `compare_gp.py`/ `single_obj_compare.py`에서 이미 굳혀졌고,
이 어댑터는 GP saturation 단독 보고용.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.academy.study import gp
from app.academy.study.nsga3 import evaluate_balances
from app.backend.genes.signals import ALL_GENES
from app.league.single_obj_compare_gp import PlateauStopCallback, PATIENCE, MIN_DELTA_PCT

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")    # type: ignore[union-attr]
    except Exception:
        pass

SEED = 42
CAP = 1500       # GP는 trial당 O(n³) 비용 — 별도 cap


def _format_genes(weights: list[float]) -> str:
    total = sum(weights) or 1.0
    main = sorted(
        [(g, w / total * 100) for g, w in zip(ALL_GENES, weights) if w / total > 0.1],
        key=lambda x: x[1], reverse=True,
    )
    return " · ".join(f"{g} {p:.0f}%" for g, p in main) or "분산"


def main() -> None:
    print("=== GP 단독 saturation — 13마리 풀, 잔고 합 max ===")
    print(f"시드 {SEED} · cap {CAP} · patience {PATIENCE} · "
          f"min_delta {MIN_DELTA_PCT*100:.3f}%\n")

    loaded_gyms, dca = gp.prepare_data()
    stop_cb = PlateauStopCallback(PATIENCE, MIN_DELTA_PCT)

    def progress(done: int, total: int, best_value: float) -> None:
        # GP는 후반부로 갈수록 trial당 비용이 커져 진행 출력 잦게.
        if done % 100 == 0:
            print(f"    [{done:>4}/{total}] best {best_value/10000:6.1f}만 · "
                  f"plateau {done - stop_cb.last_improve_trial}/{PATIENCE}")

    t0 = time.perf_counter()
    study, _, _ = gp.run_study(
        trials=CAP, seed=SEED, loaded_gyms=loaded_gyms, dca=dca,
        on_progress=progress, extra_callbacks=[stop_cb],
    )
    weights, bals, summary = gp.champion_balances(study, loaded_gyms, dca)
    elapsed = time.perf_counter() - t0
    n = len(study.trials)
    stopped = (f"plateau stop @ {stop_cb.stopped_at}" if stop_cb.stopped_at
               else f"cap hit @ {n}")

    # 기준점
    champ_w = [1.0 if g in ("VOL", "REV_RSI", "REV_BB") else 0.0 for g in ALL_GENES]
    champ_bals = evaluate_balances(champ_w, {}, loaded_gyms, dca, seed_krw=gp.SEED_KRW)
    champ_sum = sum(b["strat"] for b in champ_bals.values())
    dca_sum = sum(b["dca"] for b in champ_bals.values())
    diff = summary["balance_sum"] - champ_sum
    sign = "+" if diff >= 0 else ""

    print(f"\n=== 1등 — trial #{summary['trial']} ({elapsed:.1f}s) ===")
    print(f"  잔고 합 {summary['balance_sum']/10000:.1f}만 · 현 챔피언 "
          f"{champ_sum/10000:.1f}만 대비 {sign}{diff/10000:.1f}만 · "
          f"성실이 {dca_sum/10000:.1f}만")
    print(f"  trials {n} · saturate @ {stop_cb.saturate_trial} · {stopped}")
    print(f"  주력 : {_format_genes(weights)}")
    print(f"\n  체육관별 (전략 / 성실이):")
    for gym_name, b in bals.items():
        print(f"    {gym_name:<22} {b['strat']/10000:6.1f}만 / {b['dca']/10000:6.1f}만")


if __name__ == "__main__":
    main()
