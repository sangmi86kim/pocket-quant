"""챔피언로드 ② 배틀 프론티어 — top10 + 챔피언 + 기준선 4인방 풀라인업 (06-13).

같은 시드의 평행세계에 15명 다 입장:
  - 전천후 200세계 (블록 부트스트랩, 1999~2020.06 재배열)
  - 하락(bear) 100세계 (닷컴+리먼 블록만)
  - 회복(rebound) 100세계 (회복기 블록만)

각 arena에서:
  - 후보별 평균 종료 잔고 (100만원 시드 × 2년 평가)
  - 세계별 1등 카운트 (같은 시드라 동일 세계 비교)
  - 어플삭제맨이 bear에서 무너지는 자리 확인

산출: tools/outputs/battle_frontier_lineup.md (+ .json)
실행: python tools/battle_frontier_lineup.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

from app.backend.engine.battle import (_score_position, fight_dca, fight_savings,
                                         terminal_balance)
from app.backend.genes.signals import ALL_GENES, combine_positions, positions_with_params
from app.backend.market.data import LoadedGym, get_prices
from tests.battle_frontier import (DATA_START, DATA_END, N_WORLDS_ALL,
                                     N_WORLDS_REGIME, REGIME_SPANS, SEED,
                                     make_world)

SEED_KRW = 1_000_000
TOP10_JSON = _ROOT / "tools" / "outputs" / "top10_champions.json"
OUT_MD = _ROOT / "tools" / "outputs" / "battle_frontier_lineup.md"


def eval_weights(weights: list[float], world: LoadedGym) -> int:
    pos = combine_positions(positions_with_params(world.prices), weights)
    return terminal_balance(_score_position(pos, world), SEED_KRW)


def eval_buy_hold(world: LoadedGym) -> int:
    pos = pd.Series(1.0, index=world.prices.index)
    return terminal_balance(_score_position(pos, world), SEED_KRW)


def eval_piggy(world: LoadedGym) -> int:
    return SEED_KRW


def eval_savings(world: LoadedGym) -> int:
    return terminal_balance(fight_savings(world), SEED_KRW)


def eval_dca(world: LoadedGym) -> int:
    return terminal_balance(fight_dca(world), SEED_KRW)


def main() -> None:
    t0 = time.time()
    prices = get_prices("QQQ", DATA_START, DATA_END)
    full_returns = prices.pct_change().dropna()
    regime_returns = {
        name: pd.concat([full_returns.loc[s:e] for s, e in spans])
        for name, spans in REGIME_SPANS.items()
    }

    # 명단
    top10 = json.loads(TOP10_JSON.read_text(encoding="utf-8"))
    champion_w = [1.0 if g in ("VOL", "REV_RSI", "REV_BB") else 0.0 for g in ALL_GENES]
    weight_candidates = [("현챔피언", champion_w)]
    for i, t in enumerate(top10, 1):
        w = [t["params"][f"w_{g}"] for g in ALL_GENES]
        weight_candidates.append((f"TOP{i:02d}", w))
    baselines = [("어플삭제맨", eval_buy_hold),
                 ("저축왕", eval_savings),
                 ("성실이", eval_dca),
                 ("돼지저금통", eval_piggy)]
    all_names = [n for n, _ in weight_candidates] + [n for n, _ in baselines]

    # arena 정의
    arenas = [("전천후", None, N_WORLDS_ALL),
              ("bear", regime_returns["bear"], N_WORLDS_REGIME),
              ("rebound", regime_returns["rebound"], N_WORLDS_REGIME)]

    # arena별 {name: [세계별 잔고]}
    arena_results: dict[str, dict[str, list[int]]] = {}
    for arena, pool, n_worlds in arenas:
        print(f"\n=== {arena} arena ({n_worlds}세계) 평가 중 ===")
        bals: dict[str, list[int]] = {n: [] for n in all_names}
        rng = np.random.default_rng(SEED)
        for i in range(n_worlds):
            world = make_world(full_returns, rng, pool)
            for name, w in weight_candidates:
                bals[name].append(eval_weights(w, world))
            for name, fn in baselines:
                bals[name].append(fn(world))
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{n_worlds}")
        arena_results[arena] = bals

    # ── 집계 ──
    print(f"\n전체 평가 완료 — {time.time() - t0:.0f}초")

    md = ["# 챔피언로드 ② 배틀 프론티어 — 풀라인업 + 기준선 4인방 (06-13)", ""]
    md.append("- 시드 100만원 × 평가 2년 (504거래일) — arena별 평행세계")
    md.append(f"- 시드 {SEED}, 블록 21일 부트스트랩")
    md.append("")

    md.append("## 평균 종료 잔고 (arena × 후보, 단위 만원)")
    md.append("")
    head = "| 후보 | " + " | ".join(f"{a} ({n})" for a, _, n in arenas) + " |"
    md.append(head)
    md.append("|---|" + "---:|" * len(arenas))
    for name in all_names:
        cells = []
        for arena, _, _ in arenas:
            mean = int(np.mean(arena_results[arena][name]))
            cells.append(f"{mean // 10000:,}")
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    md.append("")

    md.append("## 세계 1등 카운트 (arena별)")
    md.append("")
    md.append("| 후보 | " + " | ".join(f"{a}" for a, _, _ in arenas) + " | 총합 |")
    md.append("|---|" + "---:|" * (len(arenas) + 1))
    wins_total = {n: 0 for n in all_names}
    wins_by_arena: dict[str, dict[str, int]] = {}
    for arena, _, n_worlds in arenas:
        wins = {n: 0 for n in all_names}
        for i in range(n_worlds):
            winner = max(all_names, key=lambda n: arena_results[arena][n][i])
            wins[winner] += 1
            wins_total[winner] += 1
        wins_by_arena[arena] = wins
    for name in all_names:
        cells = []
        for arena, _, _ in arenas:
            cells.append(f"{wins_by_arena[arena][name]}")
        md.append(f"| {name} | " + " | ".join(cells)
                  + f" | **{wins_total[name]}** |")
    md.append("")

    # 하위 5%·중앙값 분포 (위기 견디는지)
    md.append("## 분포 — bear arena 하위 5% / 중앙값 (단위 만원)")
    md.append("")
    md.append("| 후보 | 하위 5% | 중앙값 | 평균 | 손실 비율 |")
    md.append("|---|---:|---:|---:|---:|")
    for name in all_names:
        b = arena_results["bear"][name]
        p5 = int(np.percentile(b, 5))
        med = int(np.median(b))
        mean = int(np.mean(b))
        lose_ratio = sum(1 for v in b if v < SEED_KRW) / len(b)
        md.append(f"| {name} | {p5 // 10000:,} | {med // 10000:,} | "
                  f"{mean // 10000:,} | {lose_ratio:.0%} |")
    md.append("")
    md.append("> 손실 비율 = 100만원보다 잔고 적은 세계 비율. 위기에 깨지는 빈도.")
    md.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    OUT_JSON = OUT_MD.with_suffix(".json")
    OUT_JSON.write_text(json.dumps(
        {"arenas": {a: {n: arena_results[a][n] for n in all_names}
                     for a, _, _ in arenas},
         "wins_by_arena": wins_by_arena, "wins_total": wins_total},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {OUT_MD} (+ .json)")
    print("\n총 1등:")
    for n, c in sorted(wins_total.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {n}: {c}회")


if __name__ == "__main__":
    main()
