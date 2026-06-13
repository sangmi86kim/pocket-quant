"""챔피언로드 ② 배틀 프론티어 — v1.x 라인업 본판정.

`champion_road_lineup_v1x.build_lineup()`이 만든 41명 + 기준선 4인방을 같은 시드의
평행세계 3 arena(전천후/bear/rebound)에 입장. 사용자 메모리 핵심: "스페셜리스트는
전문 시험장에서 판정" — bear arena가 ★bear 본판정, rebound arena가 ★rebound 본판정.

산출:
  reports/battle_frontier_v1x.md — 평균 잔고 표, 1등 카운트, bear 분포
  (json 저장 안 함 — 좋은 결과만 따로 저장은 Regime Scanner 붙일 때, 사용자 안)

기존 `battle_frontier_lineup.py`(v1 TOP10 전용)와 같은 arena/world 상수를 재사용.
단독 실행은 안 한다 — `champion_road_lineup_v1x.py` 끝에서 `main(lineup)` 호출.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8")    # type: ignore[union-attr]
    except Exception:
        pass

import numpy as np
import pandas as pd

from app.backend.engine.battle import (_score_position, fight_dca, fight_savings,
                                         terminal_balance)
from app.backend.genes.signals import ALL_GENES, combine_positions, positions_with_params
from app.backend.data_io.data import LoadedGym, get_prices
from app.league.battle_frontier import (DATA_START, DATA_END, N_WORLDS_ALL,
                                          N_WORLDS_REGIME, REGIME_SPANS, SEED,
                                          make_world)

SEED_KRW = 1_000_000
OUT_MD = _ROOT / "reports" / "battle_frontier_v1x.md"


def eval_weights(weights: list[float], params: dict, world: LoadedGym) -> int:
    pos = combine_positions(positions_with_params(world.prices, params or None), weights)
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


def main(lineup: list[dict]) -> None:
    """lineup = champion_road_lineup_v1x.build_lineup() 결과 (graduates dict 리스트)."""
    t0 = time.time()
    print(f"=== v1.x 라인업 본판정 — {len(lineup)}명 + 기준선 4인방 ===")

    prices = get_prices("QQQ", DATA_START, DATA_END)
    full_returns = prices.pct_change().dropna()
    regime_returns = {
        name: pd.concat([full_returns.loc[s:e] for s, e in spans])
        for name, spans in REGIME_SPANS.items()
    }

    # 라인업 후보 (이름, weights, params) — 단일목적은 params={}
    weight_candidates = [(g["name"], g["weights"], g.get("params") or {}, g.get("label", ""),
                          g.get("specialist", False)) for g in lineup]
    baselines = [("어플삭제맨", eval_buy_hold),
                 ("저축왕", eval_savings),
                 ("성실이", eval_dca),
                 ("돼지저금통", eval_piggy)]
    all_names = [n for n, _, _, _, _ in weight_candidates] + [n for n, _ in baselines]

    # arena 정의
    arenas = [("전천후", None, N_WORLDS_ALL),
              ("bear", regime_returns["bear"], N_WORLDS_REGIME),
              ("rebound", regime_returns["rebound"], N_WORLDS_REGIME)]

    arena_results: dict[str, dict[str, list[int]]] = {}
    for arena, pool, n_worlds in arenas:
        print(f"\n=== {arena} arena ({n_worlds}세계) 평가 중 ===")
        bals: dict[str, list[int]] = {n: [] for n in all_names}
        rng = np.random.default_rng(SEED)
        for i in range(n_worlds):
            world = make_world(full_returns, rng, pool)
            for name, w, params, _, _ in weight_candidates:
                bals[name].append(eval_weights(w, params, world))
            for name, fn in baselines:
                bals[name].append(fn(world))
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{n_worlds}")
        arena_results[arena] = bals

    print(f"\n전체 평가 완료 — {time.time() - t0:.0f}초")

    # ── 리포트 ──
    md = ["# 챔피언로드 ② 배틀 프론티어 — v1.x 라인업 본판정 (4 sampler × 5시드)", ""]
    md.append(f"- 라인업: {len(lineup)}명 + 기준선 4인방")
    md.append(f"- 시드 100만원 × 평가 2년 (504거래일) — arena별 평행세계")
    md.append(f"- 시드 {SEED}, 블록 21일 부트스트랩")
    md.append("")

    md.append("## 평균 종료 잔고 (arena × 후보, 단위 만원)")
    md.append("")
    head = "| 후보 | 라벨 | " + " | ".join(f"{a} ({n})" for a, _, n in arenas) + " |"
    md.append(head)
    md.append("|---|---|" + "---:|" * len(arenas))
    for name, _, _, label, _ in weight_candidates:
        cells = []
        for arena, _, _ in arenas:
            mean = int(np.mean(arena_results[arena][name]))
            cells.append(f"{mean // 10000:,}")
        md.append(f"| {name} | {label} | " + " | ".join(cells) + " |")
    for name, _ in baselines:
        cells = []
        for arena, _, _ in arenas:
            mean = int(np.mean(arena_results[arena][name]))
            cells.append(f"{mean // 10000:,}")
        md.append(f"| {name} | 기준선 | " + " | ".join(cells) + " |")
    md.append("")

    md.append("## 세계 1등 카운트 (arena별)")
    md.append("")
    md.append("| 후보 | 라벨 | " + " | ".join(f"{a}" for a, _, _ in arenas) + " | 총합 |")
    md.append("|---|---|" + "---:|" * (len(arenas) + 1))
    wins_total = {n: 0 for n in all_names}
    wins_by_arena: dict[str, dict[str, int]] = {}
    for arena, _, n_worlds in arenas:
        wins = {n: 0 for n in all_names}
        for i in range(n_worlds):
            winner = max(all_names, key=lambda n: arena_results[arena][n][i])
            wins[winner] += 1
            wins_total[winner] += 1
        wins_by_arena[arena] = wins
    # 라인업 + 기준선 모두 표시 (총합 큰 순)
    for name in sorted(all_names, key=lambda n: -wins_total[n]):
        label = next((c[3] for c in weight_candidates if c[0] == name), "기준선")
        cells = [f"{wins_by_arena[arena][name]}" for arena, _, _ in arenas]
        md.append(f"| {name} | {label} | " + " | ".join(cells)
                  + f" | **{wins_total[name]}** |")
    md.append("")

    md.append("## bear arena 분포 — 하위 5% / 중앙값 / 손실비율")
    md.append("")
    md.append("| 후보 | 라벨 | 하위 5% | 중앙값 | 평균 | 손실 비율 |")
    md.append("|---|---|---:|---:|---:|---:|")
    for name in all_names:
        b = arena_results["bear"][name]
        p5 = int(np.percentile(b, 5))
        med = int(np.median(b))
        mean = int(np.mean(b))
        lose_ratio = sum(1 for v in b if v < SEED_KRW) / len(b)
        label = next((c[3] for c in weight_candidates if c[0] == name), "기준선")
        md.append(f"| {name} | {label} | {p5 // 10000:,} | {med // 10000:,} | "
                  f"{mean // 10000:,} | {lose_ratio:.0%} |")
    md.append("")
    md.append("> 손실 비율 = 100만원보다 잔고 적은 세계 비율 (위기에 깨지는 빈도).")
    md.append("")

    # 스페셜리스트 자기 arena 본판정 — ★bear는 bear에서 ★rebound는 rebound에서
    md.append("## 스페셜리스트 본판정 (자기 전문 arena 1등 카운트)")
    md.append("")
    md.append("| 스페셜리스트 | 자기 arena | 1등 | 자기 arena 평균잔고 |")
    md.append("|---|---|---:|---:|")
    for name, _, _, label, is_spec in weight_candidates:
        if not is_spec:
            continue
        # 라벨 형식: "★bear" / "★rebound" / "★crash_v" / "★bull" / "★chop"
        # bear/rebound는 arena 같음. crash_v/bull/chop은 전천후로 대체.
        if "bear" in label:
            home = "bear"
        elif "rebound" in label:
            home = "rebound"
        else:
            home = "전천후"
        wins = wins_by_arena[home][name]
        mean = int(np.mean(arena_results[home][name]))
        md.append(f"| {name} | {home} ({label}) | {wins} | {mean // 10000:,} |")
    md.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"\nsaved: {OUT_MD.relative_to(_ROOT)}")

    # 콘솔 요약 — 1등 총합 top 10 + 스페셜리스트 자기 arena 통과 여부
    print("\n=== 1등 총합 TOP 10 ===")
    for name in sorted(all_names, key=lambda n: -wins_total[n])[:10]:
        label = next((c[3] for c in weight_candidates if c[0] == name), "기준선")
        print(f"  {name:<20} {label:<14} {wins_total[name]:>3}회")

    print("\n=== 스페셜리스트 본판정 (자기 arena 승률) ===")
    for name, _, _, label, is_spec in weight_candidates:
        if not is_spec:
            continue
        if "bear" in label:
            home, n_worlds = "bear", N_WORLDS_REGIME
        elif "rebound" in label:
            home, n_worlds = "rebound", N_WORLDS_REGIME
        else:
            home, n_worlds = "전천후", N_WORLDS_ALL
        wins = wins_by_arena[home][name]
        wr = wins / n_worlds * 100
        verdict = "🎫 통과" if wr >= 5 else "🪑 벤치"   # 자기 arena 승률 5% 이상=의미있는 발휘
        print(f"  {name:<20} {label:<14} {home} {wins:>3}/{n_worlds} ({wr:>4.1f}%) {verdict}")


if __name__ == "__main__":
    print("이 모듈은 단독 실행 안 함 — champion_road_lineup_v1x.py가 build_lineup() "
          "결과를 인자로 넘겨 main(lineup) 호출합니다.")
    sys.exit(2)
