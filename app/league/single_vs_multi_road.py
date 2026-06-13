"""챔피언로드 ①②③ — 다목적(NSGA-III TOP10) vs 단일목적(TPE 5시드) 동시 출전.

[가설] 평행세계 ②에서 단일목적 TPE 답이 다목적 NSGA-III 라인업을 이길까?
       OOS 11년·사천왕 7라운드에선 어떻게 나뉠까?

[입장]
- 현 챔피언 (다목적 옛 답, 동일가중 VOL+REV_RSI+REV_BB) 1명
- 다목적 NSGA-III v1 TOP10 — `hall_of_fame.md` 가중치 그대로 (재현 안 돌리고 박아 넣음)
- 단일목적 TPE 5시드 1등 — `single_obj_sweep.py` 결과 (시드별 1등 가중치)
- 기준선 4인방 (어플삭제맨/저축왕/성실이/돼지저금통) — 관문 어댑터가 자동/우리 inline

[관문]
- ① OOS 11년: `victory_road.run_gate1(graduates)` 외부 주입 (mean5 None OK 검증됨)
- ② 평행세계 400: arena별 후보 × 세계 잔고 매트릭스 inline (battle_frontier 함수 재사용)
- ③ 사천왕 7라운드: 라운드별 잔고 매트릭스 inline (elite_four 라운드 정의 재사용)

[산출] reports/single_vs_multi_road.md + 콘솔 종합 순위.
[실행] python app/league/single_vs_multi_road.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

from app.backend.data_io.data import LoadedGym, get_prices
from app.backend.engine.battle import (_score_position, fight_dca, fight_savings,
                                       terminal_balance)
from app.backend.genes.signals import ALL_GENES, combine_positions, positions_with_params
from app.backend.market.regime import REGIME_LABELS, dominant_regime
from app.league.battle_frontier import (DATA_START, DATA_END, N_WORLDS_ALL,
                                         N_WORLDS_REGIME, REGIME_SPANS, SEED,
                                         make_world)
from app.league.elite_four import (DATA_END as HOLDOUT_END, HOLDOUT_START, ROUNDS,
                                    TICKER, _loaded_window)
from app.league.victory_road import run_gate1

SEED_KRW = 1_000_000
OUT_MD = _ROOT / "reports" / "single_vs_multi_road.md"

# ── 다목적 NSGA-III v1 TOP10 (hall_of_fame.md 정규화 가중치 그대로) ──
# 비율만 의미하므로 raw 합이 1 아니어도 OK (combine_positions가 Σw로 정규화).
MULTI_TOP10 = [
    # (label, [DD, VOL, MA, MOM, REV_RSI, REV_BB])
    ("TOP01", [0.01, 0.09, 0.00, 0.02, 0.40, 0.49]),
    ("TOP02", [0.01, 0.14, 0.01, 0.01, 0.45, 0.38]),
    ("TOP03", [0.00, 0.14, 0.02, 0.00, 0.51, 0.32]),
    ("TOP04", [0.01, 0.14, 0.02, 0.00, 0.53, 0.30]),
    ("TOP05", [0.01, 0.14, 0.02, 0.00, 0.53, 0.30]),
    ("TOP06", [0.04, 0.08, 0.00, 0.01, 0.57, 0.30]),
    ("TOP07", [0.01, 0.13, 0.00, 0.02, 0.37, 0.47]),
    ("TOP08", [0.01, 0.15, 0.01, 0.01, 0.42, 0.40]),
    ("TOP09", [0.01, 0.17, 0.00, 0.02, 0.41, 0.39]),
    ("TOP10", [0.01, 0.17, 0.00, 0.02, 0.45, 0.35]),
]

# ── 단일목적 TPE 5시드 — single_obj_sweep.py 직전 실행 결과 (정규화 %) ──
TPE_FIVE = [
    ("TPE-s42", [0.000, 0.012, 0.000, 0.000, 0.511, 0.476]),
    ("TPE-s07", [0.000, 0.031, 0.000, 0.000, 0.507, 0.461]),
    ("TPE-s11", [0.000, 0.002, 0.000, 0.000, 0.546, 0.452]),
    ("TPE-s19", [0.000, 0.057, 0.000, 0.000, 0.424, 0.519]),
    ("TPE-s23", [0.000, 0.088, 0.000, 0.000, 0.419, 0.493]),
]

CHAMPION_W = [1.0 if g in ("VOL", "REV_RSI", "REV_BB") else 0.0 for g in ALL_GENES]


def eval_weights(weights: list[float], lw: LoadedGym) -> int:
    pos = combine_positions(positions_with_params(lw.prices), weights)
    return terminal_balance(_score_position(pos, lw), SEED_KRW)


def eval_buy_hold(lw: LoadedGym) -> int:
    pos = pd.Series(1.0, index=lw.prices.index)
    return terminal_balance(_score_position(pos, lw), SEED_KRW)


def eval_piggy(lw: LoadedGym) -> int:
    return SEED_KRW


def eval_savings(lw: LoadedGym) -> int:
    return terminal_balance(fight_savings(lw), SEED_KRW)


def eval_dca(lw: LoadedGym) -> int:
    return terminal_balance(fight_dca(lw), SEED_KRW)


def _strategies() -> list[tuple[str, list[float]]]:
    """현챔 + 다목적 TOP10 + TPE 5명 = 16명."""
    return [("현챔피언", CHAMPION_W)] + MULTI_TOP10 + TPE_FIVE


BASELINES = [("어플삭제맨", eval_buy_hold),
             ("저축왕", eval_savings),
             ("성실이", eval_dca),
             ("돼지저금통", eval_piggy)]


def _gate1_graduates() -> list[dict]:
    """victory_road.run_gate1 호환 graduates — 기준선은 그 안에서 자동 추가."""
    graduates = [{
        "name": "현챔피언", "label": "기준(동일가중)",
        "weights": CHAMPION_W, "params": {},
        "mean5": None, "specialist": False,
    }]
    for name, w in MULTI_TOP10:
        graduates.append({
            "name": name, "label": "다목적", "weights": w, "params": {},
            "mean5": None, "specialist": False,
        })
    for name, w in TPE_FIVE:
        graduates.append({
            "name": name, "label": "단일목적", "weights": w, "params": {},
            "mean5": None, "specialist": False,
        })
    return graduates


# ────────────────────────────────────────────────────────────────────
# 관문 ② 평행세계 토탈
# ────────────────────────────────────────────────────────────────────
def _gate2_lineup() -> dict[str, dict]:
    print("\n=== 관문 ② 평행세계 — 다목적 vs 단일목적 토탈 잔고 ===")
    prices = get_prices("QQQ", DATA_START, DATA_END)
    full_returns = prices.pct_change().dropna()
    regime_returns = {
        name: pd.concat([full_returns.loc[s:e] for s, e in spans])
        for name, spans in REGIME_SPANS.items()
    }
    arenas = [("전천후", None, N_WORLDS_ALL),
              ("bear", regime_returns["bear"], N_WORLDS_REGIME),
              ("rebound", regime_returns["rebound"], N_WORLDS_REGIME)]

    strategies = _strategies()
    all_names = [n for n, _ in strategies] + [n for n, _ in BASELINES]
    arena_results: dict[str, dict[str, list[int]]] = {}

    for arena, pool, n_worlds in arenas:
        print(f"  {arena} arena ({n_worlds}세계) ...", end="", flush=True)
        t0 = time.time()
        bals: dict[str, list[int]] = {n: [] for n in all_names}
        rng = np.random.default_rng(SEED)
        for _ in range(n_worlds):
            world = make_world(full_returns, rng, pool)
            for name, w in strategies:
                bals[name].append(eval_weights(w, world))
            for name, fn in BASELINES:
                bals[name].append(fn(world))
        arena_results[arena] = bals
        print(f" {time.time()-t0:.0f}s")

    # 토탈 = arena별 잔고 누적합 (단위: 원)
    totals = {}
    for name in all_names:
        totals[name] = {a: sum(arena_results[a][name]) for a in arena_results}
        totals[name]["total"] = sum(totals[name].values())

    return {"arena_results": arena_results, "totals": totals,
            "all_names": all_names, "arena_names": [a for a, _, _ in arenas]}


# ────────────────────────────────────────────────────────────────────
# 관문 ③ 사천왕 7라운드
# ────────────────────────────────────────────────────────────────────
def _gate3_rounds() -> dict[str, dict]:
    print("\n=== 관문 ③ 사천왕 7라운드 — 다목적 vs 단일목적 ===")
    prices = get_prices(TICKER, "1999-03-10", HOLDOUT_END)
    rounds = [(name, start, end, _loaded_window(prices, start, end))
              for name, start, end in ROUNDS]
    strategies = _strategies()

    balances: dict[str, dict[str, int]] = {}
    for name, w in strategies:
        balances[name] = {nm: eval_weights(w, lw) for nm, _, _, lw in rounds}
    for name, fn in BASELINES:
        balances[name] = {nm: fn(lw) for nm, _, _, lw in rounds}

    regimes = {nm: REGIME_LABELS[dominant_regime(prices, s, e)]
               for nm, s, e, _ in rounds}
    round_names = [nm for nm, _, _, _ in rounds]
    return {"balances": balances, "regimes": regimes, "round_names": round_names}


# ────────────────────────────────────────────────────────────────────
# 결과 정리
# ────────────────────────────────────────────────────────────────────
def _rank_table(label: str, totals: dict, label_of: dict[str, str]) -> list[str]:
    """후보별 토탈 잔고 표 — 순위 + 그룹 라벨(다목적/단일목적/기준선)."""
    sorted_names = sorted(totals.keys(), key=lambda n: -totals[n])
    out = [f"## {label}", "",
           "| 순위 | 후보 | 그룹 | 토탈 잔고 (만) | 차(성실이 대비) |",
           "|---:|---|---|---:|---:|"]
    dca_total = totals.get("성실이", 0)
    for i, n in enumerate(sorted_names, 1):
        tot = totals[n]
        diff = tot - dca_total
        sign = "+" if diff >= 0 else ""
        out.append(f"| {i} | {n} | {label_of.get(n, '')} | "
                   f"{tot // 10000:,} | {sign}{diff // 10000:,} |")
    out.append("")
    return out


def main() -> None:
    t_start = time.time()

    # 관문 ① — victory_road 외부 주입 (기존 기능 그대로)
    print("=== 관문 ① OOS 11년 시험장 (다목적 + 단일목적 합쳐 16명) ===\n")
    graduates = _gate1_graduates()
    run_gate1(graduates)

    # 관문 ② — 평행세계 토탈 (inline 평가)
    gate2 = _gate2_lineup()

    # 관문 ③ — 사천왕 7라운드 (inline 평가)
    gate3 = _gate3_rounds()

    # 후보 라벨 매핑 (다목적/단일목적/기준/현챔)
    label_of = {"현챔피언": "다목적(옛)"}
    for n, _ in MULTI_TOP10:
        label_of[n] = "다목적"
    for n, _ in TPE_FIVE:
        label_of[n] = "단일목적"
    for n, _ in BASELINES:
        label_of[n] = "기준선"

    # 콘솔 종합 순위
    print(f"\n=== 종합 순위 ({time.time()-t_start:.0f}초) ===")
    g2_totals = {n: gate2["totals"][n]["total"] for n in gate2["totals"]}
    g3_totals = {n: sum(b.values()) for n, b in gate3["balances"].items()}
    print("\n관문 ② 평행세계 400 토탈 — 상위 10:")
    for i, n in enumerate(sorted(g2_totals, key=lambda x: -g2_totals[x])[:10], 1):
        print(f"  {i:>2}. {n:<10} {label_of[n]:<10} {g2_totals[n] // 10000:>6,}만원")
    print("\n관문 ③ 사천왕 7라운드 토탈 — 상위 10:")
    for i, n in enumerate(sorted(g3_totals, key=lambda x: -g3_totals[x])[:10], 1):
        print(f"  {i:>2}. {n:<10} {label_of[n]:<10} {g3_totals[n] // 10000:>6,}만원")

    # MD 저장
    md = [
        "# 챔피언로드 — 다목적 NSGA-III vs 단일목적 TPE 풀라인업 경쟁",
        "",
        "도전자 21명 (현챔 1 + 다목적 TOP10 + 단일목적 TPE 5시드 + 기준선 4):",
        "- **다목적(옛)** 현챔피언 = 동일가중 VOL+REV_RSI+REV_BB",
        "- **다목적** TOP01~TOP10 = NSGA-III 6목적 (`hall_of_fame.md` 가중치)",
        "- **단일목적** TPE-s42/07/11/19/23 = 6체육관 잔고 합 max (TPE × 5 시드)",
        "- **기준선** 어플삭제맨 / 저축왕 / 성실이 / 돼지저금통",
        "",
    ]
    md += _rank_table("관문 ② 평행세계 400 토탈 잔고", g2_totals, label_of)

    # 관문 ② arena별 분해
    md += ["## 관문 ② arena별 잔고 (단위 만)", "",
           "| 후보 | 그룹 | 전천후 | bear | rebound | 합 |",
           "|---|---|---:|---:|---:|---:|"]
    for n in sorted(g2_totals, key=lambda x: -g2_totals[x]):
        cells = gate2["totals"][n]
        md.append(
            f"| {n} | {label_of[n]} | "
            f"{cells['전천후'] // 10000:,} | {cells['bear'] // 10000:,} | "
            f"{cells['rebound'] // 10000:,} | **{cells['total'] // 10000:,}** |"
        )
    md.append("")

    md += _rank_table("관문 ③ 사천왕 7라운드 토탈 잔고", g3_totals, label_of)

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nsaved: {OUT_MD.relative_to(_ROOT)}")
    print(f"총 소요: {time.time()-t_start:.0f}초")


if __name__ == "__main__":
    main()
