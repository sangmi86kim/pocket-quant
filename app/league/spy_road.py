"""SPY 챔피언로드 ① OOS 11년 + ③ 사천왕 7라운드 — 자산 SPY로 출전.

[가설] QQQ 챔피언 가중치가 SPY OOS·사천왕에서는 어떻게 되나?
       체육관 6개 인샘플(`spy_robustness.py`)에서는 어플삭제맨이 1위였다 —
       OOS·사천왕 시기에서도 NPC가 챔피언들을 누르는가?

[설계]
- 기간 정의는 `victory_road.OOS_YEARS` + `elite_four.ROUNDS` 그대로.
- 자산만 SPY로 — `_loaded_window` 함수에 SPY 가격 넘김.
- 도전자 10명 (다목적 Top5 + TPE 5시드) + NPC 4명.
- 각 시기 시드 100만원 → 종료 잔고 합 (사용자 통찰: 누적 자산이 짱).

[실행] python app/league/spy_road.py
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

import pandas as pd

from app.backend.data_io.data import LoadedGym, get_prices
from app.backend.engine.battle import (_score_position, fight_dca, fight_savings,
                                       terminal_balance)
from app.backend.genes.signals import ALL_GENES, combine_positions, positions_with_params
from app.league.elite_four import DATA_END as HOLDOUT_END, ROUNDS
from app.league.elite_four import _loaded_window as _round_window
from app.league.victory_road import OOS_YEARS
from app.league.victory_road import _loaded_window as _year_window

ASSET = "SPY"
SEED_KRW = 1_000_000
OUT_MD = _ROOT / "reports" / "spy_road.md"

MULTI_TOP5 = [
    ("TOP06", [0.04, 0.08, 0.00, 0.01, 0.57, 0.30]),
    ("TOP01", [0.01, 0.09, 0.00, 0.02, 0.40, 0.49]),
    ("TOP04", [0.01, 0.14, 0.02, 0.00, 0.53, 0.30]),
    ("TOP05", [0.01, 0.14, 0.02, 0.00, 0.53, 0.30]),
    ("TOP03", [0.00, 0.14, 0.02, 0.00, 0.51, 0.32]),
]
TPE_FIVE = [
    ("TPE-s11", [0.000, 0.002, 0.000, 0.000, 0.546, 0.452]),
    ("TPE-s42", [0.000, 0.012, 0.000, 0.000, 0.511, 0.476]),
    ("TPE-s07", [0.000, 0.031, 0.000, 0.000, 0.507, 0.461]),
    ("TPE-s19", [0.000, 0.057, 0.000, 0.000, 0.424, 0.519]),
    ("TPE-s23", [0.000, 0.088, 0.000, 0.000, 0.419, 0.493]),
]


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


BASELINES = [("어플삭제맨", eval_buy_hold),
             ("저축왕", eval_savings),
             ("성실이", eval_dca),
             ("돼지저금통", eval_piggy)]


def _evaluate_round_set(window_pairs: list[tuple[str, LoadedGym]]) -> dict[str, dict[str, int]]:
    """{후보: {시기: 잔고}}. 시기는 OOS 연도 또는 사천왕 라운드."""
    strategies = MULTI_TOP5 + TPE_FIVE
    balances: dict[str, dict[str, int]] = {}
    for name, w in strategies:
        balances[name] = {label: eval_weights(w, lw) for label, lw in window_pairs}
    for name, fn in BASELINES:
        balances[name] = {label: fn(lw) for label, lw in window_pairs}
    return balances


def _label_of(n: str) -> str:
    if n in {nm for nm, _ in MULTI_TOP5}:
        return "다목적"
    if n in {nm for nm, _ in TPE_FIVE}:
        return "단일목적"
    return "NPC"


def _print_rank(title: str, totals: dict[str, int], dca_sum: int) -> None:
    rank = sorted(totals.keys(), key=lambda n: -totals[n])
    print(f"\n=== {title} ===")
    print(f"  {'순위':<4}{'후보':<12}{'그룹':<8}{'합':>9}  {'성실이 차':>8}")
    for i, n in enumerate(rank, 1):
        diff = totals[n] - dca_sum
        sgn = "+" if diff >= 0 else ""
        print(f"  {i:<4}{n:<12}{_label_of(n):<8}{totals[n]//10000:>7,}만  "
              f"{sgn}{diff//10000:>5,}만")


def _md_rank(title: str, totals: dict[str, int], dca_sum: int) -> list[str]:
    out = [f"## {title}", "",
           "| 순위 | 후보 | 그룹 | 잔고 합 (만) | 성실이 차 |",
           "|---:|---|---|---:|---:|"]
    rank = sorted(totals.keys(), key=lambda n: -totals[n])
    for i, n in enumerate(rank, 1):
        diff = totals[n] - dca_sum
        sgn = "+" if diff >= 0 else ""
        out.append(f"| {i} | {n} | {_label_of(n)} | {totals[n]//10000:,} | {sgn}{diff//10000:,} |")
    out.append("")
    return out


def main() -> None:
    t0 = time.time()
    print(f"=== SPY 챔피언로드 ① OOS + ③ 사천왕 ===")
    print(f"자산: {ASSET} · 시드 100만원 × 매 시기 독립 · 도전자 10 + NPC 4\n")

    # 한 번에 받기 — OOS·사천왕 둘 다 cover
    prices = get_prices(ASSET, "1999-03-10", HOLDOUT_END)

    # ── 관문 ① OOS 11년 ──
    oos_windows = [(str(y), _year_window(prices, y)) for y in OOS_YEARS]
    bal_oos = _evaluate_round_set(oos_windows)
    totals_oos = {n: sum(b.values()) for n, b in bal_oos.items()}
    dca_oos = totals_oos["성실이"]
    _print_rank(f"관문 ① OOS 11년 ({len(OOS_YEARS)}연도 합산)", totals_oos, dca_oos)

    # ── 관문 ③ 사천왕 7라운드 ──
    round_windows = [(nm, _round_window(prices, s, e)) for nm, s, e in ROUNDS]
    bal_e4 = _evaluate_round_set(round_windows)
    totals_e4 = {n: sum(b.values()) for n, b in bal_e4.items()}
    dca_e4 = totals_e4["성실이"]
    _print_rank(f"관문 ③ 사천왕 {len(ROUNDS)}라운드 합산", totals_e4, dca_e4)

    # MD 저장
    md = [
        f"# SPY 챔피언로드 ① OOS + ③ 사천왕 — 자산-횡단 검증",
        "",
        f"자산: **{ASSET}** · 시드 100만원 × 매 시기 독립.",
        f"도전자 10명 (다목적 Top5 + 단일목적 TPE 5시드) + NPC 4명.",
        "",
    ]
    md += _md_rank(f"관문 ① OOS 11년 ({len(OOS_YEARS)}연도 합산)", totals_oos, dca_oos)
    md += _md_rank(f"관문 ③ 사천왕 {len(ROUNDS)}라운드 합산", totals_e4, dca_e4)

    # 라운드별 매트릭스 (사천왕만 — OOS는 너무 가로로 김)
    md += ["## 사천왕 라운드별 잔고 (단위 만)", "",
           "| 후보 | 그룹 | " + " | ".join(nm for nm, _, _ in ROUNDS) + " | 합 |",
           "|---|---|" + "---:|" * (len(ROUNDS) + 1)]
    rank_e4 = sorted(totals_e4.keys(), key=lambda n: -totals_e4[n])
    for n in rank_e4:
        cells = " | ".join(f"{bal_e4[n][nm]//10000:,}" for nm, _, _ in ROUNDS)
        md.append(f"| {n} | {_label_of(n)} | {cells} | **{totals_e4[n]//10000:,}** |")
    md.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nsaved: {OUT_MD.relative_to(_ROOT)} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
