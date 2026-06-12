"""챔피언로드 ① OOS 11년 — 챔피언/Top10 vs 기준선 4인방 풀비교 (06-13).

기준선 3인방 + 성실이를 OOS 시험장에 같이 보내 100만원 시드 잔고 비교:
  - 어플삭제맨: B&H (항상 풀매수, 거래비용 0.1% 한 번)
  - 저축왕: 연 3% 무위험 복리 (낙폭 0)
  - 돼지저금통: 전부 현금 (수익 0, 금리 0)
  - 성실이: 일별 DCA (무비용 — 토스 자동 모으기)

산출: tools/outputs/champion_road_with_baselines.md (+ .json)
실행: python tools/baselines_comparison.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd

from app.backend.engine.battle import (_score_position, fight_dca, fight_savings,
                                         terminal_balance)
from app.backend.genes.signals import ALL_GENES, combine_positions, positions_with_params
from app.backend.market.regime import REGIME_LABELS, dominant_regime
from app.backend.market.data import LoadedGym, get_prices
from tests.victory_road import OOS_YEARS, TICKER, _loaded_window

SEED_KRW = 1_000_000
TOP10_JSON = _ROOT / "tools" / "outputs" / "top10_champions.json"
OUT_MD = _ROOT / "tools" / "outputs" / "champion_road_with_baselines.md"


def evaluate_strategy(weights: list[float], loaded: LoadedGym) -> int:
    """가중치형 후보의 종료 잔고."""
    pos = combine_positions(positions_with_params(loaded.prices), weights)
    res = _score_position(pos, loaded)
    return terminal_balance(res, SEED_KRW)


def evaluate_buy_hold(loaded: LoadedGym) -> int:
    """어플삭제맨: 항상 풀매수 1.0 — 동일 채점기로 첫날 매수 비용(0.1%) 차감."""
    pos = pd.Series(1.0, index=loaded.prices.index)
    res = _score_position(pos, loaded)
    return terminal_balance(res, SEED_KRW)


def evaluate_piggy_bank(loaded: LoadedGym) -> int:
    """돼지저금통: 전부 현금, 금리 0 — 시드 그대로."""
    return SEED_KRW


def evaluate_savings(loaded: LoadedGym) -> int:
    """저축왕: 연 3% 무위험 복리 (낙폭 0)."""
    return terminal_balance(fight_savings(loaded), SEED_KRW)


def evaluate_dca(loaded: LoadedGym) -> int:
    """성실이: 일별 DCA, 무비용."""
    return terminal_balance(fight_dca(loaded), SEED_KRW)


def main() -> None:
    prices = get_prices(TICKER, "1999-03-10", "2026-06-09")
    loadeds = {y: _loaded_window(prices, y) for y in OOS_YEARS}

    top10 = json.loads(TOP10_JSON.read_text(encoding="utf-8"))
    champion_w = [1.0 if g in ("VOL", "REV_RSI", "REV_BB") else 0.0
                   for g in ALL_GENES]
    strategies = [("현챔피언", champion_w)]
    for i, t in enumerate(top10, 1):
        w = [t["params"][f"w_{g}"] for g in ALL_GENES]
        strategies.append((f"TOP{i:02d}", w))

    baselines = [("어플삭제맨", evaluate_buy_hold),
                 ("저축왕", evaluate_savings),
                 ("성실이", evaluate_dca),
                 ("돼지저금통", evaluate_piggy_bank)]

    balances: dict[str, dict[int, int]] = {}
    for name, w in strategies:
        balances[name] = {y: evaluate_strategy(w, loadeds[y]) for y in OOS_YEARS}
    for name, fn in baselines:
        balances[name] = {y: fn(loadeds[y]) for y in OOS_YEARS}

    regimes = {}
    for y in OOS_YEARS:
        regime_en = dominant_regime(prices, f"{y}-01-01", f"{y}-12-31")
        regimes[y] = REGIME_LABELS[regime_en]

    md = ["# 챔피언로드 ① OOS — 풀라인업 vs 기준선 4인방 (06-13)", ""]
    md.append("- 시험장: 안 본 OOS 11개 연도 (2003~2019, hold-out 제외)")
    md.append("- 시드: 매년 100만원 새로 (단위 만원)")
    md.append("- 국면 라벨: Regime_Scanner 정의 (50/200 MA + 60일 수익률 + 변동성 백분위)")
    md.append("")

    md.append("## 종합 — OOS 11년 잔고 합 (단위 만원)")
    md.append("")
    md.append("| 순위 | 후보 | 합 | 차(성실이 대비) |")
    md.append("|---:|---|---:|---:|")
    dca_sum = sum(balances["성실이"].values())
    totals = [(n, sum(balances[n].values())) for n in balances]
    totals.sort(key=lambda x: -x[1])
    for i, (n, t) in enumerate(totals, 1):
        diff = t - dca_sum
        sign = "+" if diff >= 0 else ""
        md.append(f"| {i} | {n} | {t // 10000:,} | {sign}{diff // 10000:,} |")
    md.append("")

    md.append("## 매년 잔고 (시드 100만원, 단위 만원)")
    md.append("")
    year_head = " | ".join(f"{y - 2000:02d}" for y in OOS_YEARS)
    regime_head = " | ".join(regimes[y][:2] for y in OOS_YEARS)
    md.append(f"| 후보 | {year_head} | 합 |")
    md.append(f"| 국면 | {regime_head} | |")
    md.append("|---|" + "---:|" * (len(OOS_YEARS) + 1))
    order = (["현챔피언"] + [f"TOP{i:02d}" for i in range(1, 11)]
             + ["어플삭제맨", "저축왕", "성실이", "돼지저금통"])
    for name in order:
        cells = " | ".join(f"{balances[name][y] // 10000:,}" for y in OOS_YEARS)
        total = sum(balances[name].values()) // 10000
        md.append(f"| {name} | {cells} | **{total:,}** |")
    md.append("")

    md.append("## 매년 1등 (전체 풀에서)")
    md.append("")
    md.append("| 연도 | 국면 | 1등 | 잔고 | 성실이 | 차 |")
    md.append("|---|---|---|---:|---:|---:|")
    win_counts: dict[str, int] = {}
    for y in OOS_YEARS:
        winner = max(balances, key=lambda n: balances[n][y])
        win_counts[winner] = win_counts.get(winner, 0) + 1
        dca_bal = balances["성실이"][y]
        win_bal = balances[winner][y]
        diff = win_bal - dca_bal
        sign = "+" if diff >= 0 else ""
        md.append(f"| {y} | {regimes[y]} | {winner} | "
                  f"{win_bal // 10000:,}만 | {dca_bal // 10000:,}만 | "
                  f"{sign}{diff // 10000:,}만 |")
    md.append("")

    md.append("**1등 카운트 (전체 풀):**")
    for n, c in sorted(win_counts.items(), key=lambda kv: -kv[1]):
        md.append(f"- {n}: {c}회")
    md.append("")

    regime_dist: dict[str, int] = {}
    for r in regimes.values():
        regime_dist[r] = regime_dist.get(r, 0) + 1
    md.append("**OOS 국면 분포 (11년):**")
    for r, c in sorted(regime_dist.items(), key=lambda kv: -kv[1]):
        md.append(f"- {r}: {c}회")
    md.append("")
    md.append("⚠️ OOS 11년이 다 같은 국면이면 차별점 안 나타남 — "
              "그래서 챔피언/top10 잔고가 비슷한 것.")
    md.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    OUT_JSON = OUT_MD.with_suffix(".json")
    OUT_JSON.write_text(json.dumps(
        {"balances": {n: {str(y): v for y, v in b.items()}
                       for n, b in balances.items()},
         "regimes": {str(y): r for y, r in regimes.items()},
         "win_counts": win_counts},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {OUT_MD} (+ .json)")
    print("\n종합 순위:")
    for i, (n, t) in enumerate(totals[:6], 1):
        print(f"  {i}. {n} {t // 10000:,}만원")


if __name__ == "__main__":
    main()
