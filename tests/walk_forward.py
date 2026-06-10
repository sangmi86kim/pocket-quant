"""
walk_forward.py - 워크 포워드 테스트 (선택 과정의 아웃오브샘플 검증)

[무엇을 재나]
체육관 백테스트는 "그 구간을 이미 보고" 최강 조합을 고른 인샘플 성적이다.
실전은 다르다: 과거만 보고 골라서 미래에 내보내야 한다. 이 스크립트는 그 과정을
역사 전체에 대해 반복한다:

    [과거 TRAIN_YEARS년으로 전 조합(63개) 채점 → 1등 선발] → [다음 1년에 출전(OOS)]
    → 1년 전진, 반복 (1999 ~ 현재)

선발 기준은 본 게임과 동일(battle.fight의 적합도 = ATK/DEF/SKILL), 거래비용 포함.
워밍업도 본 게임과 동일하게 평가 구간 앞 WARMUP_DAYS만 잘라 쓴다.

[합격 기준 — 4시대 룰과 동일 철학]
이어붙인 OOS 곡선이 단순보유(B&H) 대비:
  ① 방어: MDD가 더 얕다          (허용오차 없음)
  ② 효율: 샤프가 동급 이상        (Sharpe >= B&H - 0.05, 측정 노이즈 허용)
수익(CAGR)이 B&H에 지는 건 실패가 아니다 — 이 풀은 방어로 돈값을 하는 풀이다.

실행: 프로젝트 루트에서  python tests/walk_forward.py
"""
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.backend.core.models import Gym, Strategy
from app.backend.engine.battle import TRADE_COST, TRADING_DAYS, fight
from app.backend.genes.signals import ALL_GENES, GENE_SIGNALS, combine_positions
from app.backend.market.data import LoadedGym, WARMUP_DAYS, get_prices

# ── 설정 ──────────────────────────────────────────
TICKER = "SPY"
DATA_START, DATA_END = "1994-01-01", "2026-06-09"
TRAIN_YEARS = 4            # 선발에 쓰는 과거 길이
FIRST_TEST_YEAR = 1999     # 첫 출전 연도 (1995~1998 훈련, 1994는 지표 워밍업)
MIN_TEST_DAYS = 60         # 마지막 부분 연도가 이보다 짧으면 제외
SHARPE_TOL = 0.05          # 효율 판정 허용오차 (노이즈)

ALL_COMBOS = [list(c) for k in range(1, len(ALL_GENES) + 1)
              for c in combinations(ALL_GENES, k)]


def _window(prices: pd.Series, start: str, end: str) -> pd.Series:
    """본 게임(data.load_gym)과 동일하게: 평가 시작 전 워밍업 버퍼를 포함해 자른다."""
    s = pd.Timestamp(start) - pd.Timedelta(days=WARMUP_DAYS)
    return prices.loc[s:pd.Timestamp(end)]


def _fight_window(genes: list[str], prices: pd.Series, start: str, end: str):
    """임시 체육관 하나를 만들어 본 게임 엔진(fight)으로 채점한다."""
    gym = Gym(f"{start}~{end}", difficulty=0, volatility=0,
              ticker=TICKER, start=start, end=end)
    loaded = LoadedGym(gym=gym, prices=_window(prices, start, end))
    return fight(Strategy(genes=genes, name="+".join(genes)), loaded)


def _oos_returns(genes: list[str], prices: pd.Series, start: str, end: str) -> pd.Series:
    """OOS 구간의 일별 전략 수익(비용 포함) — 전 구간 이어붙이기(스티칭)용.
    공식은 battle.fight와 동일 (포지션 lag 1 + 턴오버 × TRADE_COST)."""
    win = _window(prices, start, end)
    pos = combine_positions([GENE_SIGNALS[g](win) for g in genes]).shift(1)
    ret = pos * win.pct_change() - pos.diff().abs() * TRADE_COST
    mask = (ret.index >= pd.Timestamp(start)) & (ret.index <= pd.Timestamp(end))
    return ret[mask].dropna()


def _perf(returns: pd.Series) -> tuple[float, float, float]:
    """(CAGR, MDD, 샤프) — 이어붙인 수익 시계열의 종합 성적."""
    eq = (1 + returns).cumprod()
    cagr = float(eq.iloc[-1] ** (TRADING_DAYS / len(returns)) - 1)
    mdd = float((eq / eq.cummax() - 1).min())
    std = returns.std()
    sharpe = float(returns.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0
    return cagr, mdd, sharpe


def run_walk_forward() -> bool:
    prices = get_prices(TICKER, DATA_START, DATA_END)
    last_date = prices.index.max()

    print(f"=== 워크 포워드: 과거 {TRAIN_YEARS}년 선발 -> 다음 1년 출전 "
          f"({TICKER}, 수수료 {TRADE_COST:.1%}/편도) ===\n")
    print(f"{'출전연도':<10} {'선발 조합':<26} {'OOS수익':>8} {'B&H':>8}"
          f" {'OOS MDD':>8} {'B&H MDD':>8}  판정")

    picks = Counter()
    oos_parts, bh_parts = [], []
    beat_ret = beat_mdd = n_folds = 0

    for year in range(FIRST_TEST_YEAR, last_date.year + 1):
        train_start = f"{year - TRAIN_YEARS}-01-01"
        train_end = f"{year - 1}-12-31"
        test_start = f"{year}-01-01"
        test_end = min(pd.Timestamp(f"{year}-12-31"), last_date).strftime("%Y-%m-%d")

        # (1) 선발: 훈련 구간에서 전 조합 채점 (본 게임과 같은 적합도)
        scored = [(genes, _fight_window(genes, prices, train_start, train_end).stats.fitness)
                  for genes in ALL_COMBOS]
        best_genes, _ = max(scored, key=lambda x: x[1])

        # (2) 출전: 다음 1년 (선발에 안 쓴 데이터)
        oos = _oos_returns(best_genes, prices, test_start, test_end)
        if len(oos) < MIN_TEST_DAYS:
            continue
        win = _window(prices, test_start, test_end)
        bh = win.pct_change()[oos.index].dropna()

        strat_ret = float((1 + oos).prod() - 1)
        bh_ret = float((1 + bh).prod() - 1)
        strat_mdd = float(((1 + oos).cumprod() / (1 + oos).cumprod().cummax() - 1).min())
        bh_mdd = float(((1 + bh).cumprod() / (1 + bh).cumprod().cummax() - 1).min())

        n_folds += 1
        beat_ret += strat_ret > bh_ret
        beat_mdd += strat_mdd > bh_mdd
        picks["+".join(best_genes)] += 1
        oos_parts.append(oos)
        bh_parts.append(bh)

        flag = ("수익승" if strat_ret > bh_ret else "      ") + \
               (" 방어승" if strat_mdd > bh_mdd else "")
        print(f"{year:<10} {'+'.join(best_genes):<26} {strat_ret:>+7.1%} {bh_ret:>+7.1%}"
              f" {strat_mdd:>8.1%} {bh_mdd:>8.1%}  {flag}")

    # ── 종합 ──
    oos_all, bh_all = pd.concat(oos_parts), pd.concat(bh_parts)
    sc, sm, ss = _perf(oos_all)
    bc, bm, bs = _perf(bh_all)

    print(f"\n=== 선발 빈도 (총 {n_folds}회) ===")
    for name, cnt in picks.most_common():
        print(f"  {cnt:2}회  {name}")

    print(f"\n=== 종합 (OOS {n_folds}년 이어붙임) ===")
    print(f"  연도별: 수익 우위 {beat_ret}/{n_folds}년 · 방어(MDD) 우위 {beat_mdd}/{n_folds}년")
    print(f"  전략  : CAGR {sc:+6.1%}  MDD {sm:6.1%}  Sharpe {ss:.2f}")
    print(f"  B&H   : CAGR {bc:+6.1%}  MDD {bm:6.1%}  Sharpe {bs:.2f}")

    defense = sm > bm
    efficiency = ss >= bs - SHARPE_TOL
    print(f"\n=== 판정 ===")
    print(f"  방어 (OOS MDD가 B&H보다 얕음)        : {'PASS' if defense else 'FAIL'} ({sm:.1%} vs {bm:.1%})")
    print(f"  효율 (OOS Sharpe >= B&H - {SHARPE_TOL}) : {'PASS' if efficiency else 'FAIL'} ({ss:.2f} vs {bs:.2f})")
    return defense and efficiency


if __name__ == "__main__":
    sys.exit(0 if run_walk_forward() else 1)
