"""
walk_forward.py - 워크 포워드 테스트 (선택 과정의 아웃오브샘플 검증)

[무엇을 재나]
체육관 백테스트는 "그 구간을 이미 보고" 최강 조합을 고른 인샘플 성적이다.
실전은 다르다: 과거만 보고 골라서 미래에 내보내야 한다. 이 스크립트는 그 과정을
역사 전체에 대해 반복한다:

    [과거 train_years년으로 전 조합(63개) 채점 → 1등 선발] → [다음 1년에 출전(OOS)]
    → 1년 전진, 반복

선발 기준은 본 게임과 동일(battle.fight의 적합도), 거래비용·워밍업도 동일.

[합격 기준 — 4시대 룰과 동일 철학]
이어붙인 OOS 곡선이 단순보유(B&H) 대비:
  ① 방어: MDD가 더 얕다          (허용오차 없음)
  ② 효율: 샤프가 동급 이상        (Sharpe >= B&H - 0.05, 측정 노이즈 허용)
수익(CAGR)이 B&H에 지는 건 실패가 아니다 — 이 풀은 방어로 돈값을 하는 풀이다.

[위기장/평시장 분리 리포트]
"이 전략이 정확히 어느 장에서 돈값을 하는가"를 자동 명시한다.
위기의 해 = 그 해 B&H가 마이너스이거나 연중 MDD가 -15% 이하인 해 (B&H만으로 정의
되는 객관 기준 — 전략 성적과 무관하므로 리포트 분리용으로 안전).

[민감도 실험용 파라미터]
run_walk_forward(ticker=, train_years=, trade_cost=, verbose=) — train length /
비용 / 자산(QQQ) 민감도는 이 함수를 바꿔 부르면 된다. trade_cost는 본 게임 엔진
(battle.TRADE_COST)을 함께 바꿔 선발·출전 모두에 일관 적용한다.

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
from app.backend.engine import battle
from app.backend.genes.signals import ALL_GENES, GENE_SIGNALS, combine_positions
from app.backend.market.data import LoadedGym, WARMUP_DAYS, get_prices

# ── 기본 설정 ──────────────────────────────────────
DATA_END = "2026-06-09"
DATA_START = {"SPY": "1994-01-01", "QQQ": "1999-03-10"}   # 상장일 기준
MIN_TEST_DAYS = 60          # 마지막 부분 연도가 이보다 짧으면 제외
SHARPE_TOL = 0.05           # 효율 판정 허용오차 (노이즈)
CRISIS_MDD = -0.15          # 위기의 해 판정: B&H 연중 MDD가 이 이하 (또는 연수익 음수)

ALL_COMBOS = [list(c) for k in range(1, len(ALL_GENES) + 1)
              for c in combinations(ALL_GENES, k)]


def _window(prices: pd.Series, start: str, end: str) -> pd.Series:
    """본 게임(data.load_gym)과 동일하게: 평가 시작 전 워밍업 버퍼를 포함해 자른다."""
    s = pd.Timestamp(start) - pd.Timedelta(days=WARMUP_DAYS)
    return prices.loc[s:pd.Timestamp(end)]


def _fight_window(genes: list[str], prices: pd.Series, ticker: str, start: str, end: str):
    """임시 체육관 하나를 만들어 본 게임 엔진(fight)으로 채점한다."""
    gym = Gym(f"{start}~{end}", difficulty=0, volatility=0,
              ticker=ticker, start=start, end=end)
    loaded = LoadedGym(gym=gym, prices=_window(prices, start, end))
    return battle.fight(Strategy(genes=genes, name="+".join(genes)), loaded)


def _oos_returns(genes: list[str], prices: pd.Series, start: str, end: str) -> pd.Series:
    """OOS 구간의 일별 전략 수익(비용 포함) — 전 구간 이어붙이기(스티칭)용.
    공식은 battle.fight와 동일 (포지션 lag 1 + 턴오버 × battle.TRADE_COST)."""
    win = _window(prices, start, end)
    pos = combine_positions([GENE_SIGNALS[g](win) for g in genes]).shift(1)
    ret = pos * win.pct_change() - pos.diff().abs() * battle.TRADE_COST
    mask = (ret.index >= pd.Timestamp(start)) & (ret.index <= pd.Timestamp(end))
    return ret[mask].dropna()


def _perf(returns: pd.Series) -> tuple[float, float, float]:
    """(CAGR, MDD, 샤프) — 수익 시계열의 종합 성적."""
    eq = (1 + returns).cumprod()
    cagr = float(eq.iloc[-1] ** (battle.TRADING_DAYS / len(returns)) - 1)
    mdd = float((eq / eq.cummax() - 1).min())
    std = returns.std()
    sharpe = float(returns.mean() / std * np.sqrt(battle.TRADING_DAYS)) if std > 0 else 0.0
    return cagr, mdd, sharpe


def run_walk_forward(ticker: str = "SPY", train_years: int = 4,
                     trade_cost: float | None = None, verbose: bool = True,
                     first_test_year: int | None = None) -> dict:
    """워크 포워드 1회 실행. trade_cost를 주면 본 게임 엔진 비용도 함께 바꾼다.
    first_test_year: OOS 시작 연도 고정 — train_years가 다른 실행끼리 공정 비교용
    (안 주면 train 길이에 따라 시작 연도가 달라져 OOS 기간 자체가 달라진다).
    반환: 요약 dict (민감도 스윕에서 표로 모으는 용도)."""
    cost_backup = battle.TRADE_COST
    if trade_cost is not None:
        battle.TRADE_COST = trade_cost
    try:
        return _run(ticker, train_years, verbose, first_test_year)
    finally:
        battle.TRADE_COST = cost_backup


def _run(ticker: str, train_years: int, verbose: bool,
         first_test_year: int | None) -> dict:
    prices = get_prices(ticker, DATA_START[ticker], DATA_END)
    last_date = prices.index.max()
    # 첫 출전 기본값 = 데이터 시작 + 워밍업 1년 + 훈련 train_years년
    min_first = prices.index.min().year + 1 + train_years
    first_test_year = max(first_test_year or min_first, min_first)

    def say(msg=""):
        if verbose:
            print(msg)

    say(f"=== 워크 포워드: 과거 {train_years}년 선발 -> 다음 1년 출전 "
        f"({ticker}, 수수료 {battle.TRADE_COST:.2%}/편도) ===\n")
    say(f"{'출전연도':<10} {'선발 조합':<26} {'OOS수익':>8} {'B&H':>8}"
        f" {'OOS MDD':>8} {'B&H MDD':>8}  판정")

    picks = Counter()
    folds = []                       # (year, crisis?, s_ret, b_ret, s_mdd, b_mdd)
    oos_parts, bh_parts = [], []

    for year in range(first_test_year, last_date.year + 1):
        train_start = f"{year - train_years}-01-01"
        train_end = f"{year - 1}-12-31"
        test_start = f"{year}-01-01"
        test_end = min(pd.Timestamp(f"{year}-12-31"), last_date).strftime("%Y-%m-%d")

        # (1) 선발: 훈련 구간에서 전 조합 채점 (본 게임과 같은 적합도)
        scored = [(g, _fight_window(g, prices, ticker, train_start, train_end).stats.fitness)
                  for g in ALL_COMBOS]
        best_genes, _ = max(scored, key=lambda x: x[1])

        # (2) 출전: 다음 1년 (선발에 안 쓴 데이터)
        oos = _oos_returns(best_genes, prices, test_start, test_end)
        if len(oos) < MIN_TEST_DAYS:
            continue
        win = _window(prices, test_start, test_end)
        bh = win.pct_change()[oos.index].dropna()

        s_ret = float((1 + oos).prod() - 1)
        b_ret = float((1 + bh).prod() - 1)
        s_mdd = float(((1 + oos).cumprod() / (1 + oos).cumprod().cummax() - 1).min())
        b_mdd = float(((1 + bh).cumprod() / (1 + bh).cumprod().cummax() - 1).min())
        crisis = (b_ret < 0) or (b_mdd <= CRISIS_MDD)

        picks["+".join(best_genes)] += 1
        folds.append((year, crisis, s_ret, b_ret, s_mdd, b_mdd))
        oos_parts.append(oos)
        bh_parts.append(bh)

        flag = ("수익승" if s_ret > b_ret else "      ") + \
               (" 방어승" if s_mdd > b_mdd else "")
        crisis_mark = "*" if crisis else " "
        say(f"{year}{crisis_mark:<7} {'+'.join(best_genes):<26}"
            f" {s_ret:>+7.1%} {b_ret:>+7.1%} {s_mdd:>8.1%} {b_mdd:>8.1%}  {flag}")

    # ── 종합 ──
    n = len(folds)
    oos_all, bh_all = pd.concat(oos_parts), pd.concat(bh_parts)
    sc, sm, ss = _perf(oos_all)
    bc, bm, bs = _perf(bh_all)
    beat_ret = sum(f[2] > f[3] for f in folds)
    beat_mdd = sum(f[4] > f[5] for f in folds)
    rev_picked = sum(cnt for name, cnt in picks.items() if "REV" in name)

    say(f"\n=== 선발 빈도 (총 {n}회) ===")
    if verbose:
        for name, cnt in picks.most_common():
            say(f"  {cnt:2}회  {name}")
    say(f"  REV 포함 조합 선발률: {rev_picked}/{n}")

    # 위기장 / 평시장 분리 — "정확히 어느 장에서 돈값을 하는가"
    say(f"\n=== 위기장(*) vs 평시장 (위기 = B&H 연수익<0 또는 MDD<={CRISIS_MDD:.0%}) ===")
    groups = {}
    for label, want in (("위기장", True), ("평시장", False)):
        grp = [f for f in folds if f[1] == want]
        if not grp:
            continue
        avg_excess = float(np.mean([f[2] - f[3] for f in grp]))
        w_ret = sum(f[2] > f[3] for f in grp)
        w_mdd = sum(f[4] > f[5] for f in grp)
        groups[label] = {"n": len(grp), "excess": avg_excess,
                         "beat_ret": w_ret, "beat_mdd": w_mdd}
        say(f"  {label} {len(grp):2}년: 평균 초과수익 {avg_excess:+6.1%}/년"
            f" · 수익 우위 {w_ret}/{len(grp)} · 방어 우위 {w_mdd}/{len(grp)}")

    say(f"\n=== 종합 (OOS {n}년 이어붙임) ===")
    say(f"  연도별: 수익 우위 {beat_ret}/{n}년 · 방어(MDD) 우위 {beat_mdd}/{n}년")
    say(f"  전략  : CAGR {sc:+6.1%}  MDD {sm:6.1%}  Sharpe {ss:.2f}")
    say(f"  B&H   : CAGR {bc:+6.1%}  MDD {bm:6.1%}  Sharpe {bs:.2f}")

    defense = sm > bm
    efficiency = ss >= bs - SHARPE_TOL
    say(f"\n=== 판정 ===")
    say(f"  방어 (OOS MDD가 B&H보다 얕음)        : {'PASS' if defense else 'FAIL'} ({sm:.1%} vs {bm:.1%})")
    say(f"  효율 (OOS Sharpe >= B&H - {SHARPE_TOL}) : {'PASS' if efficiency else 'FAIL'} ({ss:.2f} vs {bs:.2f})")

    return {"ticker": ticker, "train_years": train_years, "cost": battle.TRADE_COST,
            "n_folds": n, "rev_picked": rev_picked, "beat_ret": beat_ret,
            "beat_mdd": beat_mdd, "cagr": sc, "mdd": sm, "sharpe": ss,
            "bh_cagr": bc, "bh_mdd": bm, "bh_sharpe": bs,
            "groups": groups, "passed": defense and efficiency}


if __name__ == "__main__":
    result = run_walk_forward()
    sys.exit(0 if result["passed"] else 1)
