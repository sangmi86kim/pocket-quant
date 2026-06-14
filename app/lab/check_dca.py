"""
check_dca.py - DCA 기준선 진단 (NSGA-III 목적함수 재료의 첫 계측)

[무엇을 재나]
이길 대상을 '단순보유(B&H)'에서 '사용자의 실제 적립 머신(매일 일정액 DCA)'으로
바꾸면 풍경이 어떻게 달라지는지 본다:
  1. 체육관별 DCA 기준선 자체의 성적 (vs B&H)
  2. 전 조합(63) + 기준선(CASH/FULL)을 score_vs_dca로 줄세우기
     score_vs_dca = 0.4×수익차 + 0.4×낙폭개선 + 0.2×샤프차 (양수 = DCA보다 나음)
  3. 현 챔피언(REV_BB)의 체육관별 상세

[주의] 이 점수는 GA 적합도가 아니다 — NSGA-III(다목적)의 목적함수 재료다.
체육관별 점수를 평균으로 뭉개면 또 "최약 국면 은폐"가 생기므로, 여기서는
평균과 최약을 둘 다 표시만 하고 결합은 NSGA-III(벡터 그대로)에 맡긴다.

실행: 프로젝트 루트에서  python -m app.lab.check_dca
"""
from itertools import combinations

import pandas as pd

from app.academy.exam import all_gyms
from app.pocket import signals
from app.pocket.battle import challenge, fight_dca, fight_savings, score_vs_dca
from app.pocket.models import Strategy
from app.pocket.signals import ALL_GENES
from app.world.data_loader import load_gyms


def main() -> None:
    # 기준선 유전자 등록 (test_baselines.py와 동일 패턴 — ALL_GENES엔 안 넣음)
    signals.GENE_SIGNALS["CASH"] = lambda p: pd.Series(0.0, index=p.index)
    signals.GENE_SIGNALS["FULL"] = lambda p: pd.Series(1.0, index=p.index)

    loaded = load_gyms(all_gyms())
    dca = {lg.gym.name: fight_dca(lg) for lg in loaded}

    # ── 1. 체육관별 DCA 기준선 vs B&H ──
    print("=== 1. 체육관별 라이벌 '성실이'(DCA) 기준선 (매일 1/N 적립 · 토스 자동 모으기 수수료 0원) ===")
    print(f"{'체육관':<22} {'DCA수익':>8} {'DCA MDD':>8} {'DCA샤프':>7}"
          f" {'B&H수익':>8} {'B&H MDD':>8}")
    for lg in loaded:
        d = dca[lg.gym.name]
        print(f"{lg.gym.name:<22} {d.total_return:>+7.1%} {d.max_drawdown:>8.1%}"
              f" {d.sharpe:>7.2f} {d.market_return:>+7.1%} {d.market_drawdown:>8.1%}")

    # 저축왕(연 3% 무위험, 낙폭 0) — 비교 표시 전용 (적합도 줄세우기엔 미참가)
    savings = " · ".join(f"{lg.gym.name.split()[0]} {fight_savings(lg).total_return:+.1%}"
                         for lg in loaded)
    print(f"\n  [저축왕] 같은 기간 은행 복리(연 3%): {savings}")

    # ── 2. 전수조사: score_vs_dca 줄세우기 ──
    entries = [Strategy(genes=list(c), name="+".join(c))
               for k in range(1, len(ALL_GENES) + 1)
               for c in combinations(ALL_GENES, k)]
    entries += [Strategy(genes=["CASH"], name="[기준선] 전부 현금"),
                Strategy(genes=["FULL"], name="[기준선] 항상 풀매수")]

    rows = []
    for s in entries:
        report = challenge(s, loaded)
        scores = [score_vs_dca(r, dca[r.gym_name]) for r in report.results]
        rows.append((s.name, sum(scores) / len(scores), min(scores), scores))
    rows.sort(key=lambda r: -r[1])
    rank_of = {name: i + 1 for i, (name, *_rest) in enumerate(rows)}

    print(f"\n=== 2. score_vs_dca 순위 (총 {len(rows)}마리, ×100 표기) ===")
    print(f"{'순위':>3} {'평균':>7} {'최약':>7}  조합")
    for i, (name, avg, worst, _s) in enumerate(rows[:10], start=1):
        print(f"{i:>3} {avg * 100:>+7.1f} {worst * 100:>+7.1f}  {name}")
    for name in ("[기준선] 전부 현금", "[기준선] 항상 풀매수", "REV_BB"):
        i = rank_of[name] - 1
        _n, avg, worst, _s = rows[i]
        print(f"{i + 1:>3} {avg * 100:>+7.1f} {worst * 100:>+7.1f}  {name}")

    # DCA를 모든 체육관에서 이기는(전 점수 양수) 조합 수 — Pareto 필터 미리보기
    all_positive = [name for name, _a, worst, _s in rows if worst > 0]
    print(f"\n  전 체육관에서 DCA보다 나은 조합: {len(all_positive)}개"
          f"{' — ' + ', '.join(all_positive[:5]) if all_positive else ''}")

    # ── 3. 현 챔피언(VOL+REV_RSI+REV_BB) 체육관별 상세 ──
    print("\n=== 3. 현 챔피언 VOL+REV_RSI+REV_BB의 체육관별 score_vs_dca (×100) ===")
    report = challenge(Strategy(genes=["VOL", "REV_RSI", "REV_BB"],
                                name="VOL+REV_RSI+REV_BB"), loaded)
    for r in report.results:
        d = dca[r.gym_name]
        s = score_vs_dca(r, d)
        print(f"  {r.gym_name:<22} {s * 100:>+7.1f}"
              f"  (수익 {r.total_return:>+7.1%} vs {d.total_return:>+7.1%}"
              f" · MDD {r.max_drawdown:>6.1%} vs {d.max_drawdown:>6.1%}"
              f" · 샤프 {r.sharpe:>5.2f} vs {d.sharpe:>5.2f})")


if __name__ == "__main__":
    main()
