"""
test_baselines.py - 적합도 퇴화 검증 ("아무것도 안 하기"가 최적이 되면 안 된다)

[배경]
예전 설계(HP=현금비중 가중 1.0, DEF=시장 대비 낙폭비, ATK=-25%~+25% 스케일)에서는
'전부 현금'(포지션 상수 0)이 HP 100 + DEF 100을 공짜로 받아 적합도 ~69점으로
거의 모든 진짜 전략을 이겼다. 적합도 재설계 후 이 퇴화가 사라졌는지 잰다.

[방법]
1) 기준선 2마리를 임시 유전자로 등록 (GA 풀 ALL_GENES에는 안 넣음)
     CASH = 항상 포지션 0 (전부 현금)
     FULL = 항상 포지션 1 (항상 풀매수)
2) 진짜 유전자 6개의 모든 조합(63개) + 기준선 2개 = 65마리를 전 체육관에 도전
3) 합격 조건:
     - 전부 현금(CASH) 적합도 < 항상 풀매수(FULL) 적합도
     - 전부 현금이 전체 순위 하위 25% 밖으로 못 올라옴

실행: 프로젝트 루트에서
    python tests/test_baselines.py
"""
import sys
from itertools import combinations
from pathlib import Path

# 프로젝트 루트에서 실행 안 해도 import가 되도록 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app.academy.exam import all_gyms
from app.pocket import signals
from app.pocket.battle import challenge
from app.pocket.models import Strategy
from app.pocket.signals import ALL_GENES
from app.world.data import load_gyms


def _register_baselines() -> None:
    """기준선 유전자를 시그널 레지스트리에만 등록한다 (ALL_GENES는 건드리지 않음 = GA 무관)."""
    signals.GENE_SIGNALS["CASH"] = lambda p: pd.Series(0.0, index=p.index)
    signals.GENE_SIGNALS["FULL"] = lambda p: pd.Series(1.0, index=p.index)


def run_check() -> bool:
    _register_baselines()
    loaded_gyms = load_gyms(all_gyms())

    # 진짜 유전자 전 조합(63) + 기준선 2
    entries: list[Strategy] = [
        Strategy(genes=list(combo), name="+".join(combo))
        for size in range(1, len(ALL_GENES) + 1)
        for combo in combinations(ALL_GENES, size)
    ]
    cash = Strategy(genes=["CASH"], name="[기준선] 전부 현금")
    full = Strategy(genes=["FULL"], name="[기준선] 항상 풀매수")
    entries += [cash, full]

    scored = [(s, challenge(s, loaded_gyms).fitness) for s in entries]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    rank_of = {s.name: i + 1 for i, (s, _f) in enumerate(scored)}
    fit_of = {s.name: f for s, f in scored}
    total = len(scored)

    print(f"=== 적합도 순위 (전 조합 {total - 2} + 기준선 2 = 트레이더 {total}명) ===\n")
    print("상위 10:")
    for i, (s, f) in enumerate(scored[:10], start=1):
        print(f"  {i:2}위  {f:5.1f}점  {s.name}")
    print(f"\n기준선:")
    for s in (cash, full):
        print(f"  {rank_of[s.name]:2}위  {fit_of[s.name]:5.1f}점  {s.name}")

    # 합격 조건
    cash_rank, cash_fit = rank_of[cash.name], fit_of[cash.name]
    full_fit = fit_of[full.name]
    beats_full = cash_fit < full_fit
    in_bottom = cash_rank > total * 0.75      # 하위 25% 안에 있어야 함

    print("\n=== 판정 ===")
    print(f"  전부 현금 < 항상 풀매수      : {'PASS' if beats_full else 'FAIL'} "
          f"({cash_fit:.1f} vs {full_fit:.1f})")
    print(f"  전부 현금이 하위 25%에 위치  : {'PASS' if in_bottom else 'FAIL'} "
          f"({cash_rank}/{total}위)")
    return beats_full and in_bottom


# pytest로 돌릴 때도 같은 검증을 쓴다
def test_all_cash_is_not_optimal():
    assert run_check(), "퇴화 검출: '전부 현금' 전략이 적합도 상위권에 있음"


if __name__ == "__main__":
    ok = run_check()
    sys.exit(0 if ok else 1)
