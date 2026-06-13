"""
test_no_lookahead.py - 룩어헤드(미래 훔쳐보기) 불변식 테스트

[불변식]
"오늘의 포지션은 미래 데이터를 지워도 변하지 않아야 한다."
백테스트에서 가장 흔하고 치명적인 버그 종류 — 시그널이 미래를 한 칸이라도
보면(centered rolling, 전체 기간 정규화, 미래 shift 등) 백테스트 성적이
실전에서 재현 불가능한 허수가 된다. 지금까지는 코드 리뷰로만 막았는데,
NSGA-III가 시그널 파라미터를 마구 바꾸기 시작하면 자동 검증이 필요하다.

[방법]
전체 가격으로 만든 포지션과, 특정 날짜 T에서 잘라낸 가격으로 만든 포지션이
T까지 완전히 동일한지 비교한다. 전 유전자 + 결합(combined_position) 모두.
잘라내는 날짜는 성격이 다른 3곳: 위기 한복판 / V자 바닥 / 평시.

실행: 프로젝트 루트에서  python tests/test_no_lookahead.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.backend.genes.signals import ALL_GENES, GENE_SIGNALS, combined_position
from app.backend.data_io.data import get_prices

# 잘라낼 날짜들 — 위기 한복판(리먼) / 코로나 V자 바닥 부근 / 평범한 평시
CUT_DATES = ["2008-09-15", "2020-03-20", "2015-06-30"]


def _same_until(full, truncated) -> bool:
    """전체 계산의 앞부분과 잘린 계산이 (NaN 포함) 완전히 같은가."""
    head = full.loc[truncated.index]
    if not head.index.equals(truncated.index):
        return False
    return np.array_equal(head.to_numpy(), truncated.to_numpy(), equal_nan=True)


def run_check() -> bool:
    # 캐시된 SPY 전체 역사 사용 (오프라인 OK)
    prices = get_prices("SPY", "1994-01-01", "2026-06-09")
    failures: list[str] = []

    print("=== 룩어헤드 검사: 미래를 잘라도 과거 포지션 불변 ===\n")
    for cut in CUT_DATES:
        past = prices.loc[:cut]
        print(f"[자르는 날짜 {cut}] (과거 {len(past)}일)")

        # 유전자 하나씩
        for gene in ALL_GENES:
            ok = _same_until(GENE_SIGNALS[gene](prices), GENE_SIGNALS[gene](past))
            print(f"  [{'PASS' if ok else 'FAIL'}] {gene}")
            if not ok:
                failures.append(f"{gene}@{cut}")

        # 전 유전자 결합 (기권 제외 평균까지 통째로)
        ok = _same_until(combined_position(ALL_GENES, prices),
                         combined_position(ALL_GENES, past))
        print(f"  [{'PASS' if ok else 'FAIL'}] 결합({'+'.join(ALL_GENES)})\n")
        if not ok:
            failures.append(f"combined@{cut}")

    print(f"=== 판정: {'PASS' if not failures else 'FAIL ' + str(failures)} ===")
    return not failures


# pytest로 돌릴 때도 같은 검증을 쓴다
def test_no_lookahead():
    assert run_check(), "룩어헤드 검출: 시그널이 미래 데이터를 사용함"


if __name__ == "__main__":
    sys.exit(0 if run_check() else 1)
