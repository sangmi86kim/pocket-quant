"""
test_weighted_combine.py - 가중 결합(combine_positions weights)의 불변식 검증

NSGA-III의 결정변수인 가중 결합이 기존 결합 규칙을 깨지 않는지 잰다:
  ① 동등성   : 동일 가중치(전부 1) == weights=None(기존 동일가중 평균)
  ② 배제성   : 가중치 0인 시그널 == 그 시그널을 아예 빼고 결합한 것
  ③ 스케일 불변: weights를 통째로 k배 해도 결과 동일 (비율만 의미 = 예산제약 내장)
  ④ 기권 보존 : 전원 기권한 날은 가중치와 무관하게 0.0

실행: 프로젝트 루트에서  python tests/test_weighted_combine.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.backend.genes.signals import (ALL_GENES, GENE_SIGNALS,
                                       combine_positions, positions_with_params)
from app.backend.data_io.data import get_prices


def _same(a, b) -> bool:
    return a.index.equals(b.index) and np.array_equal(
        a.to_numpy(), b.to_numpy(), equal_nan=True)


def run_check() -> bool:
    prices = get_prices("SPY", "1994-01-01", "2026-06-09")  # 캐시 사용, 오프라인 OK
    positions = [GENE_SIGNALS[g](prices) for g in ALL_GENES]
    failures: list[str] = []

    def check(label: str, ok: bool):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            failures.append(label)

    print("=== 가중 결합 불변식 ===")

    n = len(ALL_GENES)        # 시그널 풀 크기 — 풀 확장돼도 자동 반영
    idx_vol = ALL_GENES.index("VOL")
    idx_rsi = ALL_GENES.index("REV_RSI")
    idx_bb = ALL_GENES.index("REV_BB")

    # ① 동일 가중치 == 기존 동일가중 평균 (골든 경로 보호)
    check(f"동등성: weights=[1]*{n} == weights=None",
          _same(combine_positions(positions, [1.0] * n),
                combine_positions(positions)))

    # ② 가중치 0 == 시그널 제외 — VOL+REV_RSI+REV_BB만 살림
    sub = [positions[idx_vol], positions[idx_rsi], positions[idx_bb]]
    w_zeroed = [1.0 if i in (idx_vol, idx_rsi, idx_bb) else 0.0 for i in range(n)]
    check("배제성: w=0 시그널 == 명단 제외",
          _same(combine_positions(positions, w_zeroed),
                combine_positions(sub)))

    # ③ 스케일 불변 (비율만 의미)
    rng = np.random.default_rng(42)
    w = rng.uniform(0.1, 1.0, size=n).tolist()
    a = combine_positions(positions, w)
    b = combine_positions(positions, [x * 7.3 for x in w])
    check("스케일 불변: w == 7.3·w", bool(np.allclose(a, b, atol=1e-12)))

    # ④ 전원 기권한 날 = 0.0 (REV만 가중치를 주면 평소가 전원 기권)
    w_rev = [1.0 if i in (idx_rsi, idx_bb) else 0.0 for i in range(n)]
    rev_only = combine_positions(positions, w_rev)
    rev_union_abstain = positions[idx_rsi].isna() & positions[idx_bb].isna()
    check("기권 보존: 전원 기권일은 0.0",
          bool((rev_only[rev_union_abstain] == 0.0).all()))

    # ⑤ 파라미터 주입 기본값 경로 == GENE_SIGNALS 경로
    check("파라미터 주입: params=None == 기본 시그널",
          all(_same(a, b) for a, b in
              zip(positions_with_params(prices), positions)))

    print(f"\n=== 판정: {'PASS' if not failures else 'FAIL ' + str(failures)} ===")
    return not failures


def test_weighted_combine():
    assert run_check(), "가중 결합 불변식 위반"


if __name__ == "__main__":
    sys.exit(0 if run_check() else 1)
