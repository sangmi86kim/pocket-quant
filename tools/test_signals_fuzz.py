"""
test_signals_fuzz.py - 시그널 함수 속성 기반 퍼징 (Hypothesis)

[배경]
기존 tools/test_*.py 는 'SPY 실제 데이터 + 고정 날짜' 로 불변식을 검사한다.
이 테스트는 한 발 더 가서, Hypothesis 가 **일부러 못된 가격 시계열을 수백 개
생성**해 같은 종류의 불변식을 두들긴다. 깨지는 입력을 찾으면 Hypothesis 가
자동으로 최소 반례까지 줄여준다(예: "길이 2, 둘째 값이 0인 시리즈에서 터짐").

순수 랜덤 '퍼징' 대신 속성 기반 테스트(property-based)를 쓰는 이유: 검사하는
것이 '터지나'(크래시) 뿐 아니라 '약속을 지키나'(불변식)이기 때문이다.

[검사하는 불변식 — 깨지면 안 되는 약속]
  1) 최종 포지션 combined_position 은 항상 [0,1] 범위, NaN·inf 없음.
  2) 가격 기반 시그널 각각의 출력값은 기권(NaN)이거나 [0,1] (inf 없음).
  3) 가중결합 combine_positions 도 임의 가중치에서 [0,1]·유한.
  4) 어떤 입력에도 예외로 죽지 않는다 (죽으면 그게 곧 버그 리포트).
  5) safe_ratio(공용 비율 가드): 분모가 0이어도 결과는 항상 finite 이거나 NaN —
     ±inf를 만들지 않는다(불변식 1·2는 출력이 0/1/NaN이라 'inf가 발동으로 둔갑'을
     못 잡는다. 이 규약은 비율 단계 자체를 직접 두들겨야 보인다).

[범위 = signals.py 전체 14마리]
  - 가격 기반 6마리(DD/VOL/MA/MOM/REV_RSI/REV_BB): 합성 양(+)가격으로 두들긴다.
  - 야생 8마리(VOL_SPIKE/FEAR/FEAR_NQ/US10Y/DXY/SPY_TLT/QQQ_SPY/QQQ_DIA): 외부
    시계열(yfinance) 대신 attrs["external_streams"]에 0(→0-나누기)·NaN(→구멍)·극단값을
    섞은 '더러운 스트림' 9종을 주입해 두들긴다. synthetic 표시는 안전망(빠진 티커는
    네트워크 대신 기권). 전부 오프라인.

[새 야생 시그널을 잡으면(절대규칙 #4)] 그 시그널을 아래 EXTERNAL·EXTERNAL_TICKERS에
등록하고 이 퍼징을 통과시켜야 한다 — "금융 시그널은 다 더럽다"가 이 파일의 전제다.

실행: 프로젝트 루트에서  .venv/Scripts/python.exe tools/test_signals_fuzz.py
       (또는  .venv/Scripts/python.exe -m pytest tools/test_signals_fuzz.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, example, settings, strategies as st

from app.pocket.ratio_guard import safe_ratio
from app.pocket.signals import (
    SIGNAL_NAMES,
    SIGNAL_REGISTRY,
    combine_positions,
    combined_position,
    positions_with_params,
)

# 가격만으로 도는 시그널 — 합성 가격으로 직접 검사.
PRICE_ONLY = ["DD", "VOL", "MA", "MOM", "REV_RSI", "REV_BB"]

# 외부 정보원 시그널(야생 8마리) — 더러운 외부 스트림 주입으로 검사.
EXTERNAL = ["VOL_SPIKE", "FEAR", "FEAR_NQ", "US10Y", "DXY",
            "SPY_TLT", "QQQ_SPY", "QQQ_DIA"]

# 야생 시그널이 읽는 외부 티커 (signals._fetch_external / signal_VOL_SPIKE 참조).
# 새 야생 시그널이 새 티커를 읽으면 여기에 추가한다.
EXTERNAL_TICKERS = ["^VIX", "^VXN", "^TNX", "UUP", "SPY", "TLT",
                    "QQQ", "DIA", "QQQ_volume"]


@st.composite
def price_series(draw, min_len: int = 1, max_len: int = 400):
    """못된 양(+)의 가격 시계열을 생성한다.

    가격은 실자산이라 도메인상 양수다(음수·0 가격은 의미 없음). 대신 1e-3 같은
    극소값~1e6 극대값, 전부 같은 값(변동성 0), 길이 1까지 섞어 약한 지점을 노린다.
    synthetic 표시로 외부 시그널은 기권 → 오프라인.
    """
    n = draw(st.integers(min_value=min_len, max_value=max_len))
    vals = draw(st.lists(
        st.floats(min_value=1e-3, max_value=1e6,
                  allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n,
    ))
    idx = pd.bdate_range("2000-01-01", periods=n)
    s = pd.Series(vals, index=idx, name="QQQ", dtype=float)
    s.attrs["synthetic"] = True   # 외부 시그널 기권 → 네트워크 없음
    return s


@st.composite
def price_with_external(draw, min_len: int = 1, max_len: int = 120):
    """가격 + '더러운' 외부 스트림 9종을 attrs에 주입한 시계열.

    외부 스트림 값에 0(분모면 0-나누기)·NaN(구멍)·극단값을 섞는다. 야생 시그널이
    이런 더러운 입력에도 [0,1]·기권(NaN)만 내고 예외로 죽지 않는지 두들기기 위함.
    synthetic 표시는 안전망 — 혹시 빠진 티커가 있어도 네트워크 대신 기권된다.
    """
    n = draw(st.integers(min_value=min_len, max_value=max_len))
    idx = pd.bdate_range("2000-01-01", periods=n)
    pvals = draw(st.lists(
        st.floats(min_value=1e-3, max_value=1e6,
                  allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n,
    ))
    prices = pd.Series(pvals, index=idx, name="QQQ", dtype=float)

    dirty = st.one_of(                                     # 0·극단값 + NaN 구멍 = 더러움
        st.floats(min_value=0.0, max_value=1e6,
                  allow_nan=False, allow_infinity=False),
        st.just(float("nan")),
    )
    streams = {
        tk: pd.Series(draw(st.lists(dirty, min_size=n, max_size=n)),
                      index=idx, dtype=float)
        for tk in EXTERNAL_TICKERS
    }
    prices.attrs["synthetic"] = True              # 빠진 티커 안전망(기권)
    prices.attrs["external_streams"] = streams
    return prices


def _finite_unit(arr: np.ndarray) -> bool:
    """모든 값이 유한하고 [0,1] 안인가 (NaN·inf 없음)."""
    return bool(np.all(np.isfinite(arr)) and np.all(arr >= 0.0) and np.all(arr <= 1.0))


@settings(max_examples=100, deadline=None)
@example(prices=pd.Series([100.0], index=pd.bdate_range("2000-01-01", periods=1),
                          name="QQQ", dtype=float))          # 길이 1
@example(prices=pd.Series([5.0] * 300, index=pd.bdate_range("2000-01-01", periods=300),
                          name="QQQ", dtype=float))          # 완전 평탄(변동성 0)
@given(prices=price_series())
def test_combined_in_range(prices):
    """불변식 1: 전 유전자 결합 포지션은 항상 [0,1]·유한, 인덱스 보존."""
    if "synthetic" not in prices.attrs:
        prices.attrs["synthetic"] = True
    pos = combined_position(SIGNAL_NAMES, prices)
    assert pos.index.equals(prices.index), "결합 포지션이 입력 인덱스를 바꿈"
    assert _finite_unit(pos.to_numpy()), "결합 포지션이 [0,1]·유한 위반"


@settings(max_examples=100, deadline=None)
@given(prices=price_series())
def test_each_price_signal_well_formed(prices):
    """불변식 2·4: 가격 기반 시그널 각각 — 기권(NaN) 아닌 값은 [0,1]·유한, 무예외."""
    for name in PRICE_ONLY:
        out = SIGNAL_REGISTRY[name](prices)
        assert out.index.equals(prices.index), f"{name}: 인덱스 변형"
        voted = out.dropna().to_numpy()          # 기권(NaN) 제외한 실제 의견
        assert _finite_unit(voted), f"{name}: 의견값이 [0,1]·유한 위반"


@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.data_too_large])  # 스트림 9종 = 큰 입력
@given(prices=price_with_external())
def test_each_external_signal_well_formed(prices):
    """불변식 2·4 (야생 8마리): 더러운 외부 스트림(0·NaN·극단값)에도 기권 아닌
    의견값은 [0,1]·유한이고, 0-나누기에도 예외로 죽지 않는다."""
    for name in EXTERNAL:
        out = SIGNAL_REGISTRY[name](prices)
        assert out.index.equals(prices.index), f"{name}: 인덱스 변형"
        voted = out.dropna().to_numpy()          # 기권(NaN) 제외한 실제 의견
        assert _finite_unit(voted), f"{name}: 의견값이 [0,1]·유한 위반"


@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.data_too_large])  # 스트림 9종 = 큰 입력
@given(prices=price_with_external())
def test_combined_with_external_streams(prices):
    """불변식 1 (전원): 14마리 전부(가격 6 + 야생 8, 더러운 외부 스트림 포함)
    결합해도 [0,1]·유한. 외부 시그널이 실제로 발동하는 경로까지 통째로 두들긴다."""
    pos = combined_position(SIGNAL_NAMES, prices)
    assert pos.index.equals(prices.index), "결합 포지션이 입력 인덱스를 바꿈"
    assert _finite_unit(pos.to_numpy()), "외부 포함 결합이 [0,1]·유한 위반"


@settings(max_examples=100, deadline=None)
@given(prices=price_series(), data=st.data())
def test_weighted_combine_in_range(prices, data):
    """불변식 3: 임의 비음(非負) 가중치로 결합해도 [0,1]·유한 (NSGA 결정변수 퍼징)."""
    positions = positions_with_params(prices)    # synthetic → 외부 7마리 기권
    weights = data.draw(st.lists(
        st.floats(min_value=0.0, max_value=1e3,
                  allow_nan=False, allow_infinity=False),
        min_size=len(positions), max_size=len(positions),
    ))
    pos = combine_positions(positions, weights)
    assert _finite_unit(pos.to_numpy()), "가중결합이 [0,1]·유한 위반"


@settings(max_examples=200, deadline=None)
@given(data=st.data())
def test_safe_ratio_never_inf(data):
    """불변식 5: 유한한 분자 / (0·극단값 섞인) 분모 — 결과는 finite or NaN, inf 없음."""
    n = data.draw(st.integers(min_value=1, max_value=200))
    idx = pd.bdate_range("2000-01-01", periods=n)
    finite = st.floats(min_value=-1e6, max_value=1e6,
                       allow_nan=False, allow_infinity=False)
    dirty_den = st.one_of(finite, st.just(0.0), st.just(float("nan")))  # 0 = 정의 불가
    num = pd.Series(data.draw(st.lists(finite, min_size=n, max_size=n)),
                    index=idx, dtype=float)
    den = pd.Series(data.draw(st.lists(dirty_den, min_size=n, max_size=n)),
                    index=idx, dtype=float)
    out = safe_ratio(num, den).to_numpy()
    assert not np.isinf(out).any(), "safe_ratio가 ±inf를 흘렸다 (분모 0 → 기권 위반)"


def run_check() -> bool:
    """pytest 없이 스크립트로 돌릴 때의 진입점. 각 속성을 직접 호출한다."""
    checks = [
        ("결합 포지션 [0,1] (가격경로)", test_combined_in_range),
        ("가격 시그널 well-formed", test_each_price_signal_well_formed),
        ("야생 시그널 well-formed (더러운 외부)", test_each_external_signal_well_formed),
        ("결합 [0,1] (외부 스트림 포함)", test_combined_with_external_streams),
        ("가중결합 [0,1]", test_weighted_combine_in_range),
        ("safe_ratio finite-or-NaN (분모 0)", test_safe_ratio_never_inf),
    ]
    print("=== signals.py 속성 기반 퍼징 (Hypothesis) ===\n")
    failures: list[str] = []
    for label, fn in checks:
        try:
            fn()
            print(f"  [PASS] {label}")
        except AssertionError as e:
            print(f"  [FAIL] {label}\n         {e}")
            failures.append(label)
    print(f"\n=== 판정: {'PASS' if not failures else 'FAIL ' + str(failures)} ===")
    return not failures


if __name__ == "__main__":
    sys.exit(0 if run_check() else 1)
