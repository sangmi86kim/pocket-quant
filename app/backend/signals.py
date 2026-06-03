"""
signals.py - 유전자(시그널)를 '진짜 지표 로직'으로 구현하는 파일

이전엔 유전자가 그냥 점수 라벨이었다 (DD=+20점 식). 이제는 각 유전자가
가격 시계열을 받아 '그날 주식을 얼마나 들고 있을지' = 포지션(0~1) 을 만든다.
  포지션 1.0 = 풀매수,  0.0 = 전액 현금,  0.5 = 반반

[유전자 매핑 — 사용자가 정한 컨셉 그대로]
  MA  : 이평 크로스 (추세추종)   - 가격이 장기 이평 위면 탑승
  RSI : 과매도/과매수            - 과열(RSI 높음)이면 비중 축소
  BB  : 볼린저밴드 (평균회귀)    - 상단밴드 위(과열)면 현금
  DD  : 드로다운 스탑 (방어)     - 고점 대비 일정% 빠지면 현금화
  FX  : 변동성 디리스킹 (현금/헤지) - 변동성 튀면 절반 현금

[전략 = 유전자들의 평균 포지션]
  여러 유전자를 가지면 각자의 포지션을 평균낸다.
  예) MA가 1.0(탑승)인데 DD가 0.0(현금화)면 -> 0.5 (반반).
  방어 유전자(DD/FX)가 공격 유전자를 끌어내리는 '견제'가 자연스럽게 생긴다.
"""
import numpy as np
import pandas as pd


def signal_MA(prices: pd.Series, window: int = 200) -> pd.Series:
    """추세추종: 가격이 장기 이평(기본 200일) 위면 1, 아니면 0."""
    sma = prices.rolling(window, min_periods=1).mean()
    return (prices > sma).astype(float)


def signal_RSI(prices: pd.Series, n: int = 14, overbought: int = 70) -> pd.Series:
    """과매수 회피: RSI가 과열선(기본 70) 아래면 탑승(1), 과열이면 현금(0)."""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(n, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50)     # 계산 불가 구간은 중립 50
    return (rsi < overbought).astype(float)


def signal_BB(prices: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """평균회귀: 볼린저 상단밴드 위(과열)면 현금(0), 그 외엔 탑승(1)."""
    ma = prices.rolling(n, min_periods=1).mean()
    sd = prices.rolling(n, min_periods=1).std().fillna(0)
    upper = ma + k * sd
    return (prices <= upper).astype(float)


def signal_DD(prices: pd.Series, limit: float = 0.10) -> pd.Series:
    """드로다운 스탑: 최근 고점 대비 limit(기본 10%) 넘게 빠지면 현금화(0)."""
    peak = prices.cummax()
    drawdown = prices / peak - 1.0          # 0 이하의 값 (예: -0.15 = 고점대비 -15%)
    return (drawdown > -limit).astype(float)


def signal_FX(prices: pd.Series, n: int = 20, daily_vol_limit: float = 0.015) -> pd.Series:
    """변동성 디리스킹: 최근 변동성이 한계 넘으면 절반 현금(0.5), 평온하면 풀(1.0)."""
    vol = prices.pct_change().rolling(n, min_periods=1).std().fillna(0)
    pos = pd.Series(1.0, index=prices.index)
    pos[vol > daily_vol_limit] = 0.5
    return pos


# 유전자 이름 -> 시그널 함수 (models.ALL_GENES 와 키가 일치해야 함)
GENE_SIGNALS = {
    "MA": signal_MA,
    "RSI": signal_RSI,
    "BB": signal_BB,
    "DD": signal_DD,
    "FX": signal_FX,
}


def combined_position(genes: list[str], prices: pd.Series) -> pd.Series:
    """전략의 유전자들이 만드는 포지션을 평균내 최종 일별 포지션(0~1)을 만든다."""
    if not genes:                                  # 유전자 없으면 풀매수로 간주
        return pd.Series(1.0, index=prices.index)
    positions = [GENE_SIGNALS[g](prices) for g in genes]
    return (sum(positions) / len(positions)).clip(0.0, 1.0)
