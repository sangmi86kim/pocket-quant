"""
signals.py - 유전자(시그널)를 '진짜 지표 로직'으로 구현하는 파일

이전엔 유전자가 그냥 점수 라벨이었다 (DD=+20점 식). 이제는 각 유전자가
가격 시계열을 받아 '그날 주식을 얼마나 들고 있을지' = 포지션(0~1) 을 만든다.
  포지션 1.0 = 풀매수,  0.0 = 전액 현금,  0.5 = 반반

[유전자 매핑 — 사용자가 정한 컨셉]
  DD  : 리스크      - 드로다운 스탑. 고점 대비 일정% 빠지면 현금화
  RSI : 심리        - 과매수/과매도. 과열(군중 심리 과열)이면 비중 축소
  MA  : 추세        - 이평. 가격이 장기 이평 위면 탑승
  BB  : 변동성      - 볼린저밴드. 상단밴드 위(과열)면 현금
  VOL : 시장 상태   - 실현변동성 레짐. 평온=탑승 / 중간=반반 / 격동=현금
  MOM : 추세 강화   - 모멘텀. 최근 수익률이 양수면 추세에 더 올라탐

[전략 = 유전자들의 평균 포지션]
  여러 유전자를 가지면 각자의 포지션을 평균낸다.
  예) MA가 1.0(탑승)인데 DD가 0.0(현금화)면 -> 0.5 (반반).
  방어 유전자(DD/VOL)가 공격 유전자(MA/MOM)를 끌어내리는 '견제'가 자연스럽게 생긴다.
"""
import numpy as np
import pandas as pd

# ── 시그널 튜닝 파라미터 (한곳에 모음 = 진화/실험 시 여기만 만지면 됨) ──
MA_WINDOW = 200                       # MA: 장기 이평 기간(일)
RSI_PERIOD, RSI_OVERBOUGHT = 14, 70   # RSI: 기간 / 과열선
BB_PERIOD, BB_K = 20, 2.0             # BB: 기간 / 표준편차 배수
DD_LIMIT = 0.10                       # DD: 고점 대비 허용 낙폭(넘으면 현금화)
VOL_PERIOD = 20                       # VOL: 실현변동성 측정 기간
VOL_CALM, VOL_STRESSED = 0.010, 0.020 # VOL: 평온/격동 일변동성 임계
MOM_LOOKBACK = 63                     # MOM: 모멘텀 측정 기간(약 3개월)


def signal_MA(prices: pd.Series, window: int = MA_WINDOW) -> pd.Series:
    """추세추종: 가격이 장기 이평(기본 200일) 위면 1, 아니면 0."""
    sma = prices.rolling(window, min_periods=1).mean()
    return (prices > sma).astype(float)


def signal_RSI(prices: pd.Series, n: int = RSI_PERIOD,
               overbought: int = RSI_OVERBOUGHT) -> pd.Series:
    """과매수 회피: RSI가 과열선(기본 70) 아래면 탑승(1), 과열이면 현금(0)."""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(n, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50)     # 계산 불가 구간은 중립 50
    return (rsi < overbought).astype(float)


def signal_BB(prices: pd.Series, n: int = BB_PERIOD, k: float = BB_K) -> pd.Series:
    """평균회귀: 볼린저 상단밴드 위(과열)면 현금(0), 그 외엔 탑승(1)."""
    ma = prices.rolling(n, min_periods=1).mean()
    sd = prices.rolling(n, min_periods=1).std().fillna(0)
    upper = ma + k * sd
    return (prices <= upper).astype(float)


def signal_DD(prices: pd.Series, limit: float = DD_LIMIT) -> pd.Series:
    """드로다운 스탑: 최근 고점 대비 limit(기본 10%) 넘게 빠지면 현금화(0)."""
    peak = prices.cummax()
    drawdown = prices / peak - 1.0          # 0 이하의 값 (예: -0.15 = 고점대비 -15%)
    return (drawdown > -limit).astype(float)


def signal_VOL(prices: pd.Series, n: int = VOL_PERIOD,
               calm: float = VOL_CALM, stressed: float = VOL_STRESSED) -> pd.Series:
    """시장 상태(변동성 레짐): 평온하면 탑승(1.0) / 중간이면 반반(0.5) / 격동이면 현금(0.0)."""
    vol = prices.pct_change().rolling(n, min_periods=1).std().fillna(0)
    pos = pd.Series(0.5, index=prices.index)        # 기본 = 중간 상태
    pos[vol <= calm] = 1.0                            # 평온장 = 위험선호
    pos[vol > stressed] = 0.0                         # 격동장 = 위험회피
    return pos


def signal_MOM(prices: pd.Series, lookback: int = MOM_LOOKBACK) -> pd.Series:
    """추세 강화(모멘텀): 최근 lookback(약 3개월) 수익률이 양수면 탑승(1), 아니면 이탈(0)."""
    momentum = prices / prices.shift(lookback) - 1.0
    return (momentum > 0).astype(float)              # 초기(데이터 부족) 구간은 0


# 유전자 이름 -> 시그널 함수.
# 이 레지스트리가 '어떤 유전자가 존재하는가'의 단일 출처(source of truth)다.
# (예전엔 models.GENE_SCORES 라는 가짜 점수표가 명단을 정의했지만, 이제 실제
#  시그널을 가진 유전자만이 진짜 유전자다.)
GENE_SIGNALS = {
    "DD": signal_DD,
    "RSI": signal_RSI,
    "MA": signal_MA,
    "BB": signal_BB,
    "VOL": signal_VOL,
    "MOM": signal_MOM,
}

# 사용 가능한 모든 유전자 이름 -> ["DD", "RSI", "MA", "BB", "VOL", "MOM"]
ALL_GENES = list(GENE_SIGNALS.keys())

# 참고: 유전자 설명 카드(포켓몬 도감)는 dex.py(SIGNAL_CARDS)에 있다.


def combined_position(genes: list[str], prices: pd.Series) -> pd.Series:
    """전략의 유전자들이 만드는 포지션을 평균내 최종 일별 포지션(0~1)을 만든다."""
    if not genes:                                  # 유전자 없으면 풀매수로 간주
        return pd.Series(1.0, index=prices.index)
    positions = [GENE_SIGNALS[g](prices) for g in genes]
    return (sum(positions) / len(positions)).clip(0.0, 1.0)
