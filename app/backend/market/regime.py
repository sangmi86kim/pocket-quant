"""시장 국면 판독 — Regime_Scanner와 라벨 정의 동기 (단일 소스).

[출처] C:\\HomeLab\\my_project\\quant\\Regime_Scanner\\backend\\(signals.py + config.py)
[배경] 사용자 안 (06-13): PocketQuant 시험장(시험장·평행세계·사천왕)의 각
구간마다 우세 국면을 라벨링해 reports/regime_picks.json에 저장 → 추후
Regime_Scanner가 같은 정의로 추론하면 학습 데이터(여기서 만든 picks)와
추론 입력이 호환된다.

[판정 로직] (Regime_Scanner와 동일)
  1) 추세 점수 = 종가 vs MA50/MA200 + MA50 vs MA200 + 60일 수익률 부호(±3% 밴드).
  2) 점수 ≥ +2 → bull,  ≤ -2 → bear.
  3) 모호(-1~+1)이고 20일 실현변동성 백분위 ≥ 0.85 → volatile, 아니면 sideways.
  4) 변동성 백분위는 그 시점까지의 과거 분포 대비 (룩어헤드 방지).

⚠️ Regime_Scanner config 변경(파라미터·티커)이 PocketQuant에 자동 반영되지
않는다. 한쪽 손대면 다른 쪽도 같이 — 양쪽 동기화 책임은 사람.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252
VOL_WINDOW = 20
VOL_PCT_THRESHOLD = 0.85
RET_WINDOW = 60
RET_BAND = 0.03
TREND_BULL = 2
TREND_BEAR = -2

REGIME_LABELS = {"bull": "상승장", "bear": "하락장",
                 "sideways": "횡보장", "volatile": "변동장"}


def classify_daily(prices: pd.Series) -> pd.Series:
    """가격 시계열 → 일별 국면 라벨 (bull/bear/sideways/volatile).

    워밍업: MA200 + 60일 수익률 + 252일 변동성 백분위 → 약 252일 필요.
    이 일수가 안 채워진 시점은 결과에서 자동 dropna.
    """
    ma50 = prices.rolling(50).mean()
    ma200 = prices.rolling(200).mean()
    ret60 = prices.pct_change(RET_WINDOW)
    vol20 = prices.pct_change().rolling(VOL_WINDOW).std() * np.sqrt(TRADING_DAYS)
    vol_pct = vol20.expanding(min_periods=TRADING_DAYS).rank(pct=True)

    df = pd.DataFrame({"close": prices, "ma50": ma50, "ma200": ma200,
                       "ret60": ret60, "vol_pct": vol_pct}).dropna(
                           subset=["ma200", "ret60", "vol_pct"])

    score = (np.where(df["close"] > df["ma50"], 1, -1)
             + np.where(df["close"] > df["ma200"], 1, -1)
             + np.where(df["ma50"] > df["ma200"], 1, -1)
             + np.where(df["ret60"] > RET_BAND, 1,
                        np.where(df["ret60"] < -RET_BAND, -1, 0)))

    label = pd.Series("sideways", index=df.index)
    label[score >= TREND_BULL] = "bull"
    label[score <= TREND_BEAR] = "bear"
    label[(score > TREND_BEAR) & (score < TREND_BULL)
          & (df["vol_pct"] >= VOL_PCT_THRESHOLD)] = "volatile"
    return label


def dominant_regime(prices: pd.Series, start: str, end: str) -> str:
    """가격 시계열의 [start, end] 구간 우세 라벨 (가장 많은 일수의 라벨).
    빈 구간이면 sideways 폴백 — 호출 측에서 일어나서는 안 될 경로지만 안전."""
    daily = classify_daily(prices)
    mask = (daily.index >= pd.Timestamp(start)) & (daily.index <= pd.Timestamp(end))
    sub = daily[mask]
    if len(sub) == 0:
        return "sideways"
    return str(sub.value_counts().idxmax())
