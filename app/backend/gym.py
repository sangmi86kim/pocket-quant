"""
gym.py - 체육관(시장 국면) 데이터를 정의하는 파일

[v0.3] 이제 가짜 데이터가 아니다. 각 체육관은 실제 역사적 기간을 가리키고,
battle.py가 그 기간의 SPY(S&P500 ETF) 가격을 받아 진짜 백테스트를 돌린다.
기간은 '직전 고점 ~ 회복/안정'까지 잡아 폭락과 그 여파를 함께 본다.

[difficulty / volatility]
  이제 판정에는 안 쓰는 '연출용 메타데이터'다(시장 설명·정렬 참고용).
  실제 난이도는 그 기간 가격이 직접 만든다.

[기간 근거 — 실제 역사]
  닷컴(2000~02)   : S&P -49%(나스닥 -78%), 길고 느린 약세장
  금융위기(2008)  : S&P -57%, 시스템 붕괴, 최대 낙폭
  코로나(2020)    : S&P -34%지만 V자 즉시 회복 → 버티면 생존
  금리쇼크(2022)  : S&P -25%, 채권 동반 하락, 질서있는 하락
"""
from .models import Gym

# SPY는 1993년 상장이라 아래 4개 국면을 모두 커버한다.
GYMS = [
    # 최대 낙폭·시스템 붕괴·장기 → 최난도
    Gym(name="FINANCIAL_CRISIS", difficulty=90, volatility=80,
        ticker="SPY", start="2007-10-01", end="2009-06-30"),
    # 2.5년 장기 약세장, 패닉 스파이크 없는 '느린' 하락
    Gym(name="DOTCOM", difficulty=85, volatility=55,
        ticker="SPY", start="2000-01-01", end="2002-12-31"),
    # 중간 낙폭, 채권 헤지도 실패했으나 비교적 질서있는 하락
    Gym(name="RATE_SHOCK", difficulty=60, volatility=40,
        ticker="SPY", start="2022-01-01", end="2022-12-31"),
    # V자 즉시 회복이라 버티면 살아남음, 단 변동성은 역대 최고(VIX 82.7)
    Gym(name="COVID", difficulty=40, volatility=95,
        ticker="SPY", start="2020-01-01", end="2020-08-31"),
]


def all_gyms() -> list[Gym]:
    """전체 체육관 목록의 '복사본'을 돌려준다(원본 GYMS 보호용)."""
    return list(GYMS)
