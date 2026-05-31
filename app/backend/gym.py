"""
gym.py - 체육관(맵) 데이터를 정의하는 파일

실제 주가 데이터는 쓰지 않습니다(MVP라서 '가짜 데이터').
각 체육관은 이름 / 난이도(difficulty) / 변동성(volatility) 세 값만 갖습니다.

[값의 의미 — 두 축은 서로 다른 걸 측정한다]
  difficulty(생존 난이도) : 하락의 깊이 × 기간 × 회피 불가능성. 높을수록 통과(생존) 어려움.
  volatility(변동성)      : 그 시기의 실현 변동성 / VIX 피크. 얼마나 출렁였나.
  ※ 둘은 비례하지 않는다. (예: 코로나는 변동성 최고지만 V자 회복이라 생존은 쉬움)

[아래 값은 임의값이 아니라 실제 역사 데이터에 근거]
  닷컴(2000~02)   : S&P -49%(나스닥 -78%), ~2.5년, VIX 피크 ~45  → 길고 깊지만 '느린' 약세장
  금융위기(2008)  : S&P -57%, ~1.5년, VIX 피크 ~80             → 최대 낙폭·시스템 붕괴, 최난도
  코로나(2020)    : S&P -34%, ~1개월, VIX 82.7(역대 최고)       → 폭락했으나 즉시 V자 회복 → 생존 쉬움
  금리쇼크(2022)  : S&P -25%, ~1년, VIX 피크 ~36               → 채권도 동반 하락, 질서있는 하락
"""
from .models import Gym   # 같은 폴더 models.py의 Gym 설계도를 가져옴

# 미리 만들어 둔 체육관 4곳. (difficulty가 높을수록 통과하기 어렵다)
GYMS = [
    # 최대 낙폭·시스템 붕괴·장기 → 생존 난이도 최고, 변동성도 매우 큼(VIX~80)
    Gym(name="FINANCIAL_CRISIS", difficulty=90, volatility=80),
    # 2.5년 장기 약세장이라 생존은 어렵지만, 패닉 스파이크 없는 '느린' 하락(VIX~45)
    Gym(name="DOTCOM", difficulty=85, volatility=55),
    # 중간 낙폭, 헤지(채권)도 실패했으나 비교적 질서있는 하락(VIX~36)
    Gym(name="RATE_SHOCK", difficulty=60, volatility=40),
    # V자 즉시 회복이라 버티면 살아남음 → 난이도 낮음. 단, VIX 82.7 역대 최고 → 변동성 최강
    Gym(name="COVID", difficulty=40, volatility=95),
]


def all_gyms() -> list[Gym]:
    """
    전체 체육관 목록을 돌려준다.
    list(GYMS)로 '복사본'을 만들어 주는 이유:
      바깥에서 받은 목록을 실수로 수정해도 원본 GYMS가 안 망가지게 하려는 안전장치.
    """
    return list(GYMS)
