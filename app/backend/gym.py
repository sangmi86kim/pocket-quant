"""
gym.py - 체육관(맵) 데이터를 정의하는 파일

실제 주가 데이터는 쓰지 않습니다(MVP라서 '가짜 데이터').
각 체육관은 이름 / 난이도(difficulty) / 변동성(volatility) 세 값만 갖습니다.
"""
from .models import Gym   # 같은 폴더 models.py의 Gym 설계도를 가져옴

# 미리 만들어 둔 체육관 4곳.
# difficulty가 높을수록 통과(생존)하기 어렵습니다.
GYMS = [
    Gym(name="DOTCOM", difficulty=90, volatility=80),            # 닷컴버블: 가장 어려움
    Gym(name="FINANCIAL_CRISIS", difficulty=85, volatility=70),  # 금융위기
    Gym(name="COVID", difficulty=40, volatility=90),             # 코로나: 쉽지만 변동성 큼
    Gym(name="RATE_SHOCK", difficulty=60, volatility=50),        # 금리쇼크: 중간 난이도
]


def all_gyms() -> list[Gym]:
    """
    전체 체육관 목록을 돌려준다.
    list(GYMS)로 '복사본'을 만들어 주는 이유:
      바깥에서 받은 목록을 실수로 수정해도 원본 GYMS가 안 망가지게 하려는 안전장치.
    """
    return list(GYMS)
