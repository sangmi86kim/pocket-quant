"""공식 체육관 명단 — 아카데미 시험지(과목표).

[책임]
아카데미가 강한 트레이더를 길러내려면 시험지(어떤 체육관을 통과해야 하는가)도
아카데미가 정해야 한다. 시장 모듈은 입력·국면을 기술할 뿐, 시험 과목을 정하지
않는다.

[구성]
- 이 파일(`exam_paper.py`) = 시험 **과목표** (GYMS, 이름→국면키 매핑)
- `grade.py`              = 시험 **채점** (수험생 1명을 6과목에 응시시켜 점수/잔고 산출)

패키지 `__init__.py`는 이 모듈을 re-export만 한다 — `from app.academy.exam import all_gyms`
경로를 그대로 유지하기 위함.
"""

from app.pocket.models import Gym

GYMS = [
    Gym("닷컴 붕괴 체육관", difficulty=10, volatility=8,
        ticker="QQQ", start="2000-03-01", end="2002-12-31"),
    Gym("금융위기 체육관", difficulty=10, volatility=10,
        ticker="QQQ", start="2008-01-01", end="2009-06-30"),
    Gym("회복장 체육관", difficulty=6, volatility=7,
        ticker="QQQ", start="2009-03-01", end="2010-12-31"),
    Gym("코로나 급락 체육관", difficulty=9, volatility=10,
        ticker="QQQ", start="2020-02-01", end="2020-06-30"),
    Gym("불사조 상승장 체육관", difficulty=5, volatility=3,
        ticker="QQQ", start="2017-01-01", end="2017-12-31"),
    Gym("횡보장 체육관", difficulty=6, volatility=6,
        ticker="QQQ", start="2015-01-01", end="2016-12-31"),
]


# 체육관 이름 → 다목적 목적함수 키 (이름이 바뀌면 여기만 맞추면 됨)
GYM_KEYS = {
    "닷컴": "dotcom", "금융위기": "gfc", "회복장": "rebound",
    "코로나": "crash_v", "상승장": "bull", "횡보장": "chop",
}


def all_gyms() -> list[Gym]:
    """공식 시험 과목 명단(복사본)."""
    return list(GYMS)


def gym_key(gym_name: str) -> str:
    """체육관 이름에서 국면 키(dotcom/gfc/…)를 뽑는다. 채점·라벨이 공용."""
    for token, key in GYM_KEYS.items():
        if token in gym_name:
            return key
    raise KeyError(f"[exam] 국면 키를 모르는 체육관: {gym_name!r} — GYM_KEYS에 추가 필요")
