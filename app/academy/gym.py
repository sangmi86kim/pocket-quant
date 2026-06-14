"""Official gym battles for strategy search.

Academy owns the gym challenge set because the academy's job is to train
strong traders. Market modules should describe market inputs/regimes, not decide
which gym leaders a trader must defeat.
"""

from app.academy.core.models import Gym

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


def all_gyms() -> list[Gym]:
    """Return a copy of the official gym battle list."""
    return list(GYMS)
