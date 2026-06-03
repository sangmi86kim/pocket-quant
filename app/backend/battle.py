"""
battle.py - 전투(백테스트)를 '계산'하는 파일, 게임의 엔진

[책임] 순수 계산만 한다. 데이터 로딩(I/O)은 data.py가 미리 끝내서 넘겨준다.
흐름:
  fight()     : 전략 1마리 vs 미리 로딩된 체육관 1곳 -> 그 시장에서의 스탯블록
  challenge() : 전략 1마리 vs 여러 체육관           -> 종합 성적표(Report)

[핵심 계산] (가격은 이미 LoadedGym으로 받아둔 상태)
  1) signals.combined_position 으로 일별 포지션(0~1)을 만든다
  2) 포지션을 하루 lag(shift 1)해서 다음날 수익에 적용 (룩어헤드 방지)
  3) 워밍업 버퍼를 잘라내고 평가 구간만 남긴다
  4) 자산곡선에서 CAGR / 최대낙폭 / 샤프 / 평균현금 을 뽑는다
  5) 그 원시값들을 0~100 스탯(HP/ATK/DEF/SKILL)으로 정규화한다
"""
import numpy as np
import pandas as pd

from .data import LoadedGym
from .models import BattleResult, Report, Stats, Strategy
from .signals import combined_position

TRADING_DAYS = 252          # 연율화 기준 거래일 수

# ── 스탯 정규화 구간 (이 양 끝값이 0점 / 100점) — 튜닝 포인트 ──
ATK_CAGR_LO, ATK_CAGR_HI = -0.25, 0.25     # CAGR -25% -> 0,  +25% -> 100
SKILL_SHARPE_LO, SKILL_SHARPE_HI = -1.0, 3.0  # 샤프 -1 -> 0, 3 -> 100


def _scale(value: float, lo: float, hi: float) -> float:
    """value를 [lo,hi] 구간에서 0~100으로 선형 변환(범위 밖은 0/100으로 클램프)."""
    if hi == lo:
        return 0.0
    pct = (value - lo) / (hi - lo) * 100.0
    return float(min(100.0, max(0.0, pct)))


def fight(strategy: Strategy, loaded: LoadedGym) -> BattleResult:
    """전략 한 마리가 (미리 로딩된) 한 시장 국면을 통과하며 만든 스탯블록을 계산한다."""
    gym = loaded.gym
    prices = loaded.prices          # 이미 워밍업 버퍼 포함해 받아둔 가격

    # (1) 일별 포지션(0~1) + (2) 하루 lag 적용한 전략/시장 수익
    position = combined_position(strategy.genes, prices)
    effective_position = position.shift(1)
    market_ret = prices.pct_change()
    strat_ret = effective_position * market_ret

    # (3) 평가 구간(체육관 기간)만 잘라낸다 — 앞쪽 버퍼는 지표 데우는 데만 쓰임
    window_start = pd.Timestamp(gym.start)
    window_end = pd.Timestamp(gym.end)
    mask = (prices.index >= window_start) & (prices.index <= window_end)
    strat_ret = strat_ret[mask].dropna()
    market_ret = market_ret[mask].dropna()
    effective_position = effective_position[mask].dropna()

    # 데이터가 비정상적으로 비면 0점 스탯으로 방어 반환
    if len(strat_ret) < 2:
        return BattleResult(gym_name=gym.name, stats=Stats())

    # (4) 자산곡선 → 지표들
    equity = (1.0 + strat_ret).cumprod()
    cagr = float(equity.iloc[-1] ** (TRADING_DAYS / len(strat_ret)) - 1.0)

    my_dd = (equity / equity.cummax() - 1.0).min()              # 음수
    market_equity = (1.0 + market_ret).cumprod()
    market_dd = (market_equity / market_equity.cummax() - 1.0).min()  # 음수

    std = strat_ret.std()
    sharpe = float(strat_ret.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0

    avg_cash = float((1.0 - effective_position).mean())         # 0~1

    # (5) 0~100 스탯으로 정규화
    hp = _scale(avg_cash, 0.0, 1.0)                             # 현금 100% -> HP 100
    atk = _scale(cagr, ATK_CAGR_LO, ATK_CAGR_HI)
    # 방어력: 시장 낙폭 대비 내가 얼마나 덜 빠졌나. 시장이 안 빠졌으면(0) 만점.
    if market_dd < 0:
        def_ratio = 1.0 - (float(my_dd) / float(market_dd))     # 둘 다 음수 → 양수 비율
        deff = _scale(def_ratio, 0.0, 1.0)
    else:
        deff = 100.0
    skill = _scale(sharpe, SKILL_SHARPE_LO, SKILL_SHARPE_HI)

    stats = Stats(hp=hp, atk=atk, def_=deff, skill=skill)
    return BattleResult(
        gym_name=gym.name, stats=stats, cagr=cagr,
        max_drawdown=float(my_dd), market_drawdown=float(market_dd),
    )


def challenge(strategy: Strategy, loaded_gyms: list[LoadedGym]) -> Report:
    """전략 한 마리가 (미리 로딩된) 여러 시장 국면을 모두 통과하며 성적표를 만든다."""
    report = Report(strategy=strategy)
    for loaded in loaded_gyms:
        report.results.append(fight(strategy, loaded))
    return report
