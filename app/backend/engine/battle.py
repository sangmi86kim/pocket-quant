"""
battle.py - 전투(백테스트)를 '계산'하는 파일, 게임의 엔진

[책임] 순수 계산만 한다. 데이터 로딩(I/O)은 data.py가 미리 끝내서 넘겨준다.
흐름:
  fight()        : 전략 1명 vs 미리 로딩된 체육관 1곳 -> 그 시장에서의 스탯블록
  challenge()    : 전략 1명 vs 여러 체육관            -> 종합 성적표(Report)
  fight_dca()    : 라이벌 '성실이'(일별 DCA, 무비용) 소환 -> 같은 체육관 성적
  score_vs_dca() : 그 체육관에서 성실이 대비 얼마나 나았나 (NSGA-III 목적 재료)
스탯(0~100)은 사람 읽기용, 최적화는 BattleResult의 raw 지표(sharpe/turnover 등)를 쓴다.

[핵심 계산] (가격은 이미 LoadedGym으로 받아둔 상태)
  1) signals.combined_position 으로 일별 포지션(0~1)을 만든다
  2) 포지션을 하루 lag(shift 1)해서 다음날 수익에 적용 (룩어헤드 방지)
     + 포지션 변화량(턴오버) × TRADE_COST(토스 0.1%)를 거래비용으로 차감
  3) 워밍업 버퍼를 잘라내고 평가 구간만 남긴다
  4) 자산곡선에서 CAGR / 최대낙폭 / 샤프 / 평균현금 을 뽑는다
  5) 그 원시값들을 0~100 스탯(HP/ATK/DEF/SKILL)으로 정규화한다
"""
import numpy as np
import pandas as pd

from ..core.models import BattleResult, Report, Stats, Strategy
from ..genes.signals import combined_position
from ..market.data import LoadedGym

TRADING_DAYS = 252          # 연율화 기준 거래일 수

# ── 거래비용 (토스증권 미국주식 기준) ──
# 위탁수수료 = 거래대금의 0.1% (2025-12-01부터 상시 적용). 매도 시 SEC Fee 0.00206%는
# 무시 가능한 수준이라 제외. 포지션 변화량(턴오버)만큼 편도 수수료를 차감한다:
#   예) 현금(0) → 풀매수(1) 진입 = 자본의 100% 거래 = 0.1% 비용.
# 평가 구간 안의 매매만 과금한다(워밍업 중 진입한 초기 포지션은 무료 = 구간 철학과 일치).
TRADE_COST = 0.001

# ── 스탯 정규화 구간 (이 양 끝값이 0점 / 100점) — 튜닝 포인트 ──
# [퇴화 방지 재설계] 예전 스케일(-25%~+25%)에선 '전부 현금'(CAGR 0%)이 ATK 50점을
# 공짜로 받아 수익 차이에 둔감했다. 0%를 0점으로 내려 '안 벌면 공격력 없음'으로 만든다.
ATK_CAGR_LO, ATK_CAGR_HI = 0.0, 0.25          # CAGR 0% -> 0, +25% -> 100
SKILL_SHARPE_LO, SKILL_SHARPE_HI = -1.0, 3.0  # 샤프 -1 -> 0, 3 -> 100
# DEF = Calmar(CAGR / |내MDD|) 기반. 예전 '1 - 내MDD/시장MDD'는 비중을 줄일수록
# 거의 1:1로 점수가 올라 '아무것도 안 하기'가 방어 만점이었다. Calmar는 비중 일괄
# 축소에 거의 불변(분자/분모가 같이 줄어듦)이라 그 퇴화 경사가 사라지고,
# '낙폭을 적게 겪으면서 번 놈'만 진짜 방어 점수를 받는다.
DEF_CALMAR_LO, DEF_CALMAR_HI = -1.0, 3.0      # Calmar -1 -> 0, 3 -> 100 (현금=0 -> 25)


def _scale(value: float, lo: float, hi: float) -> float:
    """value를 [lo,hi] 구간에서 0~100으로 선형 변환(범위 밖은 0/100으로 클램프)."""
    if hi == lo:
        return 0.0
    pct = (value - lo) / (hi - lo) * 100.0
    return float(min(100.0, max(0.0, pct)))


def fight(strategy: Strategy, loaded: LoadedGym) -> BattleResult:
    """전략 한 마리가 (미리 로딩된) 한 시장 국면을 통과하며 만든 스탯블록을 계산한다."""
    # (1) 유전자들로 일별 포지션(0~1)을 만들고, 공용 채점기에 넘긴다
    position = combined_position(strategy.genes, loaded.prices)
    return _score_position(position, loaded)


def _score_position(position, loaded: LoadedGym,
                    trade_cost: float | None = None) -> BattleResult:
    """일별 포지션(0~1) 시계열 하나를 채점한다 — fight와 DCA 기준선이 공유하는 엔진.
    실행 모델(하루 lag, 턴오버 과금, 워밍업 컷)이 모든 참가자에게 동일해야 공정 비교다.

    trade_cost: None이면 모듈 전역 TRADE_COST(워크포워드가 몽키패치하는 그 값).
    DCA 기준선만 0.0을 넘긴다 — 토스 '주식 자동 모으기'는 매수 수수료 0원이라
    실제 비용 구조가 비대칭이기 때문(전략의 타이밍 매매는 0.1% 그대로)."""
    gym = loaded.gym
    prices = loaded.prices          # 이미 워밍업 버퍼 포함해 받아둔 가격

    cost = TRADE_COST if trade_cost is None else trade_cost

    # (2) 하루 lag 적용한 전략/시장 수익
    effective_position = position.shift(1)
    market_ret = prices.pct_change()
    turnover = effective_position.diff().abs()              # 그날 매매한 자본 비율
    strat_ret = effective_position * market_ret - turnover * cost

    # (3) 평가 구간(체육관 기간)만 잘라낸다 — 앞쪽 버퍼는 지표 데우는 데만 쓰임
    window_start = pd.Timestamp(gym.start)
    window_end = pd.Timestamp(gym.end)
    mask = (prices.index >= window_start) & (prices.index <= window_end)
    strat_ret = strat_ret[mask].dropna()
    market_ret = market_ret[mask].dropna()
    effective_position = effective_position[mask].dropna()
    turnover = turnover[mask].dropna()

    # 데이터가 비정상적으로 비면 0점 스탯으로 방어 반환
    if len(strat_ret) < 2:
        return BattleResult(gym_name=gym.name, stats=Stats())

    # (4) 자산곡선 → 지표들
    equity = (1.0 + strat_ret).cumprod()
    cagr = float(equity.iloc[-1] ** (TRADING_DAYS / len(strat_ret)) - 1.0)
    total_return = float(equity.iloc[-1] - 1.0)                 # 기간 총수익(실투자 시뮬용)

    my_dd = (equity / equity.cummax() - 1.0).min()              # 음수
    market_equity = (1.0 + market_ret).cumprod()
    market_dd = (market_equity / market_equity.cummax() - 1.0).min()  # 음수
    market_return = float(market_equity.iloc[-1] - 1.0)        # 단순보유 기간 총수익(비교용)

    std = strat_ret.std()
    sharpe = float(strat_ret.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0

    avg_cash = float((1.0 - effective_position).mean())         # 0~1

    # (5) 0~100 스탯으로 정규화
    hp = _scale(avg_cash, 0.0, 1.0)                             # 현금 100% -> HP 100 (표시 전용)
    atk = _scale(cagr, ATK_CAGR_LO, ATK_CAGR_HI)
    # 방어력 = Calmar(CAGR / |내MDD|): 낙폭 한 단위당 얼마나 벌었나(위험조정 수익).
    # 낙폭이 0이면 나누기가 안 되므로: 벌었으면 만점, 못 벌었으면(현금) Calmar 0 취급.
    if my_dd < 0:
        calmar = cagr / abs(float(my_dd))
        deff = _scale(calmar, DEF_CALMAR_LO, DEF_CALMAR_HI)
    else:
        deff = 100.0 if cagr > 0 else _scale(0.0, DEF_CALMAR_LO, DEF_CALMAR_HI)
    skill = _scale(sharpe, SKILL_SHARPE_LO, SKILL_SHARPE_HI)

    stats = Stats(hp=hp, atk=atk, def_=deff, skill=skill)
    return BattleResult(
        gym_name=gym.name, stats=stats, cagr=cagr,
        total_return=total_return, market_return=market_return,
        max_drawdown=float(my_dd), market_drawdown=float(market_dd),
        sharpe=sharpe, turnover=float(turnover.mean()) if len(turnover) else 0.0,
    )


# ──────────────────────────────────────────────
# DCA 기준선 — "이길 대상"은 단순보유가 아니라 사용자의 실제 적립 머신
# (토스 매일 $20 QQQM 자동매수). 금액은 비율이라 무관하다.
# ──────────────────────────────────────────────
def _dca_position(loaded: LoadedGym):
    """일별 DCA를 포지션 스케줄로 표현한다.

    평가 구간이 N거래일이면 매일 종가에 총자본의 1/N씩 사 모은다:
    k번째 거래일 매수 후 투자 비중 = k/N (시작 0 → 끝 100%, 매도 없음).
    같은 채점기(_score_position)를 타므로 하루 lag·거래비용(매수마다 0.1%)도
    전략과 똑같이 적용된다 = 공정 비교."""
    prices = loaded.prices
    mask = (prices.index >= pd.Timestamp(loaded.gym.start)) \
         & (prices.index <= pd.Timestamp(loaded.gym.end))
    position = pd.Series(0.0, index=prices.index)
    n = int(mask.sum())
    if n > 0:
        position[mask] = np.arange(1, n + 1) / n
    return position


def fight_dca(loaded: LoadedGym) -> BattleResult:
    """라이벌 '성실이'(DCA 기준선)의 성적 — 한 체육관을 매일 1/N 적립으로 통과한 결과.
    수수료 0원: 토스 '주식 자동 모으기'는 매수 수수료 면제(사용자 실계좌 확인,
    2026-06-11). 전략은 타이밍 매매라 면제 대상이 아님 = 비대칭이 현실이고,
    그만큼 DCA를 이기는 기준이 높아진다."""
    return _score_position(_dca_position(loaded), loaded, trade_cost=0.0)


def score_vs_dca(strat: BattleResult, dca: BattleResult) -> float:
    """그 체육관에서 전략이 DCA보다 얼마나 나았나 (양수 = DCA 머신 개선).

    NSGA-III 목적함수 재료 (코덱스 설계, 가중치는 단순 고정 — 과적합 방지):
      0.4 × 수익 차이      (전략 총수익 − DCA 총수익)
      0.4 × 낙폭 개선      (DCA 낙폭 − 전략 낙폭, 얕을수록 +)
      0.2 × 샤프 차이      (위험 대비 효율)
    raw 지표만 사용 — 0~100 클램프 스탯/BST 금지(현금 뒷문 방지)."""
    return (0.4 * (strat.total_return - dca.total_return)
            + 0.4 * (abs(dca.max_drawdown) - abs(strat.max_drawdown))
            + 0.2 * (strat.sharpe - dca.sharpe))


def challenge(strategy: Strategy, loaded_gyms: list[LoadedGym]) -> Report:
    """전략 한 마리가 (미리 로딩된) 여러 시장 국면을 모두 통과하며 성적표를 만든다."""
    report = Report(strategy=strategy)
    for loaded in loaded_gyms:
        report.results.append(fight(strategy, loaded))
    return report
