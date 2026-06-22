"""NPC 4인방 — 시즌 무관 정식 선수.

[책임]
어플삭제단·저축왕·돼지저금통·성실이를 챔피언로드 라인업에 정식 선수로 입장시키는
경로. 옛 `v1/baselines_comparison.py`(별도 후처리 도구)를 폐기하고, 시즌 어댑터가
NPC를 가중치 후보들과 같은 graduate dict 포맷으로 명단에 섞을 수 있게 한다.

[설계]
graduate dict는 본래 `{"name", "label", "weights", "params", ...}` — 시그널 가중치
경로(`combine_positions + positions_with_params`)로 채점된다. NPC는 가중치 경로와
다르므로 graduate에 추가 키 `"evaluator"`를 박는다:

  evaluator(loaded, seed_krw) -> (returns: pd.Series, terminal_balance: int)
    - returns       : 평가 구간(loaded.gym.start~end) 일별 수익률 시리즈
    - terminal      : 종료 잔고 (시드 = seed_krw)

`victory_road.run_gate1`이 graduate에 evaluator가 있으면 그걸 호출, 없으면 기존
가중치 경로로 빠진다. NPC는 도전권(score_vs_dca) 판정에서 제외 — 표시·매년 1등
매트릭스에만 참여.

[NPC 정의]
- 어플삭제단 = B&H 300명 (로켓단 컨셉) — 각자 주사위 랜덤 진입일에 풀매수(진입 전 현금),
  진입 비용 0.1% 한 번. 첫날(체육관 시작=종종 바닥) 완벽 진입 특혜를 300명으로 흩어 제거.
  대표값 = 300명의 **중앙값 단원** = '아무 날에나 산 평범한 존버러'의 공정 buy-hold 기준선.
- 저축왕     = 연 3% 무위험 복리 (낙폭 0)
- 돼지저금통  = 전부 현금 (수익 0, 금리 0)
- 성실이     = 일별 DCA, 무비용 (토스 자동 모으기)
"""
import numpy as np
import pandas as pd

from app.pocket.battle import (SAVINGS_RATE_ANNUAL, SLIPPAGE_COST, TRADE_COST,
                               TRADING_DAYS, _dca_position)
from app.world.data_loader import LoadedGym


def _slice(loaded: LoadedGym) -> tuple[pd.Series, pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(loaded.gym.start)
    end = pd.Timestamp(loaded.gym.end)
    px = loaded.prices.loc[start:end]
    return px, start, end


# 어플삭제단(로켓단) — 300명이 각자 주사위(시드 고정) 랜덤 진입일에 풀매수. 첫날 특혜 제거.
_BH_SQUAD_N = 300
_BH_ENTRY_FRACS = np.random.default_rng(20260615).random(_BH_SQUAD_N)


def _app_deletion_squad(loaded: LoadedGym, seed_krw: int) -> tuple[pd.Series, int]:
    """어플삭제단 — 300명이 각자 랜덤 진입일에 풀매수(진입 전 현금)하고 어플 삭제(존버).

    한 명 한 명은 진입 운빨이 제각각이라, 단(團) 전체의 **중앙값 단원**을 대표로 반환한다
    (returns, terminal). 옛 어플삭제맨은 무조건 첫날(체육관 시작=종종 바닥) 진입이라 '완벽
    진입' 공짜 특혜가 붙었는데, 300명으로 진입일을 흩으면 그 특혜가 사라지고 중앙값은
    '아무 날에나 산 평범한 존버러' = 공정한 buy-hold 벤치마크가 된다.
    """
    px, _, _ = _slice(loaded)
    base = px.pct_change().dropna()
    n = len(base)
    if n == 0:
        return base, seed_krw
    arr = base.to_numpy()
    members = []     # (terminal, entry_idx)
    for f in _BH_ENTRY_FRACS:
        e = min(int(f * n), n - 1)
        a = arr.copy()
        a[:e] = 0.0              # 진입 전 현금
        a[e] -= TRADE_COST + SLIPPAGE_COST  # 진입일 매수 비용 + 슬리피지
        members.append((seed_krw * float(np.prod(1.0 + a)), e))
    members.sort()
    med_term, med_e = members[len(members) // 2]   # 중앙값 단원 = 대표
    rets = base.copy()
    rets.iloc[:med_e] = 0.0
    rets.iloc[med_e] -= TRADE_COST + SLIPPAGE_COST
    return rets, int(med_term)


def _savings(loaded: LoadedGym, seed_krw: int) -> tuple[pd.Series, int]:
    """저축왕 — 연 3% 무위험 복리. 일별 (1+r)^(1/252)-1로 환산, 낙폭 0."""
    px, _, _ = _slice(loaded)
    n = max(len(px) - 1, 0)            # pct_change 기준 일수
    if n == 0:
        return pd.Series(dtype=float), seed_krw
    daily = (1 + SAVINGS_RATE_ANNUAL) ** (1 / TRADING_DAYS) - 1
    rets = pd.Series(daily, index=px.index[1:])
    terminal = int(seed_krw * (1 + daily) ** n)
    return rets, terminal


def _piggy_bank(loaded: LoadedGym, seed_krw: int) -> tuple[pd.Series, int]:
    """돼지저금통 — 전부 현금, 금리 0. 일별 수익 0, 잔고 = 시드 그대로."""
    px, _, _ = _slice(loaded)
    rets = pd.Series(0.0, index=px.index[1:]) if len(px) > 1 else pd.Series(dtype=float)
    return rets, seed_krw


def _dca(loaded: LoadedGym, seed_krw: int) -> tuple[pd.Series, int]:
    """성실이 — 일별 DCA, 수수료 0원. 슬리피지는 전원 공통으로 붙인다."""
    pos = _dca_position(loaded).shift(1)
    rets = pos * loaded.prices.pct_change() - pos.diff().abs() * SLIPPAGE_COST
    _, start, end = _slice(loaded)
    mask = (rets.index >= start) & (rets.index <= end)
    rets = rets[mask].dropna()
    terminal = int(seed_krw * float((1 + rets).cumprod().iloc[-1])) if len(rets) else seed_krw
    return rets, terminal


# graduate dict 템플릿 (run_gate1이 그대로 처리)
_NPC_GRADUATES: list[dict] = [
    {"name": "어플삭제단", "label": "B&H×300 로켓단",
     "weights": [], "params": {}, "academy": None, "specialist": False,
     "evaluator": _app_deletion_squad},
    {"name": "저축왕", "label": "연3%",
     "weights": [], "params": {}, "academy": None, "specialist": False,
     "evaluator": _savings},
    {"name": "성실이", "label": "DCA",
     "weights": [], "params": {}, "academy": None, "specialist": False,
     "evaluator": _dca},
    {"name": "돼지저금통", "label": "현금0%",
     "weights": [], "params": {}, "academy": None, "specialist": False,
     "evaluator": _piggy_bank},
]


def npc_graduates() -> list[dict]:
    """NPC 4인방 graduate dict 복사본 — 시즌 어댑터가 명단에 그대로 붙임."""
    return [dict(g) for g in _NPC_GRADUATES]
