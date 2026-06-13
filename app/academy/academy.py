"""아카데미 — 합성 체육관 시스템 (사용자 안, 2026-06-14 도입 선언).

[왜 아카데미인가]
v1.x 라인업 챔피언로드 ① 결과: 인샘플↔OOS 상관 -0.38 (인샘플 점수 높을수록
OOS에서 박살). 원인 = 6체육관 고정 = 그 특정 가격 경로에 가중치가 fit. 위기 4
+ 평시 2로 위기 편중까지 겹쳐 OOS(평시 11년) 미스매치.

아카데미는 **합성 체육관**을 매번 생성해 학습장 자체를 변동시킨다. 트레이너가
한 경로가 아닌 "같은 분포에서 뽑은 N개 평행세계 평균"에 fit → 특정 path 의존
약화 = robust. 본업의 augmentation 정신과 1:1.

[v0 — 블록 부트스트랩 1기]
재료: QQQ 1999-03~2020-06 (battle_frontier와 동일 — 사천왕 봉인 유지).
21일 블록 셔플로 평행세계 N개 생성. 각 세계는 학습 체육관 역할.

[로드맵 — 메모리]
- v0 (이번): 블록 부트스트랩 ✅
- v1 (다음): cGAN — 국면 라벨 조건 합성. mixture distribution 직접 학습.
  fat-tail/leptokurtic 보존 문제 + hold-out 봉인 규칙 (2020-07 이전 학습 only).
"""
from __future__ import annotations

from app.backend.data_io.data import LoadedGym, get_prices
from app.backend.core.models import Gym
from app.backend.engine.battle import fight_dca
from app.league.battle_frontier import DATA_START, DATA_END, make_world

import numpy as np


def bootstrap_gyms(n: int = 20, seed: int = 42,
                   start: str = DATA_START, end: str = DATA_END,
                   ticker: str = "QQQ") -> list[LoadedGym]:
    """아카데미 1기 — 블록 부트스트랩 합성 체육관 N개.

    같은 seed면 매번 같은 N개 생성 = 학습 재현성 보장 (단일목적 sampler가
    deterministic objective 가정해도 깨지지 않음).
    """
    prices = get_prices(ticker, start, end)
    full_returns = prices.pct_change().dropna()
    rng = np.random.default_rng(seed)
    gyms = []
    for i in range(n):
        world = make_world(full_returns, rng, None)         # 전천후 풀
        # 체육관 이름만 알아보기 쉽게 교체 — prices는 그대로 (합성 시계열)
        gyms.append(LoadedGym(
            gym=Gym(f"아카데미#{i+1:02d}", difficulty=0, volatility=0,
                    ticker="SYNTH", start=world.gym.start, end=world.gym.end),
            prices=world.prices,
        ))
    return gyms


def prepare_academy_data(n_gyms: int = 20, seed: int = 42
                          ) -> tuple[list[LoadedGym], dict]:
    """아카데미 체육관 + 성실이(DCA) 기준선.

    tpe/cma_es/gp.prepare_data와 같은 인터페이스 — sweep 어댑터에 그대로 주입.
    """
    gyms = bootstrap_gyms(n_gyms, seed)
    dca = {lg.gym.name: fight_dca(lg) for lg in gyms}
    return gyms, dca
