"""교과과정 (curriculum) — 아카데미 학사.

[책임]
`textbook.make_world()`가 찍어낸 평행세계 1권을 받아, 학기 한 코스(N권)로 엮는다.
책임 경계:
  - textbook    : **1권** 만들기 (블록 셔플 + 외부 스트림 attrs 부착)
  - curriculum  : **N권 학기 코스** (학습장 이름 부여 + DCA 페어 + train/validation 분반)

[왜 합성 체육관인가 — 학사 도입 배경]
v1.x 라인업 챔피언로드 ① 결과: 인샘플↔OOS 상관 -0.38 (인샘플 점수 높을수록
OOS에서 박살). 원인 = 6체육관 고정 = 그 특정 가격 경로에 가중치가 fit. 위기 4
+ 평시 2로 위기 편중까지 겹쳐 OOS(평시 11년) 미스매치.

학사가 매 학기 다른 N권을 짜서 트레이더가 한 경로가 아닌 "같은 분포에서 뽑은
N개 평행세계 평균"에 fit하게 만든다 → 특정 path 의존 약화 = robust.
데이터 증강(augmentation) 관점과 1:1.

[v1.1 — 다중 시계열 블록 부트스트랩]
재료: QQQ + 야생 정보원 1999-03~2020-06 (battle_frontier와 동일 — 사천왕 봉인 유지).
21일 블록 셔플로 평행세계 N개 생성. 각 세계는 합성 QQQ 가격과
`prices.attrs["external_streams"]`를 함께 가진 학습 체육관 역할.

[로드맵]
- v1.1 (이번): 다중 시계열 블록 부트스트랩 ✅
- v1 (다음): cGAN — 국면 라벨 조건 합성. mixture distribution 직접 학습.
  fat-tail/leptokurtic 보존 문제 + hold-out 봉인 규칙 (2020-07 이전 학습 only).
"""

from app.academy.curriculum.textbook import make_world
from app.pocket.battle import fight_dca
from app.pocket.models import Gym
from app.world.data_loader import LoadedGym

DATA_START = "1999-03-10"
DATA_END = "2020-06-30"
TRAIN_SEED = 42
VALIDATION_SEED = 10_042
N_TRAIN_GYMS = 20
N_VALIDATION_GYMS = 20


def bootstrap_gyms(n: int = 20, seed: int = 42,
                   start: str = DATA_START, end: str = DATA_END,
                   ticker: str = "QQQ") -> list[LoadedGym]:
    """아카데미 v1.1 — 다중 시계열 블록 부트스트랩 합성 체육관 N개.

    같은 seed면 매번 같은 N개 생성 = 학습 재현성 보장 (단일목적 sampler가
    deterministic objective 가정해도 깨지지 않음).
    """
    gyms = []

    for i in range(n):
        # seed+i: 같은 학기 안에서도 체육관마다 다른 시험지, 전체 호출은 재현 가능.
        world = make_world(seed=seed + i, start=start, end=end, ticker=ticker)
        # 체육관 이름만 알아보기 쉽게 교체 — prices attrs의 합성 외부 스트림은 그대로 보존.
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


def prepare_academy_split(
    n_train: int = N_TRAIN_GYMS,
    n_validation: int = N_VALIDATION_GYMS,
    train_seed: int = TRAIN_SEED,
    validation_seed: int = VALIDATION_SEED,
) -> tuple[tuple[list[LoadedGym], dict], tuple[list[LoadedGym], dict]]:
    """학습용 합성장과 숨은 검증 합성장을 분리해 만든다.

    Optuna objective는 train만 본다. validation은 trial 선택 후 점검용으로만 써야
    같은 합성 세계에 다시 과적합하는 사고를 줄일 수 있다.
    """
    if train_seed == validation_seed:
        raise ValueError("train_seed와 validation_seed는 달라야 한다.")
    train = prepare_academy_data(n_train, train_seed)
    validation = prepare_academy_data(n_validation, validation_seed)
    return train, validation
