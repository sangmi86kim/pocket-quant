"""교과과정 (curriculum) — 아카데미 학사.

[책임]
`textbook.make_world()`가 찍어낸 평행세계 1권을 받아, 학기 한 코스(N권)로 엮는다.
책임 경계:
  - textbook: 1권 만들기 (블록 셔플 + 외부 스트림 attrs 부착)
  - course: N권 학기 코스 (학습장 이름 부여 + DCA 페어 + train/validation 분반)

[교과서]
재료: QQQ + 야생 정보원 1999-03~2020-06 (사천왕 봉인 유지).
각 세계는 합성 QQQ 가격과 `prices.attrs["external_streams"]`를 함께 가진 학습 체육관 역할.

- `block`: 21일 블록을 무작위로 잇는 기본 교과서.
- `rs`: 국면 순서를 먼저 뽑고, 해당 국면의 진짜 21일 토막을 끼우는 현재 기본 교과서.
"""

from app.academy.curriculum.textbook import make_world, make_world_rs
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
                   ticker: str = "QQQ",
                   textbook: str = "rs",
                   skew: dict | None = None,
                   name_prefix: str = "아카데미") -> list[LoadedGym]:
    """아카데미 합성 체육관 N개.

    같은 seed면 매번 같은 N개 생성 = 학습 재현성 보장 (단일목적 sampler가
    deterministic objective 가정해도 깨지지 않음).
    """
    gyms = []

    for i in range(n):
        # seed+i: 같은 학기 안에서도 체육관마다 다른 시험지, 전체 호출은 재현 가능.
        if textbook == "block":
            world = make_world(seed=seed + i, start=start, end=end, ticker=ticker)
        elif textbook == "rs":
            world = make_world_rs(seed=seed + i, start=start, end=end,
                                  ticker=ticker, skew=skew)
        else:
            raise ValueError(f"unknown textbook: {textbook!r}")
        # 체육관 이름만 알아보기 쉽게 교체 — prices attrs의 합성 외부 스트림은 그대로 보존.
        gyms.append(LoadedGym(
            gym=Gym(f"{name_prefix}#{i+1:02d}", difficulty=0, volatility=0,
                    ticker="SYNTH", start=world.gym.start, end=world.gym.end),
            prices=world.prices,
        ))
    return gyms


def prepare_academy_data(n_gyms: int = 20, seed: int = 42,
                         textbook: str = "rs",
                         skew: dict | None = None,
                         name_prefix: str = "아카데미",
                         ) -> tuple[list[LoadedGym], dict]:
    """아카데미 체육관 + 성실이(DCA) 기준선.

    tpe/cma_es/gp.prepare_data와 같은 인터페이스 — sweep 어댑터에 그대로 주입.
    """
    gyms = bootstrap_gyms(n_gyms, seed, textbook=textbook, skew=skew,
                          name_prefix=name_prefix)
    dca = {lg.gym.name: fight_dca(lg) for lg in gyms}
    return gyms, dca


def prepare_academy_split(
    n_train: int = N_TRAIN_GYMS,
    n_validation: int = N_VALIDATION_GYMS,
    train_seed: int = TRAIN_SEED,
    validation_seed: int = VALIDATION_SEED,
    textbook: str = "rs",
) -> tuple[tuple[list[LoadedGym], dict], tuple[list[LoadedGym], dict]]:
    """학습용 합성장과 숨은 검증 합성장을 분리해 만든다.

    Optuna objective는 train만 본다. validation은 trial 선택 후 점검용으로만 써야
    같은 합성 세계에 다시 과적합하는 사고를 줄일 수 있다.
    """
    if train_seed == validation_seed:
        raise ValueError("train_seed와 validation_seed는 달라야 한다.")
    train = prepare_academy_data(n_train, train_seed, textbook=textbook,
                                 name_prefix="TRAIN")
    validation = prepare_academy_data(n_validation, validation_seed,
                                      textbook=textbook, name_prefix="VALID")
    return train, validation
