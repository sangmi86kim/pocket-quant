"""NSGA-III 학교 엔진 — 합성장 3목적 최적화.

[왜 학교 전용인가]
학교 밖에서 따로 NSGA-III를 돌리지 않는다. 6체육관/5국면 NSGA는 폐기했고,
이 패키지가 학교 2차 교육의 다목적 엔진이다.

[학교 목적]
  maximize  평균 누적자산
  maximize  최악 누적자산
  minimize  turnover

단일목적(TPE/CMA-ES/GP)은 누적자산 합 1등을 찾고, 이 패키지는 평균도 좋고 최악장도
버티며 매매도 덜 하는 트레이더를 찾는다. 성실이/어플삭제맨 비교는 objective가
아니라 졸업 필터에서 본다.

[모듈 구성 — 한 파일을 책임별로 뽀갬]
  - objectives : 목적 상수 + 후보 제시 + 3목적 환산 + 학기 데이터 준비
  - callbacks  : HV 조기종료 + 적응 변이율 (sampler 튜닝 기계장치)
  - graduate   : 졸업 필터/라벨 (채점·선발 성격)
  - engine     : sampler 배선 + study.optimize (run_study)

공개 API는 예전 단일 모듈과 동일하다 — `from ...classroom import nsga3` 후
`nsga3.run_study(...)` 식 호출이 그대로 동작한다.
"""
from app.academy.training.classroom.nsga3.callbacks import (
    adaptive_mutation_callback,
    hv_early_stop_callback,
)
from app.academy.training.classroom.nsga3.engine import run_study
from app.academy.training.classroom.nsga3.graduate import summarize_front
from app.academy.training.classroom.nsga3.objectives import (
    DIRECTIONS,
    OBJECTIVE_NAMES,
    SEED_KRW,
    academy_metrics,
    evaluate_objectives,
    make_objective,
    prepare_data,
    roll_seed,
    suggest_candidate,
)

__all__ = [
    "DIRECTIONS",
    "OBJECTIVE_NAMES",
    "SEED_KRW",
    "academy_metrics",
    "adaptive_mutation_callback",
    "evaluate_objectives",
    "hv_early_stop_callback",
    "make_objective",
    "prepare_data",
    "roll_seed",
    "run_study",
    "suggest_candidate",
    "summarize_front",
]
