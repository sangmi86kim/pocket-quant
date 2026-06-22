"""NSGA-III 학교 엔진 — 합성장 2목적 최적화 + turnover 스펙 필터.

[왜 학교 전용인가]
학교 밖에서 따로 NSGA-III를 돌리지 않는다. 6체육관/5국면 NSGA는 폐기했고,
이 패키지가 학교 2차 교육의 다목적 엔진이다.

[학교 목적]
  maximize  중앙 누적자산
  maximize  최악 누적자산

단일목적(TPE/CMA-ES/GP)은 누적자산 합 1등을 찾고, 이 패키지는 평균도 좋고 최악장도
버티는 트레이더를 찾는다. turnover는 돈 버는 목적이 아니라 운용 스펙이므로 objective가
아니라 졸업 필터에서 cap으로 자른다.

[모듈 구성 — 한 파일을 책임별로 뽀갬]
  - objectives : 목적 상수 + 후보 제시 + 2목적 환산 + 학기 데이터 준비
  - callbacks  : HV 트렌드 트래커 + 조기종료/적응변이 리스너 (sampler 튜닝 기계장치)
  - graduate   : 졸업 필터/라벨 (채점·선발 성격)
  - engine     : sampler 배선 + study.optimize (run_study)

공개 API는 예전 단일 모듈과 동일하다 — `from ...multi_objective import nsga3` 후
`nsga3.run_study(...)` 식 호출이 그대로 동작한다.
"""
from app.academy.training.multi_objective.nsga3.callbacks import (
    AdaptiveMutation,
    HVEarlyStopper,
    HVTrendTracker,
)
from app.academy.training.multi_objective.nsga3.engine import run_study
from app.academy.training.multi_objective.nsga3.graduate import summarize_front
from app.academy.training.multi_objective.nsga3.objectives import (
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
    "AdaptiveMutation",
    "DIRECTIONS",
    "HVEarlyStopper",
    "HVTrendTracker",
    "OBJECTIVE_NAMES",
    "SEED_KRW",
    "academy_metrics",
    "evaluate_objectives",
    "make_objective",
    "prepare_data",
    "roll_seed",
    "run_study",
    "suggest_candidate",
    "summarize_front",
]
