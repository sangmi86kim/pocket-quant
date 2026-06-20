"""exam 패키지 — 시험지(과목표)는 `exam_paper`, 채점은 `grade`.

시험 과목표 심볼은 `exam_paper`에서 re-export해 기존
`from app.academy.exam import all_gyms` 경로를 그대로 유지한다.
"""
from app.academy.exam.exam_paper import GYMS, GYM_KEYS, all_gyms, gym_key

__all__ = ["GYMS", "GYM_KEYS", "all_gyms", "gym_key"]
