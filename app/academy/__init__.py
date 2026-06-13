"""아카데미 패키지 — 합성 체육관 시스템.

[모듈 구성]
- academy.py : v0 블록 부트스트랩 생성기 (현재)
- (예정) cgan.py : v1 cGAN 학습기 (국면 조건 합성)

호출자는 패키지 레벨에서 직접 import:
  from app.academy import bootstrap_gyms, prepare_academy_data
"""
from app.academy.academy import bootstrap_gyms, prepare_academy_data

__all__ = ["bootstrap_gyms", "prepare_academy_data"]
