"""아카데미 패키지 — 합성 체육관 시스템.

[모듈 구성]
- academy.py : v1.1 다중 시계열 블록 부트스트랩 생성기
- world_factory.py : 합성 QQQ + 야생 정보원 attrs 생성
- (예정) cgan.py : v1 cGAN 학습기 (국면 조건 합성)

호출자는 패키지 레벨에서 직접 import:
  from app.academy import bootstrap_gyms, prepare_academy_data, prepare_academy_split
"""
from app.academy.academy import bootstrap_gyms, prepare_academy_data, prepare_academy_split

__all__ = ["bootstrap_gyms", "prepare_academy_data", "prepare_academy_split"]
