"""
strategy.py - 트레이더(전략)를 '만드는' 파일

하는 일은 두 가지:
  1) 포켓몬(시그널)을 랜덤하게 뽑아서 트레이더 한 명을 생성한다 (단판 모드의 입구).
  2) 그 트레이더에 어울리는 이름을 자동으로 지어준다 (연출용).
※ 용어: '마리'는 포켓몬(시그널) 전용, 전략은 트레이더 '명'으로 센다.
"""
import random

from ..core.models import Strategy
from ..genes.signals import ALL_GENES   # 유전자 명단의 진짜 출처(실제 시그널 레지스트리)

# 이름 자동 생성에 쓸 단어 풀 — 순전히 연출용, 판정과 무관
SUFFIXES = ["몬", "드래곤", "킹", "마스터"]
TITLES = ["ATH", "디아블로", "헤르메스", "타이탄"]


def make_name(genes: list[str]) -> str:
    """유전자 조합으로 전략 이름을 짓는다. 20%는 칭호형(타이탄 드래곤),
    나머지는 조합형(DD-MA몬 — '몬'이 살짝 더 자주 나오게 풀에 중복)."""
    if random.random() < 0.2:
        return f"{random.choice(TITLES)} {random.choice(SUFFIXES[1:])}"
    return f"{'-'.join(genes)}{random.choice(SUFFIXES[:1] + SUFFIXES)}"


def create_strategy(gene_count: int | None = None) -> Strategy:
    """트레이더 한 명을 만든다. gene_count=None이면 데려갈 포켓몬 수도 랜덤.
    범위는 항상 1 ~ len(ALL_GENES)로 강제 — 빈 전략(유전자 0개) 방지."""
    if gene_count is None:
        gene_count = random.randint(1, len(ALL_GENES))
    gene_count = max(1, min(gene_count, len(ALL_GENES)))

    genes = random.sample(ALL_GENES, gene_count)    # 중복 없이 뽑기
    return Strategy(genes=genes, name=make_name(genes))
