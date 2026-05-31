"""
strategy.py - 전략 포켓몬을 '만드는' 파일

하는 일은 두 가지:
  1) 유전자를 랜덤하게 뽑아서 전략 한 마리를 생성한다.
  2) 그 전략에 어울리는 이름을 자동으로 지어준다.
"""
import random   # 무작위(랜덤) 기능을 쓰기 위한 표준 라이브러리

# 같은 폴더(app/backend) 안의 models.py에서 필요한 것들을 가져온다.
# 맨 앞의 점(.)은 "같은 패키지(폴더) 안에서 가져와라"는 뜻 (상대 import)
from .models import ALL_GENES, Strategy

# 이름 자동 생성에 쓸 단어 풀(pool)
SUFFIXES = ["몬", "드래곤", "킹", "마스터"]          # 접미사: DD'몬', ATH '드래곤'
TITLES = ["ATH", "디아블로", "헤르메스", "타이탄"]    # 가끔 붙는 멋진 칭호


def make_name(genes: list[str]) -> str:
    """유전자 조합을 받아서 전략 이름 문자열을 만들어 돌려준다."""
    # 유전자들을 '-'로 이어붙임. 예) ["DD","RSI"] -> "DD-RSI"
    base = "-".join(genes)

    # random.random()은 0.0 ~ 1.0 사이 랜덤 소수.
    # 그게 0.2보다 작을 확률(=20%)일 때만 '멋진 칭호' 이름을 준다.
    if random.random() < 0.2:
        # random.choice(리스트) : 리스트에서 무작위로 하나 뽑기
        # SUFFIXES[1:] = "몬"을 뺀 나머지(드래곤/킹/마스터) -> 칭호엔 화려한 접미사
        return f"{random.choice(TITLES)} {random.choice(SUFFIXES[1:])}"

    # 나머지 80%는 평범하게: "유전자들 + 접미사"
    # SUFFIXES[:1] + SUFFIXES = ["몬"] + 전체  -> "몬"이 뽑힐 확률을 살짝 높인 것
    return f"{base}{random.choice(SUFFIXES[:1] + SUFFIXES)}"


def create_strategy(gene_count: int | None = None) -> Strategy:
    """
    전략 포켓몬 한 마리를 만들어 돌려준다.

    gene_count : 유전자를 몇 개 가질지.
      - 'int | None' = 정수 또는 None(둘 다 가능)이라는 뜻
      - 기본값 None -> "안 정해주면 개수도 랜덤으로 정한다"
    """
    if gene_count is None:
        # 1개 ~ 전체개수(5개) 사이에서 랜덤으로 개수 결정
        gene_count = random.randint(1, len(ALL_GENES))

    # 안전장치: 개수가 너무 작거나 크면 1~5 범위로 강제로 맞춘다.
    #   max(1, ...) -> 최소 1 보장,  min(..., 5) -> 최대 5 보장
    gene_count = max(1, min(gene_count, len(ALL_GENES)))

    # random.sample(목록, 개수) : 목록에서 '중복 없이' 개수만큼 뽑기
    # 예) ALL_GENES에서 3개 -> ["MA", "DD", "FX"] 같은 식
    genes = random.sample(ALL_GENES, gene_count)

    # 뽑은 유전자 + 자동 생성한 이름으로 전략(데이터)을 만들어 반환
    return Strategy(genes=genes, name=make_name(genes))
