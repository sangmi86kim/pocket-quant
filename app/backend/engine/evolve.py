"""
evolve.py - 단일목적 유전 알고리즘(GA) MVP

[하는 일]
트레이더 여러 명(개체군)을 만들어 → 체육관 성적으로 점수를 매기고 →
잘한 트레이더끼리 교배 + 돌연변이시켜 다음 세대를 만든다. 이걸 여러 세대 반복.

[단일목적]
적합도(fitness) = 체육관별 적합도의 [평균 70% + 최약 30%] (models.Report 참고)
← 숫자 하나로 줄세움. 스탯블록(HP/ATK/DEF/SKILL)은 실데이터 백테스트(battle.py).

[v0.3 변경]
실데이터는 결정론적이라 같은 전략이면 매번 같은 결과 → trials 평균이 불필요.
이제 evaluate는 1회만 돈다.

[솔직한 한계 — 의도된 것]
단일목적이라 국면 간 충돌(닷컴 방어 ↔ 회복장 공격)이 숫자 하나로 뭉개진다.
그 충돌을 벡터 그대로 다루는 다목적 버전이 nsga3.py(Optuna NSGA-III) —
이 손코딩 GA는 원리 이해용 교보재로 유지한다.

[GA 4단계]
  평가(evaluate) → 선택(selection) → 교배(crossover) → 돌연변이(mutation)
"""
import random

from ..core.models import Strategy
from ..genes.signals import ALL_GENES   # 유전자 명단의 진짜 출처(실제 시그널 레지스트리)
from .battle import challenge
from .strategy import make_name


def _make(genes: list[str]) -> Strategy:
    """유전자 목록으로 전략(이름 자동) 한 마리를 만든다."""
    return Strategy(genes=genes, name=make_name(genes))


# ── 1. 평가 (적합도 계산) ─────────────────────────────
def evaluate(strategy: Strategy, loaded_gyms: list) -> dict:
    """
    전략 하나를 (미리 로딩된) 전 체육관에 도전시켜 종합 스탯·적합도를 잰다.
    실데이터는 결정론적이라 1회만 돌리면 된다.

    반환: {
      "fitness": 종합 적합도(0~100, 평균 50% + 최약 체육관 50%),
      "per_gym": {체육관: 종족치BST},     # 시장별 강함 한눈에 (표시 전용)
      "weakest": (최약 체육관 이름, 적합도),
      "stats":   종합 스탯블록(Stats),
    }
    """
    report = challenge(strategy, loaded_gyms)      # 결정론적 1회 도전
    per_gym = {r.gym_name: r.stats.bst for r in report.results}
    return {"fitness": report.fitness, "per_gym": per_gym,
            "weakest": report.weakest_gym, "stats": report.stats}


# ── 2. 선택 (자연선택) ────────────────────────────────
def select(scored: list, keep: int) -> list:
    """적합도 높은 순으로 줄세워 상위 keep마리만 부모로 남긴다(절단 선택)."""
    scored.sort(key=lambda pair: pair[1]["fitness"], reverse=True)
    return [strategy for strategy, _stats in scored[:keep]]


# ── 3. 교배 (crossover) ───────────────────────────────
def crossover(genes_a: list[str], genes_b: list[str]) -> list[str]:
    """
    두 부모의 유전자를 섞어 자식 유전자를 만든다(균등 교배).
    각 유전자마다 동전 던져서 둘 중 한 부모에게서 물려받음.
    """
    set_a, set_b = set(genes_a), set(genes_b)
    child = []
    for gene in ALL_GENES:
        source = set_a if random.random() < 0.5 else set_b
        if gene in source:          # 선택된 부모가 그 유전자를 갖고 있으면 물려받음
            child.append(gene)
    if not child:                   # 최소 1개는 보장(빈 전략 방지)
        child.append(random.choice(ALL_GENES))
    return child


# ── 4. 돌연변이 (mutation) ────────────────────────────
def mutate(genes: list[str], rate: float = 0.3) -> list[str]:
    """
    낮은 확률로 유전자 하나를 추가/제거(기존 풀 안에서). 다양성 유지용.
    이게 없으면 개체군이 한 조합으로 굳어 더 못 나아간다(조기 수렴).
    """
    genes = list(genes)
    if random.random() < rate:
        gene = random.choice(ALL_GENES)
        if gene in genes and len(genes) > 1:
            genes.remove(gene)      # 갖고 있으면 제거 (단, 최소 1개는 남김)
        elif gene not in genes:
            genes.append(gene)      # 없으면 추가
    return genes


# ── 전체 루프: 세대 진화 ──────────────────────────────
def evolve(loaded_gyms: list, pop_size: int = 20, generations: int = 10,
           on_generation=None) -> tuple:
    """
    (미리 로딩된) 체육관에서 개체군을 여러 세대 진화시키고 (최강 전략, 그 성적)을 돌려준다.

    on_generation: 세대마다 호출되는 콜백 훅. on_generation(세대번호, 최강전략, 성적)
      - 로깅 / 진행상황 출력 / (확장 시) early stop 등에 쓰는 자리.
    """
    # 안전 가드: 교배엔 부모 2마리가 필요하고, 세대는 최소 1번 돌아야 한다.
    pop_size = max(2, pop_size)
    generations = max(1, generations)

    # 초기 개체군: 랜덤 트레이더 pop_size 명
    population = [_make(random.sample(ALL_GENES, random.randint(1, len(ALL_GENES))))
                 for _ in range(pop_size)]

    best, best_stats = None, None
    keep = max(2, pop_size // 2)        # 상위 절반을 부모로 (최소 2)

    for gen in range(1, generations + 1):
        # (1) 평가: 모든 개체의 적합도 측정
        scored = [(s, evaluate(s, loaded_gyms)) for s in population]

        # 이번 세대 최강 기록
        scored.sort(key=lambda pair: pair[1]["fitness"], reverse=True)
        best, best_stats = scored[0]

        # 콜백 훅: 세대별로 바깥에 알림(출력 등은 호출자가 담당)
        if on_generation:
            on_generation(gen, best, best_stats)

        # 마지막 세대면 번식 안 하고 종료(최강만 반환)
        if gen == generations:
            break

        # (2) 선택 → (3)(4) 교배+돌연변이로 다음 세대 채우기
        parents = select(scored, keep)
        children = []
        while len(children) < pop_size:
            mom, dad = random.sample(parents, 2)
            child_genes = mutate(crossover(mom.genes, dad.genes))
            children.append(_make(child_genes))
        population = children

    return best, best_stats
