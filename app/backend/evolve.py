"""
evolve.py - 단일목적 유전 알고리즘(GA) MVP

[하는 일]
전략 여러 마리(개체군)를 만들어 → 체육관 성적으로 점수를 매기고 →
잘한 놈끼리 교배 + 돌연변이시켜 다음 세대를 만든다. 이걸 여러 세대 반복.

[단일목적]
적합도(fitness) = 전 체육관 평균 생존률  ← 숫자 하나로 줄세움.

[솔직한 한계 — 의도된 것]
지금 점수 체계에선 '유전자 많을수록 무조건 유리'라, 결국 전 유전자 조합으로 수렴한다.
그래도 OK. 이 MVP의 목적은 두 가지다:
  (1) GA 기계(선택→교배→돌연변이→세대)가 제대로 도는지 검증
  (2) "최적 전략조차 어느 시장에서 박살나는지" 두 눈으로 관찰
이 관찰이 나중에 왜 다목적(NSGA-III)/타입상성이 필요한지를 알려준다.

[GA 4단계]
  평가(evaluate) → 선택(selection) → 교배(crossover) → 돌연변이(mutation)
"""
import random

from .battle import challenge
from .models import ALL_GENES, Strategy
from .strategy import make_name


def _make(genes: list[str]) -> Strategy:
    """유전자 목록으로 전략(이름 자동) 한 마리를 만든다."""
    return Strategy(genes=genes, name=make_name(genes))


# ── 1. 평가 (적합도 계산) ─────────────────────────────
def evaluate(strategy: Strategy, gyms: list, trials: int = 20) -> dict:
    """
    전략 하나를 여러 번(trials) 도전시켜 평균 생존률을 잰다.
    전투에 랜덤 보정(±20)이 있어서, 한 번만 재면 운에 휘둘린다 → 여러 번 평균.

    반환: {"fitness": 전체평균생존률, "per_gym": {체육관: 생존률}}
    """
    per_gym = {g.name: 0 for g in gyms}        # 체육관별 생존 횟수 카운터
    for _ in range(trials):
        report = challenge(strategy, gyms)     # 한 바퀴 도전
        for r in report.results:
            if r.survived:
                per_gym[r.gym_name] += 1
    # 횟수 → 생존률(0~1)로 환산
    per_gym_rate = {name: cnt / trials for name, cnt in per_gym.items()}
    # 단일목적: 체육관별 생존률을 전부 평균낸 값 하나
    fitness = sum(per_gym_rate.values()) / len(gyms)
    return {"fitness": fitness, "per_gym": per_gym_rate}


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
def evolve(gyms: list, pop_size: int = 20, generations: int = 10,
           trials: int = 20, on_generation=None) -> tuple:
    """
    개체군을 여러 세대 진화시키고 (최강 전략, 그 성적)을 돌려준다.

    on_generation: 세대마다 호출되는 콜백 훅. on_generation(세대번호, 최강전략, 성적)
      - 로깅 / 진행상황 출력 / (확장 시) early stop 등에 쓰는 자리.
    """
    # 초기 개체군: 랜덤 전략 pop_size 마리
    population = [_make(random.sample(ALL_GENES, random.randint(1, len(ALL_GENES))))
                 for _ in range(pop_size)]

    best, best_stats = None, None
    keep = max(2, pop_size // 2)        # 상위 절반을 부모로 (최소 2)

    for gen in range(1, generations + 1):
        # (1) 평가: 모든 개체의 적합도 측정
        scored = [(s, evaluate(s, gyms, trials)) for s in population]

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
