"""
battle.py - 전투(백테스트)를 '계산'하는 파일, 게임의 엔진

흐름:
  fight()     : 전략 1마리 vs 체육관 1곳  -> 한 판 결과(BattleResult)
  challenge() : 전략 1마리 vs 체육관 여러 곳 -> 전체 성적표(Report)
"""
import random

from .models import BattleResult, Gym, Report, Strategy

# 랜덤 보정 범위. 매 전투마다 -20 ~ +20 사이 운(運)이 점수에 더해진다.
# 같은 전략이라도 결과가 매번 조금씩 달라지는 이유가 바로 이것.
RANDOM_SWING = 20


def fight(strategy: Strategy, gym: Gym) -> BattleResult:
    """전략 한 마리가 체육관 한 곳에 도전하는 '한 판'을 계산한다."""
    # randint(a, b) : a 이상 b 이하 정수 중 랜덤 하나. 여기선 -20 ~ +20
    bonus = random.randint(-RANDOM_SWING, RANDOM_SWING)

    # 최종 점수 = 유전자 기본 점수 합 + 랜덤 보정
    # strategy.base_score()는 models.py의 Strategy에 정의돼 있음
    score = strategy.base_score() + bonus

    # 판정: 최종 점수가 체육관 난이도 이상이면 생존(True), 아니면 사망(False)
    survived = score >= gym.difficulty

    # 이 한 판의 결과를 데이터로 묶어서 돌려준다
    return BattleResult(gym_name=gym.name, score=score, survived=survived)


def challenge(strategy: Strategy, gyms: list[Gym]) -> Report:
    """전략 한 마리가 여러 체육관에 '차례대로' 도전하고 성적표를 만든다."""
    # 빈 성적표를 먼저 만든다(이 전략의 결과를 담을 그릇)
    report = Report(strategy=strategy)

    # 체육관을 하나씩 돌면서
    for gym in gyms:
        # 한 판 싸우고(fight), 그 결과를 성적표의 results 리스트에 추가
        report.results.append(fight(strategy, gym))

    # 모든 도전이 끝난 성적표를 돌려준다
    # (생존 수/사망 수/등급 등은 Report 안에서 자동 계산됨)
    return report
