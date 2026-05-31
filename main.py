"""
main.py - 프로그램의 시작점(진입점), 게임 진행자 역할

여기서는 어려운 계산을 하지 않습니다. 다른 파일의 기능을 순서대로 불러 출력만 합니다.

두 가지 모드:
  1) 단판 모드 (기본)   : 전략 1마리 만들어 전 체육관 도전        -> run_single()
  2) 진화 모드 (--evolve): 전략 여러 마리를 GA로 여러 세대 진화   -> run_evolve()
"""
import argparse
import random

from app.backend.battle import challenge
from app.backend.evolve import evolve
from app.backend.gym import all_gyms
from app.backend.strategy import create_strategy


def run_single(gene_count: int | None) -> None:
    """[단판 모드] 전략 한 마리를 만들어 전 체육관에 도전하고 결과 출력."""
    print("=== PocketQuant ===\n")

    print("전략 생성")
    strategy = create_strategy(gene_count)
    print(" + ".join(strategy.genes))
    print(f"이름: {strategy.name}\n")

    print("체육관 도전")
    report = challenge(strategy, all_gyms())
    for r in report.results:
        print(f"\n{r.gym_name}")
        print("생존" if r.survived else "사망")

    print("\n결과")
    print(f"생존 {report.survive_count}")
    print(f"사망 {report.death_count}")
    print(f"등급 {report.grade}")


def _format_per_gym(per_gym: dict) -> str:
    """체육관별 생존률을 '생존률 낮은 순(=박살난 순)'으로 막대그래프 출력."""
    lines = []
    # rate 오름차순 정렬 -> 가장 처참하게 죽은 시장이 맨 위로
    for name, rate in sorted(per_gym.items(), key=lambda x: x[1]):
        bar = "#" * round(rate * 20)          # 생존률 100% = '#' 20칸
        lines.append(f"    {name:<18} {rate * 100:5.1f}%  {bar}")
    return "\n".join(lines)


def run_evolve(pop: int, generations: int, trials: int) -> None:
    """[진화 모드] 단일목적 GA로 개체군을 여러 세대 진화시키고 챔피언 출력."""
    print("=== PocketQuant · 진화 모드 (단일목적 GA) ===")
    print(f"개체군 {pop} · 세대 {generations} · 시도 {trials}\n")

    # 세대마다 호출될 콜백: 진행상황을 한 줄씩 출력 (회사에서 쓰는 그 콜백 자리)
    def on_generation(gen, best, stats):
        genes = "+".join(best.genes)
        print(f"[세대 {gen:2}] 최고적합도 {stats['fitness']:.2f}  최강: {genes}")

    best, stats = evolve(all_gyms(), pop_size=pop, generations=generations,
                         trials=trials, on_generation=on_generation)

    # 최종 챔피언 + '시장별 박살 현황'
    print("\n=== 최종 챔피언 ===")
    print(f"유전자: {', '.join(best.genes)}")
    print(f"이름: {best.name}")
    print(f"적합도(평균 생존률): {stats['fitness'] * 100:.1f}%")
    print("\n시장별 박살 현황 (생존률 낮은 순):")
    print(_format_per_gym(stats["per_gym"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="PocketQuant - 전략 포켓몬 생존 테스트")
    parser.add_argument("-g", "--genes", type=int, default=None,
                        help="[단판] 유전자 개수 (생략 시 랜덤)")
    parser.add_argument("--evolve", action="store_true",
                        help="진화 모드(단일목적 GA) 실행")
    parser.add_argument("--pop", type=int, default=20, help="[진화] 개체군 크기")
    parser.add_argument("--generations", type=int, default=10, help="[진화] 세대 수")
    parser.add_argument("--trials", type=int, default=20,
                        help="[진화] 전략당 평가 도전 횟수(평균낼 표본)")
    parser.add_argument("--seed", type=int, default=None,
                        help="랜덤 시드 고정 (재현 가능 -> GA 검증용)")
    args = parser.parse_args()

    # 시드 고정 시 매번 같은 결과 -> GA가 제대로 도는지 재현 검증할 때 필수
    if args.seed is not None:
        random.seed(args.seed)

    if args.evolve:
        run_evolve(args.pop, args.generations, args.trials)
    else:
        run_single(args.genes)


if __name__ == "__main__":
    main()
