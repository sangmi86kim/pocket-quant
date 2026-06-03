"""
service.py - 실행 '흐름'을 조립하는 층 (애플리케이션 서비스)

3층 구조에서 가운데를 맡는다:
  main.py    = CLI 입력만 받음 (argparse → 어떤 서비스를 부를지 결정)
  service.py = 단판/진화 '실행 순서'를 조립 (← 이 파일)
  backend/*  = 실제 기능(데이터 로딩·전략·백테스트·GA·계산)

여기서는 어려운 계산을 하지 않는다. backend 기능을 '순서대로' 불러
파이프라인(① 데이터 → ② 전략 → ③ 전투 → ④ 결과 → ⑤ 진화)을 엮고 출력만 한다.
"""
import random

from app.backend.battle import challenge
from app.backend.data import load_gyms
from app.backend.evolve import evolve
from app.backend.gym import all_gyms
from app.backend.dex import SIGNAL_CARDS
from app.backend.signals import ALL_GENES
from app.backend.strategy import create_strategy

# 스탯 표시 라벨 (이모지는 콘솔 인코딩 이슈 피하려 텍스트로)
_STAT_ROWS = [
    ("HP   (자본력)", "hp"),
    ("ATK  (공격력)", "atk"),
    ("DEF  (방어력)", "def_"),
    ("SKILL(솜씨)  ", "skill"),
]


def _apply_seed(seed: int | None) -> None:
    """시드 고정 시 GA(초기 개체군/교배/돌연변이)가 매번 같게 재현된다."""
    if seed is not None:
        random.seed(seed)


def _format_stats(stats) -> str:
    """스탯블록을 막대그래프로 출력 (각 스탯 0~100 → '#' 20칸)."""
    lines = []
    for label, attr in _STAT_ROWS:
        value = getattr(stats, attr)
        bar = "#" * round(value / 100 * 20)
        lines.append(f"    {label}  {value:5.1f}  {bar}")
    return "\n".join(lines)


def _format_per_gym_bst(per_gym: dict) -> str:
    """체육관별 종족치(BST)를 '약한 순(=박살난 순)'으로 막대그래프 출력."""
    lines = []
    for name, bst in sorted(per_gym.items(), key=lambda x: x[1]):  # 약한 시장이 맨 위
        bar = "#" * round(bst / 400 * 20)                          # BST 400 = 20칸
        lines.append(f"    {name:<18} {bst:6.1f}  {bar}")
    return "\n".join(lines)


def run_pokedex() -> None:
    """[도감] 전 유전자(포켓몬)의 설명 카드를 출력한다."""
    print("=== PocketQuant 도감 ===\n")
    for gene in ALL_GENES:                  # 실제 명단 순서대로
        c = SIGNAL_CARDS[gene]
        print(f"[{gene:<3}] {c['name']}   ({c['type']} · {c['role']})")
        print(f"      성격: {c['personality']}")
        print(f"      효과: {c['effect']}")
        print(f"      강점: {c['strength']}")
        print(f"      약점: {c['weakness']}\n")


def run_single(gene_count: int | None, seed: int | None = None) -> None:
    """[단판 모드] 전략 한 마리를 만들어 전 시장 백테스트하고 스탯을 출력."""
    _apply_seed(seed)
    print("=== PocketQuant ===\n")

    print("데이터 로딩 (SPY · 4개 국면)")        # ① 데이터 땡겨오고
    loaded_gyms = load_gyms(all_gyms())

    print("\n전략 생성")                           # ② 전략 만들어
    strategy = create_strategy(gene_count)
    print(" + ".join(strategy.genes))
    print(f"이름: {strategy.name}\n")

    print("시장별 백테스트 (실데이터)")            # ③ 싸워
    report = challenge(strategy, loaded_gyms)
    for r in report.results:
        print(f"  {r.gym_name:<18} "
              f"CAGR {r.cagr * 100:6.1f}%  "
              f"MDD {r.max_drawdown * 100:6.1f}% (시장 {r.market_drawdown * 100:6.1f}%)  "
              f"BST {r.stats.bst:5.1f}")

    print("\n스탯블록 (종합)")                      # ④ 결과
    print(_format_stats(report.stats))
    print(f"\n종족치(BST) {report.bst:.1f} / 400   적합도 {report.fitness:.1f}   등급 {report.grade}")


def run_evolve(pop: int, generations: int, seed: int | None = None) -> None:
    """[진화 모드] 단일목적 GA(적합도=스탯 가중합)로 챔피언을 진화시킨다."""
    _apply_seed(seed)
    print("=== PocketQuant · 진화 모드 (단일목적 GA · 스탯 BST) ===")
    print(f"개체군 {pop} · 세대 {generations}\n")

    print("데이터 로딩 (SPY · 4개 국면)")        # ① 데이터 먼저 (세대 내내 재사용)
    loaded_gyms = load_gyms(all_gyms())
    print()

    # 세대마다 호출될 콜백: 진행상황을 한 줄씩 출력 (회사에서 쓰는 그 콜백 자리)
    def on_generation(gen, best, stats):
        genes = "+".join(best.genes)
        print(f"[세대 {gen:2}] 최고적합도 {stats['fitness']:5.1f}  최강: {genes}")

    best, stats = evolve(loaded_gyms, pop_size=pop, generations=generations,  # ⑤ 진화(②③ 반복)
                         on_generation=on_generation)

    # 최종 챔피언 + 스탯블록 + 시장별 강함                                       # ④ 결과
    print("\n=== 최종 챔피언 ===")
    print(f"유전자: {', '.join(best.genes)}")
    print(f"이름: {best.name}")
    print(f"적합도(스탯 가중평균): {stats['fitness']:.1f} / 100\n")
    print("스탯블록")
    print(_format_stats(stats["stats"]))
    print("\n시장별 강함 (종족치 낮은 순 = 약한 시장):")
    print(_format_per_gym_bst(stats["per_gym"]))
