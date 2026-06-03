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
from pathlib import Path

from app.backend.battle import challenge
from app.backend.data import load_gyms
from app.backend.evolve import evolve
from app.backend.gym import all_gyms
from app.backend.dex import SIGNAL_CARDS
from app.backend.signals import ALL_GENES
from app.backend.strategy import create_strategy

# 스탯 표시 라벨 (이모지는 콘솔 인코딩 이슈 피하려 텍스트로)
_STAT_ROWS = [
    ("체력   HP    현금 비중", "hp"),
    ("공격력 ATK   연수익", "atk"),
    ("방어력 DEF   하락 방어", "def_"),
    ("솜씨   SKILL 샤프", "skill"),
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
        lines.append(f"  {label:<18} {value:5.1f}점  {bar}")
    return "\n".join(lines)


def _format_per_gym_bst(per_gym: dict) -> str:
    """체육관별 종족치(BST)를 '약한 순(=박살난 순)'으로 막대그래프 출력."""
    lines = []
    for name, bst in sorted(per_gym.items(), key=lambda x: x[1]):  # 약한 시장이 맨 위
        bar = "#" * round(bst / 400 * 20)                          # BST 400 = 20칸
        lines.append(f"  {name:<18} {bst:6.1f}점  {bar}")
    return "\n".join(lines)


def _bar(value: float, scale: float = 100.0, width: int = 20) -> str:
    """Markdown/콘솔에 같이 쓰는 간단한 막대그래프."""
    return "#" * round(value / scale * width)


def _resolve_md_path(path: str | None, mode: str) -> Path | None:
    """--md 인자를 실제 저장 경로로 바꾼다."""
    if path is None:
        return None
    if path == "":
        return Path("reports") / f"pocketquant_{mode}_report.md"
    return Path(path)


def _write_markdown(path: Path | None, content: str) -> None:
    """Markdown 리포트를 저장하고 콘솔에 위치를 알려준다."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"\nMarkdown 리포트 저장: {path}")


def _markdown_report(title: str, report) -> str:
    """Report 객체를 사람이 읽기 쉬운 Markdown으로 만든다."""
    strategy = report.strategy
    stats = report.stats
    lines = [
        f"# {title}",
        "",
        "## 전략",
        "",
        f"- 이름: {strategy.name}",
        f"- 유전자: {', '.join(strategy.genes)}",
        "",
        "## 시장별 백테스트",
        "",
        "| 시장 | 연수익 | 최대낙폭 | 시장낙폭 | 종족치 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in report.results:
        lines.append(
            f"| {r.gym_name} | {r.cagr * 100:.1f}% | "
            f"{r.max_drawdown * 100:.1f}% | {r.market_drawdown * 100:.1f}% | "
            f"{r.stats.bst:.1f} |"
        )

    lines.extend([
        "",
        "## 종합 스탯",
        "",
        "| 스탯 | 점수 | 막대 |",
        "|---|---:|---|",
        f"| HP 자본력 | {stats.hp:.1f} | `{_bar(stats.hp)}` |",
        f"| ATK 공격력 | {stats.atk:.1f} | `{_bar(stats.atk)}` |",
        f"| DEF 방어력 | {stats.def_:.1f} | `{_bar(stats.def_)}` |",
        f"| SKILL 솜씨 | {stats.skill:.1f} | `{_bar(stats.skill)}` |",
        "",
        "## 판정",
        "",
        f"- 종족치 합계: {report.bst:.1f} / 400",
        f"- 최종 적합도: {report.fitness:.1f} / 100",
        f"- 등급: {report.grade}",
        "",
    ])
    return "\n".join(lines)


def run_pokedex() -> None:
    """[도감] 전 유전자(포켓몬)의 설명 카드를 출력한다."""
    print("=== PocketQuant 유전자 도감 ===\n")
    for gene in ALL_GENES:                  # 실제 명단 순서대로
        c = SIGNAL_CARDS[gene]
        print(f"[{gene:<3}] {c['name']}   ({c['type']} · {c['role']})")
        print(f"      성격: {c['personality']}")
        print(f"      효과: {c['effect']}")
        print(f"      강점: {c['strength']}")
        print(f"      약점: {c['weakness']}\n")


def run_single(gene_count: int | None, seed: int | None = None,
               md_path: str | None = None) -> None:
    """[단판 모드] 전략 한 마리를 만들어 전 시장 백테스트하고 스탯을 출력."""
    _apply_seed(seed)
    print("=== PocketQuant 단판 백테스트 ===\n")

    print("1. 데이터 로딩: SPY 실데이터 4개 국면")
    loaded_gyms = load_gyms(all_gyms())

    print("\n2. 전략 생성")
    strategy = create_strategy(gene_count)
    print(f"  유전자: {' + '.join(strategy.genes)}")
    print(f"  이름: {strategy.name}\n")

    print("3. 시장별 백테스트 결과")
    report = challenge(strategy, loaded_gyms)
    for r in report.results:
        print(f"  {r.gym_name:<18} "
              f"연수익 {r.cagr * 100:6.1f}%  "
              f"최대낙폭 {r.max_drawdown * 100:6.1f}%  "
              f"시장낙폭 {r.market_drawdown * 100:6.1f}%  "
              f"종족치 {r.stats.bst:5.1f}점")

    print("\n4. 종합 스탯")
    print(_format_stats(report.stats))
    print(f"\n종족치 합계 {report.bst:.1f} / 400")
    print(f"최종 적합도 {report.fitness:.1f}점   등급 {report.grade}")

    path = _resolve_md_path(md_path, "single")
    _write_markdown(path, _markdown_report("PocketQuant 단판 백테스트 리포트", report))


def run_evolve(pop: int, generations: int, seed: int | None = None,
               md_path: str | None = None) -> None:
    """[진화 모드] 단일목적 GA(적합도=스탯 가중합)로 챔피언을 진화시킨다."""
    _apply_seed(seed)
    print("=== PocketQuant 진화 백테스트 ===")
    print(f"개체군 {pop}마리 · 진화 {generations}세대\n")

    print("1. 데이터 로딩: SPY 실데이터 4개 국면")
    loaded_gyms = load_gyms(all_gyms())
    print()

    # 세대마다 호출될 콜백: 진행상황을 한 줄씩 출력 (회사에서 쓰는 그 콜백 자리)
    def on_generation(gen, best, stats):
        genes = "+".join(best.genes)
        print(f"[{gen:2}세대] 최고 적합도 {stats['fitness']:5.1f}점  최강 전략: {genes}")

    best, stats = evolve(loaded_gyms, pop_size=pop, generations=generations,  # ⑤ 진화(②③ 반복)
                         on_generation=on_generation)

    # 최종 챔피언 + 스탯블록 + 시장별 강함                                       # ④ 결과
    print("\n=== 최종 챔피언 ===")
    print(f"유전자: {', '.join(best.genes)}")
    print(f"이름: {best.name}")
    print(f"최종 적합도: {stats['fitness']:.1f}점 / 100\n")
    print("종합 스탯")
    print(_format_stats(stats["stats"]))
    print("\n시장별 성적 (종족치 낮은 순 = 약한 시장):")
    print(_format_per_gym_bst(stats["per_gym"]))

    path = _resolve_md_path(md_path, "evolve")
    if path is not None:
        champion_report = challenge(best, loaded_gyms)
        _write_markdown(path, _markdown_report("PocketQuant 진화 백테스트 리포트", champion_report))
