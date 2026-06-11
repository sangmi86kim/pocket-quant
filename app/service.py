"""
service.py - 실행 '흐름'을 조립하는 층 (애플리케이션 서비스)

3층 구조에서 가운데를 맡는다:
  main.py    = config.json 읽어 어떤 서비스를 부를지 결정 (CLI 플래그 없음)
  service.py = 단판/진화/다목적 '실행 순서'를 조립 (← 이 파일)
  backend/*  = 실제 기능(데이터 로딩·전략·백테스트·GA·NSGA-III·계산)

여기서는 어려운 계산을 하지 않는다. backend 기능을 '순서대로' 불러
파이프라인(① 데이터 → ② 전략 → ③ 전투 → ④ 결과 → ⑤ 진화)을 엮고 출력만 한다.
"""
import random
from pathlib import Path

from app.backend.engine.battle import challenge
from app.backend.engine.evolve import evolve
from app.backend.engine.strategy import create_strategy
from app.backend.genes.dex import GYM_LEADERS, NPC_CARDS, SIGNAL_CARDS
from app.backend.genes.signals import ALL_GENES
from app.backend.market.data import load_gyms
from app.backend.market.gym import all_gyms

# 스탯 표시 라벨 (이모지는 콘솔 인코딩 이슈 피하려 텍스트로)
# HP는 표시 전용(적합도 가중치 0), DEF는 Calmar(낙폭 대비 수익) 기반.
_STAT_ROWS = [
    ("체력   HP    현금 비중", "hp"),
    ("공격력 ATK   연수익", "atk"),
    ("방어력 DEF   낙폭대비수익", "def_"),
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


def _format_simulation(report, capital: float) -> str:
    """실투자 시뮬레이션: 각 국면에 시작 자본을 '따로' 넣었다고 가정한 최종 잔고.
    (체육관들은 서로 다른 시대라 한 번에 굴릴 수 없어 국면별 독립 시뮬이다.)"""
    lines = [f"  (각 국면에 {capital:,.0f}원씩 따로 투자했다고 가정)"]
    for r in report.results:
        final = capital * (1.0 + r.total_return)      # 전략 최종 잔고
        hold = capital * (1.0 + r.market_return)      # 단순보유 최종 잔고
        profit = final - capital
        sign = "+" if profit >= 0 else "-"
        lines.append(
            f"  {r.gym_name:<18} "
            f"전략 {final:>13,.0f}원 ({sign}{abs(profit):,.0f}, {r.total_return * 100:+.1f}%)"
            f"   단순보유 {hold:>13,.0f}원"
        )
    return "\n".join(lines)


def _bar(value: float, scale: float = 100.0, width: int = 20) -> str:
    """Markdown/콘솔에 같이 쓰는 간단한 막대그래프."""
    return "#" * round(value / scale * width)


def _grade_score(score: float) -> str:
    """0~100 점수를 포켓퀀트식 등급으로 바꾼다."""
    if score >= 90:
        return "S"
    if score >= 70:
        return "A"
    if score >= 50:
        return "B"
    if score >= 30:
        return "C"
    return "D"


def _style_profile(stats) -> dict:
    """단일 등급이 놓치는 전략 성격을 보조 판정으로 분해한다."""
    profile = {
        "체력": stats.hp,
        "공격": stats.atk,
        "방어": stats.def_,
        "효율": stats.skill,
        "종합": stats.fitness,
    }
    best_role = max(("체력", "공격", "방어", "효율"), key=lambda key: profile[key])
    style_names = {
        "체력": "현금 방어형",
        "공격": "상승장 공격형",
        "방어": "낙폭 방어형",
        "효율": "위험 대비 효율형",
    }
    profile["스타일"] = style_names[best_role]
    return profile


def _format_profile(stats) -> str:
    """전략 스타일과 보조 등급을 콘솔용으로 출력한다."""
    profile = _style_profile(stats)
    rows = [
        ("종합", profile["종합"], "ATK/DEF/SKILL 평균"),
        ("공격", profile["공격"], "연수익"),
        ("방어", profile["방어"], "낙폭 대비 수익(Calmar)"),
        ("체력", profile["체력"], "현금 비중 · 표시 전용"),
        ("효율", profile["효율"], "샤프"),
    ]
    lines = [f"  스타일: {profile['스타일']}"]
    for name, score, note in rows:
        lines.append(f"  {name:<4} {_grade_score(score)}등급  {score:5.1f}점  ({note})")
    return "\n".join(lines)


def _format_objective_vector(values: list[float]) -> str:
    """[NSGA-III] 목적 벡터를 한 줄로 (점수 ×100 표기, 턴오버는 원값)."""
    names = ["bear", "rebound", "crash_v", "bull", "chop"]
    scores = " ".join(f"{n} {v * 100:+6.1f}" for n, v in zip(names, values[:5]))
    return f"{scores}  | turnover {values[5]:.4f}/일"


def _format_candidate_params(params: dict) -> str:
    """[NSGA-III] 트레이더의 X를 사람 읽는 형태로 — 가중치는 비율(%)로 정규화."""
    w = [params[f"w_{g}"] for g in ALL_GENES]
    total = sum(w) or 1.0
    weights = " ".join(f"{g} {x / total * 100:.0f}%" for g, x in zip(ALL_GENES, w))
    if "DD_LIMIT" not in params:                 # 가중치 전용 리그 (v2, A안)
        return f"가중치: {weights}\n  파라미터: 기본값 고정"
    tunables = (f"DD {params['DD_LIMIT']:.2f} · MA {params['MA_WINDOW']} · "
                f"MOM {params['MOM_LOOKBACK']} · RSI<{params['RSI_OVERSOLD']} · "
                f"BB k{params['BB_K']:.2f} · VOL {params['VOL_CALM']:.3f}"
                f"~{params['VOL_CALM'] + params['VOL_SPREAD']:.3f}")
    return f"가중치: {weights}\n  파라미터: {tunables}"


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
        "| 시장 | 연수익 | 전략 최대낙폭 | 시장 최대낙폭 | 종족치 |",
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
        "## 전략 판정",
        "",
        f"- 스타일: {_style_profile(stats)['스타일']}",
        f"- 종족치 합계: {report.bst:.1f} / 400",
        f"- 최종 적합도: {report.fitness:.1f} / 100 (평균 70% + 최약 체육관 30%)",
        f"- 최약 체육관: {report.weakest_gym[0]} ({report.weakest_gym[1]:.1f}점)",
        f"- 등급: {report.grade}",
        "",
        "| 관점 | 등급 | 점수 | 의미 |",
        "|---|---:|---:|---|",
        f"| 종합 | {_grade_score(stats.fitness)} | {stats.fitness:.1f} | ATK/DEF/SKILL 평균 |",
        f"| 공격 | {_grade_score(stats.atk)} | {stats.atk:.1f} | 연수익 |",
        f"| 방어 | {_grade_score(stats.def_)} | {stats.def_:.1f} | 낙폭 대비 수익(Calmar) |",
        f"| 체력 | {_grade_score(stats.hp)} | {stats.hp:.1f} | 현금 비중 (표시 전용) |",
        f"| 효율 | {_grade_score(stats.skill)} | {stats.skill:.1f} | 샤프 |",
        "",
    ])
    return "\n".join(lines)


def run_pokedex() -> None:
    """[도감] 포켓퀀트(시그널) 카드 + 체육관 관장 + NPC를 출력한다."""
    print("=== PocketQuant 도감 ===\n")

    print("─── 스타팅 6마리 (시그널) ───\n")
    for gene in ALL_GENES:                  # 실제 명단 순서대로
        c = SIGNAL_CARDS[gene]
        print(f"[{gene:<3}] {c['name']}   ({c['type']} · {c['role']})")
        print(f"      성격: {c['personality']}")
        print(f"      효과: {c['effect']}")
        print(f"      강점: {c['strength']}")
        print(f"      약점: {c['weakness']}\n")

    print("─── 체육관 관장 6명 (시장 국면) ───\n")
    for gym_name, leader in GYM_LEADERS.items():
        print(f"[{leader['name']}] {gym_name}   ({leader['type']} 타입)")
        print(f"      \"{leader['quote']}\"")
        print(f"      주특기: {leader['specialty']}")
        print(f"      공략법: {leader['counter']}\n")

    print("─── NPC ───\n")
    for name, c in NPC_CARDS.items():
        print(f"[{name}] {c['role']}")
        print(f"      성격: {c['personality']}")
        print(f"      무서운 점: {c['scary']}")
        print(f"      전적: {c['record']}\n")


def _oak_briefing(report) -> None:
    """오박사 브리핑 출력 (LLM 해설 — 판정 아님). 연구소 부재 시 조용히 안내만."""
    from app.oak import professor_briefing   # LM Studio 없을 때도 import 비용 없게 지연
    print("\n=== 🔬 오박사 브리핑 (해설 전용 — 판정 아님) ===")
    text = professor_briefing(report)
    print(text if text else
          "  오박사가 연구소에 안 계십니다. (LM Studio가 켜져 있는지 확인: localhost:1234)")


def run_single(gene_count: int | None, seed: int | None = None,
               md_path: str | None = None, capital: float | None = None,
               oak: bool = False) -> None:
    """[단판 모드] 트레이더 한 명을 만들어 전 시장 백테스트하고 스탯을 출력."""
    _apply_seed(seed)
    print("=== PocketQuant 단판 백테스트 ===\n")

    gyms = all_gyms()
    print(f"1. 데이터 로딩: QQQ 실데이터 {len(gyms)}개 국면")   # 체육관 수는 gym.py가 정함
    loaded_gyms = load_gyms(gyms)

    print("\n2. 전략 생성")
    strategy = create_strategy(gene_count)
    print(f"  유전자: {' + '.join(strategy.genes)}")
    print(f"  이름: {strategy.name}\n")

    print("3. 시장별 백테스트 결과")
    report = challenge(strategy, loaded_gyms)
    for r in report.results:
        print(f"  {r.gym_name:<18} "
              f"연수익 {r.cagr * 100:6.1f}%  "
              f"전략 최대낙폭 {r.max_drawdown * 100:6.1f}%  "
              f"시장 최대낙폭 {r.market_drawdown * 100:6.1f}%  "
              f"종족치 {r.stats.bst:5.1f}점")

    print("\n4. 종합 스탯")
    print(_format_stats(report.stats))
    print(f"\n종족치 합계 {report.bst:.1f} / 400")
    weak_name, weak_fit = report.weakest_gym
    print(f"최종 적합도 {report.fitness:.1f}점 (평균 70% + 최약 30%)   등급 {report.grade}")
    print(f"최약 체육관: {weak_name} ({weak_fit:.1f}점)")
    print("\n5. 전략 판정")
    print(_format_profile(report.stats))

    if capital is not None:
        print(f"\n6. 실전 시뮬레이션 (시작 자본 {capital:,.0f}원 · 국면별 독립)")
        print(_format_simulation(report, capital))

    if oak:
        _oak_briefing(report)

    path = _resolve_md_path(md_path, "single")
    _write_markdown(path, _markdown_report("PocketQuant 단판 백테스트 리포트", report))


def run_nsga3(trials: int, seed: int | None = None,
              storage: str | None = None, study_name: str = "nsga3_v2_weights",
              tune_params: bool = False) -> None:
    """[다목적 모드] Optuna NSGA-III — 국면별 라이벌(DCA)전 5목적 + 턴오버.
    결과는 챔피언 1명이 아니라 Pareto front(트레이더 라인업)다.
    기본 = 가중치 전용 탐색(A안). tune_params=True는 고도화 단계용."""
    from app.backend.engine import nsga3   # optuna는 이 모드에서만 필요 — 지연 import

    space = "가중치 6 + 파라미터 7" if tune_params else "가중치 6 (파라미터 기본값 고정)"
    print("=== PocketQuant NSGA-III 다목적 최적화 ===")
    print(f"트라이얼 {trials} · 목적 {nsga3.OBJECTIVE_NAMES} · X = {space} · 시드 {seed}\n")

    def on_progress(done, total, front_size):
        print(f"  [{done:>5}/{total}] Pareto front {front_size}개")

    study, loaded_gyms, dca = nsga3.run_study(
        trials, seed=seed, storage=storage, study_name=study_name,
        tune_params=tune_params, on_progress=on_progress)

    # 비교 기준: 현 단일목적 챔피언 (동일가중 VOL+REV_RSI+REV_BB)
    ref = nsga3.reference_vector(loaded_gyms, dca)
    print("\n=== 기준점: 현 챔피언 VOL+REV_RSI+REV_BB (동일가중·기본 파라미터) ===")
    print("  " + _format_objective_vector(
        [min(ref["dotcom"], ref["gfc"]), ref["rebound"], ref["crash_v"],
         ref["bull"], ref["chop"], ref["turnover"]]))

    summary = nsga3.summarize_front(study)
    print(f"\n=== Pareto front {summary['front_size']}개 → 하드 필터 통과 "
          f"{len(summary['passed'])}개 (전 국면 ≥ -{summary['tolerance'] * 100:.0f}, "
          f"턴오버 ≤ {summary['turnover_cap']}) ===")

    for label, row in summary["labels"].items():
        print(f"\n[{label}]  trial #{row['number']}")
        print("  " + _format_objective_vector(row["values"]))
        print("  " + _format_candidate_params(row["params"]))

    if not summary["labels"]:
        print("\n  ⚠️ 필터 통과 후보 없음 — tolerance/turnover_cap을 조정해 다시 보세요.")


def run_evolve(pop: int, generations: int, seed: int | None = None,
               md_path: str | None = None, capital: float | None = None,
               oak: bool = False) -> None:
    """[진화 모드] 단일목적 GA(적합도=스탯 가중합)로 챔피언을 진화시킨다."""
    _apply_seed(seed)
    print("=== PocketQuant 진화 백테스트 ===")
    print(f"개체군 트레이더 {pop}명 · 진화 {generations}세대\n")

    gyms = all_gyms()
    print(f"1. 데이터 로딩: QQQ 실데이터 {len(gyms)}개 국면")   # 체육관 수는 gym.py가 정함
    loaded_gyms = load_gyms(gyms)
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
    weak_name, weak_fit = stats["weakest"]
    print(f"최종 적합도: {stats['fitness']:.1f}점 / 100 (평균 70% + 최약 30%)")
    print(f"최약 체육관: {weak_name} ({weak_fit:.1f}점)\n")
    print("종합 스탯")
    print(_format_stats(stats["stats"]))
    print("\n시장별 성적 (종족치 낮은 순 = 약한 시장):")
    print(_format_per_gym_bst(stats["per_gym"]))
    print("\n전략 판정")
    print(_format_profile(stats["stats"]))

    # 챔피언 리포트는 시뮬/마크다운/오박사 중 하나라도 필요하면 한 번만 계산
    path = _resolve_md_path(md_path, "evolve")
    if capital is not None or path is not None or oak:
        champion_report = challenge(best, loaded_gyms)
        if capital is not None:
            print(f"\n실전 시뮬레이션 (시작 자본 {capital:,.0f}원 · 국면별 독립)")
            print(_format_simulation(champion_report, capital))
        if oak:
            _oak_briefing(champion_report)
        _write_markdown(path, _markdown_report("PocketQuant 진화 백테스트 리포트", champion_report))
