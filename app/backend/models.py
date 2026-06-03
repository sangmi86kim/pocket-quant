"""
models.py - 데이터의 '모양'을 정의하는 파일

여기서는 로직(계산)을 거의 하지 않습니다.
"전략은 어떻게 생겼나?", "스탯은 어떤 값을 갖나?" 처럼
프로그램에서 다루는 '명사(데이터)'의 설계도만 모아둔 곳입니다.

[v0.3 변경] 생존/사망 이진판정을 버리고, 전략을 '스탯 포켓몬'으로 바꿨다.
  ❤️ HP   (자본력)  = 현금 비중      = 위기 때 버틸 체력
  ⚔️ ATK  (공격력)  = CAGR           = 돈 버는 능력
  🛡️ DEF  (방어력)  = 하락 방어율    = 시장 하락 대비 손실 방어
  ✨ SKILL(솜씨)    = 샤프비율       = 같은 수익을 얼마나 효율적으로 냈나
  각 스탯은 0~100으로 정규화된다(포켓몬식). 합 = 종족치(BST).
"""
from dataclasses import dataclass, field


# ──────────────────────────────────────────────
# 유전자 명단 (값은 레거시 점수 — 실데이터 도입 후 판정엔 안 쓴다)
# 진짜 점수는 signals.py가 가격으로 직접 계산한다.
# 여기 dict의 '키'들이 곧 사용 가능한 유전자 목록이다.
# ──────────────────────────────────────────────
GENE_SCORES = {
    "DD": 20,
    "RSI": 15,
    "MA": 25,
    "BB": 10,
    "FX": 5,
}

# 사용 가능한 모든 유전자 이름 -> ["DD", "RSI", "MA", "BB", "FX"]
ALL_GENES = list(GENE_SCORES.keys())

# ──────────────────────────────────────────────
# GA 적합도 = 스탯 가중합. 가중치를 바꾸면 진화 방향이 바뀐다.
#   예) 공격형 진화를 원하면 ATK 가중치를 올린다.
# ──────────────────────────────────────────────
STAT_WEIGHTS = {"HP": 1.0, "ATK": 1.0, "DEF": 1.0, "SKILL": 1.0}

# ──────────────────────────────────────────────
# 등급 테이블 (적합도 하한선 -> 등급). 적합도는 0~100 → 0~1로 환산해 비교.
# ──────────────────────────────────────────────
GRADES = [
    (0.9, "S"),
    (0.7, "A"),
    (0.5, "B"),
    (0.3, "C"),
    (0.0, "D"),
]


@dataclass
class Stats:
    """전략 포켓몬의 스탯블록. 각 값은 0~100으로 정규화됨."""
    hp: float = 0.0       # ❤️ 자본력 (현금 비중)
    atk: float = 0.0      # ⚔️ 공격력 (CAGR)
    def_: float = 0.0     # 🛡️ 방어력 (하락 방어율)  ※ def는 예약어라 def_
    skill: float = 0.0    # ✨ 솜씨 (샤프비율)

    @property
    def bst(self) -> float:
        """종족치(Base Stat Total) = 네 스탯의 단순 합 (0~400)."""
        return self.hp + self.atk + self.def_ + self.skill

    @property
    def fitness(self) -> float:
        """GA 적합도 = 스탯 가중평균 (0~100). 가중치는 STAT_WEIGHTS."""
        w = STAT_WEIGHTS
        total = w["HP"] + w["ATK"] + w["DEF"] + w["SKILL"]
        return (self.hp * w["HP"] + self.atk * w["ATK"]
                + self.def_ * w["DEF"] + self.skill * w["SKILL"]) / total


@dataclass
class Strategy:
    """전략 포켓몬 한 마리를 표현하는 데이터"""
    genes: list[str]      # 유전자 목록. 예) ["DD", "RSI"]
    name: str = ""        # 전략 이름 (자동 생성)

    def base_score(self) -> int:
        """[레거시] 유전자 점수 합. 실데이터 도입 후 판정엔 안 쓰임."""
        return sum(GENE_SCORES[g] for g in self.genes)


@dataclass
class Gym:
    """
    체육관 하나 = 하나의 시장 국면(역사적 기간).
    실데이터 도입 후 ticker/start/end 로 그 기간 가격을 받아 백테스트한다.
    difficulty/volatility 는 이제 판정에 안 쓰는 '연출용 메타데이터'다.
    """
    name: str
    difficulty: int       # (연출용) 생존 난이도 설명값
    volatility: int       # (연출용) 변동성 설명값
    ticker: str = "SPY"   # 어떤 자산으로 그 시기를 재현할지
    start: str = ""       # 평가 시작일 (YYYY-MM-DD)
    end: str = ""         # 평가 종료일 (YYYY-MM-DD)


@dataclass
class BattleResult:
    """한 체육관(시장 국면)에서 백테스트한 결과 = 그 시장에서의 스탯블록."""
    gym_name: str
    stats: Stats                  # 그 시장에서 뽑힌 HP/ATK/DEF/SKILL
    cagr: float = 0.0             # 연율수익률 (원시값, 표시용)
    total_return: float = 0.0     # 기간 총수익률 (원시값, 표시용)
    max_drawdown: float = 0.0     # 내 전략의 최대낙폭 (음수)
    market_drawdown: float = 0.0  # 시장(단순보유)의 최대낙폭 (음수, 비교용)


@dataclass
class Report:
    """전체 도전 성적표. 체육관별 결과를 모아 종합 스탯블록을 만든다."""
    strategy: Strategy
    results: list[BattleResult] = field(default_factory=list)

    @property
    def stats(self) -> Stats:
        """종합 스탯블록 = 체육관별 스탯의 평균."""
        if not self.results:
            return Stats()
        n = len(self.results)
        return Stats(
            hp=sum(r.stats.hp for r in self.results) / n,
            atk=sum(r.stats.atk for r in self.results) / n,
            def_=sum(r.stats.def_ for r in self.results) / n,
            skill=sum(r.stats.skill for r in self.results) / n,
        )

    @property
    def fitness(self) -> float:
        """종합 적합도 (0~100)."""
        return self.stats.fitness

    @property
    def bst(self) -> float:
        """종합 종족치 (0~400)."""
        return self.stats.bst

    @property
    def grade(self) -> str:
        """종합 적합도(0~100)를 0~1로 환산해 GRADES에서 등급을 정한다."""
        f = self.fitness / 100
        for threshold, grade in GRADES:
            if f >= threshold:
                return grade
        return "D"
