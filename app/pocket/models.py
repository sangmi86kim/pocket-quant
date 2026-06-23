"""
models.py - 데이터의 '모양'을 정의하는 파일

여기서는 로직(계산)을 거의 하지 않습니다.
"전략은 어떻게 생겼나?", "스탯은 어떤 값을 갖나?" 처럼
프로그램에서 다루는 '명사(데이터)'의 설계도만 모아둔 곳입니다.

[현재 계약] 전략은 생존/탈락 이진판정이 아니라 스탯블록으로 표시한다.
  ❤️ HP   (자본력)  = 현금 비중      = 위기 때 버틸 체력 (표시 전용, 적합도 제외)
  ⚔️ ATK  (공격력)  = CAGR           = 돈 버는 능력
  🛡️ DEF  (방어력)  = Calmar         = 낙폭(MDD) 한 단위당 수익 = 위험조정 방어
  ✨ SKILL(솜씨)    = 샤프비율       = 같은 수익을 얼마나 효율적으로 냈나
  각 스탯은 0~100으로 정규화된다(포켓퀀트식).
"""
from dataclasses import dataclass, field


# ──────────────────────────────────────────────
# 유전자 명단은 여기에 없다. 진짜 출처는 signals.py(SIGNAL_REGISTRY / SIGNAL_NAMES)다.
# 옛날엔 여기 GENE_SCORES 라는 가짜 점수표가 명단을 정의했지만, 실데이터 도입 후
# '실제 시그널을 가진 유전자'만이 진짜 유전자라서 그 정의를 signals로 옮겼다.
# ──────────────────────────────────────────────
# 표시용 fitness = 스탯 가중합. 시즌3 선발 화폐는 raw 잔고/리그 결과이며,
# 이 값은 구 단일목적 리포트와 사람 읽기용 성격표시에만 남아 있다.
#
# [HP=0 인 이유 — 퇴화 방지] HP(현금 비중)를 적합도에 넣으면 '전부 현금'이
# HP 100 + DEF 만점을 받아 "아무것도 안 하기"가 최적해가 된다(실측: 전부 현금
# ~69점 vs 풀매수 ~29점). 현금 보유는 수단이지 목표가 아니므로 HP는 표시
# 전용 스탯으로 두고, 적합도는 ATK/DEF/SKILL 세 성과 스탯으로만 잰다.
# ──────────────────────────────────────────────
STAT_WEIGHTS = {"HP": 0.0, "ATK": 1.0, "DEF": 1.0, "SKILL": 1.0}

@dataclass
class Stats:
    """트레이더(전략)의 스탯블록. 각 값은 0~100으로 정규화됨."""
    hp: float = 0.0       # ❤️ 자본력 (현금 비중, 표시 전용 — 적합도 가중치 0)
    atk: float = 0.0      # ⚔️ 공격력 (CAGR)
    def_: float = 0.0     # 🛡️ 방어력 (Calmar = CAGR/|MDD|)  ※ def는 예약어라 def_
    skill: float = 0.0    # ✨ 솜씨 (샤프비율)

    @property
    def fitness(self) -> float:
        """표시용 스탯 가중평균 (0~100). 가중치는 STAT_WEIGHTS."""
        w = STAT_WEIGHTS
        total = w["HP"] + w["ATK"] + w["DEF"] + w["SKILL"]
        return (self.hp * w["HP"] + self.atk * w["ATK"]
                + self.def_ * w["DEF"] + self.skill * w["SKILL"]) / total


@dataclass
class Strategy:
    """트레이더(전략) 한 명을 표현하는 데이터 — 어떤 포켓퀀트(시그널)들을 데려가는가"""
    genes: list[str]      # 데려가는 포켓퀀트 목록. 예) ["DD", "REV_BB"]
    name: str = ""        # 트레이더 이름 (자동 생성)


@dataclass
class Gym:
    """
    체육관 하나 = 하나의 시장 국면(역사적 기간).
    실데이터 도입 후 ticker/start/end 로 그 기간 가격을 받아 백테스트한다.
    difficulty/volatility 는 이제 판정에 안 쓰는 '연출용 메타데이터'다.
    """
    name: str

    difficulty: int
    # 연출용 난이도 메타데이터.
    # 현재 백테스트 계산에는 사용하지 않음.
    # 향후 UI 표시, 체육관 설명, 시즌 난이도 분류에 사용 가능.

    volatility: int
    # 연출용 변동성 메타데이터.
    # 현재 백테스트 계산에는 사용하지 않음.
    # 향후 체육관 속성 표시 및 리그 분류에 사용 가능.

    ticker: str = "SPY"   # 어떤 자산으로 그 시기를 재현할지
    start: str = ""       # 평가 시작일 (YYYY-MM-DD)
    end: str = ""         # 평가 종료일 (YYYY-MM-DD)


@dataclass
class BattleResult:
    """한 체육관(시장 국면)에서 백테스트한 결과 = 그 시장에서의 스탯블록.

    [raw 지표] 0~100 스탯은 사람 읽기용(클램프됨). 최적화(NSGA-III)·DCA 비교는
    아래 원시값을 쓴다 — 코덱스 리뷰 4번(옵티마이저는 raw, 표시는 스탯) 반영.
    """
    gym_name: str
    stats: Stats                  # 그 시장에서 뽑힌 HP/ATK/DEF/SKILL
    cagr: float = 0.0             # 연율수익률 (원시값, 표시용)
    total_return: float = 0.0     # 기간 총수익률 (실투자 시뮬용: 시작자본 × (1+이값))
    market_return: float = 0.0    # 단순보유 기간 총수익률 (비교용)
    max_drawdown: float = 0.0     # 내 전략의 최대낙폭 (음수)
    market_drawdown: float = 0.0  # 시장(단순보유)의 최대낙폭 (음수, 비교용)
    sharpe: float = 0.0           # 샤프 raw (연율화. 출렁임 대비 수익 효율)
    turnover: float = 0.0         # 일평균 매매 비율 (포지션 변화량 |diff| 평균, 비용·NSGA-III 목적 재료)


@dataclass
class Report:
    """전체 도전 성적표. 체육관별 결과를 모아 종합 스탯블록을 만든다."""
    strategy: Strategy
    results: list[BattleResult] = field(default_factory=list)

    @property
    def fitness(self) -> float:
        """
        표시용 종합 점수 (0~100) = 체육관별 fitness의 [평균 70% + 최약 30%].

        [worst-case 반영 이유] 예전 단일 점수 리포트에서 평균만 쓰면 "평시장 점수로 위기장 낙제를 덮는"
        전략이 1등이 된다. 최약 체육관을 반영해 '어느 국면에서도 무너지지
        않는' 전략을 우대한다.

        [가중치 30%인 이유 — 실측] '전부 현금'은 모든 체육관에서 균일 16.7점
        (= 최약 국면이 없음)이라, min 가중치를 올릴수록 현금이 상대적으로 떠오른다.
        2026-06-11 스윕: min 40%부터 현금이 43/65위로 올라와 퇴화 게이트
        (tests/test_baselines.py, 하위 25% 룰) FAIL. 30%가 게이트를 지키는
        최대 worst-case 가중치(현금 64/65위). 상위권 순서는 0~50% 전 구간 동일.
        시즌3 선발/리그 판정은 이 값을 쓰지 않고 raw 잔고와 비용모델 계약을 쓴다.
        ※ 기준은 BST가 아니라 fitness(HP 가중치 0) — BST를 쓰면 HP(현금 비중)가
          뒷문으로 들어와 '아무것도 안 하기' 퇴화가 부활한다.
        """
        if not self.results:
            return 0.0
        per_gym = [r.stats.fitness for r in self.results]
        return 0.7 * (sum(per_gym) / len(per_gym)) + 0.3 * min(per_gym)
