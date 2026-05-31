"""
models.py - 데이터의 '모양'을 정의하는 파일

여기서는 로직(계산)을 거의 하지 않습니다.
"전략은 어떻게 생겼나?", "체육관은 어떤 정보를 갖나?" 처럼
프로그램에서 다루는 '명사(데이터)'의 설계도만 모아둔 곳입니다.

핵심 도구 두 가지:
  - @dataclass : 데이터를 담는 클래스를 아주 간단하게 만들어주는 파이썬 기능
  - @property  : '함수처럼 계산하지만 변수처럼 꺼내 쓰는' 값
"""
from dataclasses import dataclass, field


# ──────────────────────────────────────────────
# 유전자별 점수 보정값
# "이 유전자를 가지면 점수를 몇 점 더 받나?"를 dict(사전)으로 정리
# 예) DD 유전자가 있으면 +20점
# ──────────────────────────────────────────────
GENE_SCORES = {
    "DD": 20,
    "RSI": 15,
    "MA": 25,
    "BB": 10,
    "FX": 5,
}

# 사용 가능한 모든 유전자 이름 목록 -> ["DD", "RSI", "MA", "BB", "FX"]
# GENE_SCORES의 '열쇠(key)'들만 뽑아서 리스트로 만든 것
ALL_GENES = list(GENE_SCORES.keys())

# ──────────────────────────────────────────────
# 등급 테이블 (생존률 하한선 -> 등급 글자)
# 위에서부터 검사해서, 생존률이 기준 이상이면 그 등급을 줍니다.
# 예) 생존률 0.95 -> 0.9 이상이므로 "S"
#     생존률 0.6  -> 0.5 이상이므로 "B"
# ──────────────────────────────────────────────
GRADES = [
    (0.9, "S"),
    (0.7, "A"),
    (0.5, "B"),
    (0.3, "C"),
    (0.0, "D"),
]


# ==============================================================
# @dataclass 란?
#   원래 클래스를 만들려면 __init__ 같은 걸 직접 써야 하는데,
#   @dataclass를 붙이면 "필드만 적어두면" 생성자를 자동으로 만들어줍니다.
#   즉, 아래 Strategy는 Strategy(genes=["DD"], name="DD몬") 처럼 바로 만들 수 있어요.
# ==============================================================

@dataclass
class Strategy:
    """전략 포켓몬 한 마리를 표현하는 데이터"""
    genes: list[str]      # 유전자 목록. 예) ["DD", "RSI"]  ('list[str]' = 문자열들의 리스트)
    name: str = ""        # 전략 이름. '= ""'는 기본값(안 넣으면 빈 문자열)

    def base_score(self) -> int:
        """
        이 전략이 가진 유전자 점수를 전부 더한 '기본 점수'를 돌려준다.
        sum(...) : 괄호 안 값들을 모두 더하는 함수
        예) 유전자가 ["DD", "RSI"]면 -> 20 + 15 = 35
        """
        return sum(GENE_SCORES[g] for g in self.genes)


@dataclass
class Gym:
    """체육관 하나를 표현하는 데이터. 변수 3개만 가진다(이름/난이도/변동성)."""
    name: str             # 체육관 이름. 예) "DOTCOM"
    difficulty: int       # 난이도(이 점수 이상이어야 생존)
    volatility: int       # 변동성(지금은 연출용 정보, 판정에는 아직 안 씀)


@dataclass
class BattleResult:
    """한 체육관에서 싸운 결과(전투 1회분)를 표현하는 데이터."""
    gym_name: str         # 어떤 체육관이었나
    score: int            # 그때 나온 최종 점수
    survived: bool        # 살아남았나? (True=생존, False=사망)


@dataclass
class Report:
    """전체 도전이 끝난 뒤의 '성적표'. 전투 결과 여러 개를 모아 요약한다."""
    strategy: Strategy
    # field(default_factory=list)
    #   = "기본값은 빈 리스트로 시작해라"는 뜻.
    #   리스트 같은 건 기본값을 그냥 [] 로 적으면 안 되고 이렇게 적어야 안전합니다(파이썬 규칙).
    results: list[BattleResult] = field(default_factory=list)

    # ── 아래 @property들은 '계산해서 꺼내 쓰는 값' ──
    #   report.survive_count 처럼 괄호() 없이 변수처럼 사용합니다.

    @property
    def survive_count(self) -> int:
        """생존한 전투 횟수를 센다."""
        # results 안에서 survived가 True인 것마다 1을 더함 -> 생존 횟수
        return sum(1 for r in self.results if r.survived)

    @property
    def death_count(self) -> int:
        """사망한 전투 횟수를 센다 (survived가 False인 것)."""
        return sum(1 for r in self.results if not r.survived)

    @property
    def survive_rate(self) -> float:
        """생존률 = 생존 횟수 / 전체 횟수 (0.0 ~ 1.0 사이 소수)."""
        if not self.results:        # 도전 기록이 하나도 없으면(빈 리스트면)
            return 0.0              # 0으로 나누는 오류를 피하려고 0.0 반환
        return self.survive_count / len(self.results)

    @property
    def grade(self) -> str:
        """생존률을 보고 위 GRADES 표에서 등급(S~D)을 정한다."""
        for threshold, grade in GRADES:          # (기준선, 등급)을 위에서부터 하나씩
            if self.survive_rate >= threshold:   # 생존률이 기준선 이상이면
                return grade                     # 그 등급을 돌려주고 끝
        return "D"                               # (안전장치) 어디에도 안 걸리면 D
