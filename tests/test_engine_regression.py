"""
test_engine_regression.py - 골든 넘버 회귀 테스트 ("엔진이 안 깨졌나" 자동 검증)

[배경]
기존 tests/ 는 전부 "전략이 좋은가"를 재는 도구였고, "채점 엔진 자체가
리팩토링/신규 코드에 깨지지 않았나"를 재는 게 없었다. 그래서 battle.py를
고칠 때마다 test_baselines를 다시 돌려 눈으로 숫자를 비교해야 했다.
이 테스트는 알려진 정답(골든 넘버)을 박아두고 자동으로 비교한다.

[골든 넘버의 출처]
2026-06-11 엔진(커밋 c26b0d1 시점: 5체육관 · 적합도 70/30 · 비용 0.1% ·
DCA 기준선 무비용)에서 풀 정밀도로 실측한 값. 캐시된 가격(data_cache/)을
쓰므로 오프라인에서도 돈다.

[숫자가 달라졌다면]
  - 의도한 설계 변경(체육관/가중치/비용 변경 등) → 여기 골든 넘버를 갱신하고
    워크로그에 "왜 바뀌었는지"를 기록한다.
  - 의도하지 않았다 → 버그. 커밋 전에 원인을 찾는다.

실행: 프로젝트 루트에서  python tests/test_engine_regression.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backend.core.models import Strategy
from app.backend.engine.battle import challenge, fight_dca, score_vs_dca
from app.backend.market.data import load_gyms
from app.backend.market.gym import all_gyms

# 부동소수 노이즈(라이브러리 버전 차)만 허용 — 로직 변화는 이보다 훨씬 크게 어긋난다.
REL_TOL = 1e-6

# ── 골든 넘버 (2026-06-11 실측) ───────────────────────
GOLDEN_FITNESS = {
    "REV_BB": 40.288399500291796,                      # 현 챔피언
    "VOL+REV_BB": 35.83755524112411,                   # 2위 (이전 챔피언)
    "DD+VOL+MA+MOM+REV_RSI+REV_BB": 20.599916730667257,  # 전 유전자 합체
}

# REV_BB의 닷컴 체육관 raw 지표 (총수익 / 최대낙폭 / 샤프 / 일평균 턴오버)
GOLDEN_REV_BB_DOTCOM = {
    "total_return": 0.3065253378687751,
    "max_drawdown": -0.12727238857828482,
    "sharpe": 0.6938530760088509,
    "turnover": 0.07584269662921349,
}

# DCA 기준선 (무비용) raw 지표
GOLDEN_DCA = {
    "2000-02 닷컴 붕괴 체육관": {
        "total_return": -0.4344834503021633,
        "max_drawdown": -0.5347526469271779,
        "sharpe": -0.6575951841300133,
    },
    "2020 코로나 급락 체육관": {
        "total_return": 0.07305252136669904,
        "max_drawdown": -0.08782291812982201,
        "sharpe": 0.9749255040163419,
    },
}

GOLDEN_REV_BB_SCORE_VS_DCA_AVG = 0.16543869389660154


def _check(label: str, actual: float, expected: float, failures: list) -> None:
    ok = math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=1e-12)
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}: {actual!r} (기대 {expected!r})")
    if not ok:
        failures.append(label)


def run_check() -> bool:
    loaded = load_gyms(all_gyms())
    failures: list[str] = []

    print("=== 1. 적합도 골든 넘버 (평균 70% + 최약 30%) ===")
    for name, expected in GOLDEN_FITNESS.items():
        report = challenge(Strategy(genes=name.split("+"), name=name), loaded)
        _check(f"fitness {name}", report.fitness, expected, failures)

    print("\n=== 2. REV_BB 닷컴 체육관 raw 지표 ===")
    report = challenge(Strategy(genes=["REV_BB"], name="REV_BB"), loaded)
    dotcom = report.results[0]
    for field, expected in GOLDEN_REV_BB_DOTCOM.items():
        _check(f"REV_BB dotcom {field}", getattr(dotcom, field), expected, failures)

    print("\n=== 3. DCA 기준선 raw 지표 (무비용) ===")
    dca = {lg.gym.name: fight_dca(lg) for lg in loaded}
    for gym_name, fields in GOLDEN_DCA.items():
        for field, expected in fields.items():
            _check(f"DCA {gym_name} {field}",
                   getattr(dca[gym_name], field), expected, failures)

    print("\n=== 4. score_vs_dca ===")
    scores = [score_vs_dca(r, dca[r.gym_name]) for r in report.results]
    _check("REV_BB score_vs_dca 평균", sum(scores) / len(scores),
           GOLDEN_REV_BB_SCORE_VS_DCA_AVG, failures)

    print(f"\n=== 판정: {'PASS' if not failures else 'FAIL ' + str(failures)} ===")
    return not failures


# pytest로 돌릴 때도 같은 검증을 쓴다
def test_engine_golden_numbers():
    assert run_check(), "골든 넘버 불일치: 엔진 계산이 바뀜 (의도한 변경이면 골든 갱신)"


if __name__ == "__main__":
    sys.exit(0 if run_check() else 1)
