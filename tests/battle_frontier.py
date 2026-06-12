"""
battle_frontier.py - 챔피언로드 관문 ②: 배틀 프론티어 (평행세계 운빨 검사)

[관문 ①과의 역할 분담]
관문 ①(리그 본선)은 "처음 보는 데이터에서도 통하나"를 깨끗한 OOS 연도로 쟀다.
관문 ②가 묻는 건 다르다: **"그 우위가 역사의 특정 '순서' 덕분(운빨)은 아닌가?"**
진짜 역사는 한 번뿐이라, 실제 QQQ 일별 수익률을 한 달(21거래일) 블록으로 잘라
무작위로 다시 이어붙인 평행세계를 수백 개 만들고, 각 세계에서 라이벌 성실이와
다시 붙인다. 같은 재료를 다른 순서로 섞어도 이기면 실력이다.

  - 재료: QQQ 1999-03 ~ 2020-06 수익률만 (사천왕 구간 2020-07~ 은 여기서도 봉인)
  - 한 달 블록 = 변동성 뭉침(공포는 며칠씩 몰려온다) 같은 단기 질감 보존
  - ⚠️ 한계: 훈련 기간 데이터의 재배열이므로 '새 정보' 시험이 아니다 —
    그건 관문 ①(깨끗한 연도)과 관문 ③(사천왕)의 몫. 여기는 순서-운빨 검사 전담.

[스페셜리스트 트랙]
한 국면 몰빵형은 전천후 세계가 아니라 **자기 전문 국면 블록으로만 만든 세계**에서
본판정한다 (bear 1위 → 하락 세계, rebound 1위 → 회복 세계). 평시 잣대로
스페셜리스트를 벤치 보내지 않기 위함 — 걔들은 Regime Scanner 틸트 후보다.

[판정 — 벤치 ≠ 사망 (상폐가 아니면 뒤진 게 아니다)]
  ① 라이벌전 승률 ≥ 55%  (평행세계 과반에서 성실이를 이김)
  ② 방어 승률    ≥ 80%  (평행세계 8할에서 낙폭이 B&H보다 얕음)
  둘 다 = 관문 ② 통과. 미달은 벤치(명단 보존, 재도전 가능).

실행: 프로젝트 루트에서  python tests/battle_frontier.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.backend.core.models import Gym
from app.backend.engine.battle import _score_position, fight_dca, score_vs_dca
from app.backend.genes.signals import combine_positions, positions_with_params
from app.backend.market.data import LoadedGym, get_prices
from tests.victory_road import load_graduates

# ── 세계 생성 설정 ────────────────────────────────
SEED = 42
BLOCK_DAYS = 21              # 한 달 블록 — 변동성 뭉침 보존, 그보다 긴 추세는 섞임
WARMUP_TDAYS = 270           # 지표 예열용 선행 구간 (캘린더 400일 ≈ 거래일 270일)
EVAL_TDAYS = 504             # 평가 구간 2년 (체육관 평균 길이와 비슷)
N_WORLDS_ALL = 200           # 전천후 세계 수
N_WORLDS_REGIME = 100        # 국면 세계 수 (스페셜리스트 본판정용)

# 사천왕 봉인: 부트스트랩 재료도 hold-out 직전까지만
DATA_START, DATA_END = "1999-03-10", "2020-06-30"

# 국면 블록 풀 — 훈련 체육관과 같은 구분 (스페셜리스트 전문 시험장 재료)
REGIME_SPANS = {
    "bear": [("2000-03-01", "2002-12-31"), ("2008-01-01", "2009-06-30")],
    "rebound": [("2003-01-01", "2004-12-31"), ("2009-03-01", "2010-12-31")],
}

# 판정 기준 (도전권 — 사망 판정 아님)
WIN_RATE_BAR = 0.55
DEFENSE_BAR = 0.80


def _sample_days(returns: pd.Series, n_days: int, rng) -> np.ndarray:
    """블록 부트스트랩: returns에서 BLOCK_DAYS짜리 블록을 복원추출해 n_days 길이로."""
    values = returns.to_numpy()
    chunks = []
    total = 0
    while total < n_days:
        i = rng.integers(0, len(values) - BLOCK_DAYS)
        chunks.append(values[i:i + BLOCK_DAYS])
        total += BLOCK_DAYS
    return np.concatenate(chunks)[:n_days]


def make_world(full_returns: pd.Series, rng,
               regime_returns: pd.Series | None = None) -> LoadedGym:
    """평행세계 1개 = [예열 구간(전체 역사 블록) + 평가 구간] 가격 시계열.

    예열은 항상 전체 역사에서 뽑는다 — 지표 초기화용일 뿐 성적과 무관.
    평가 구간은 전천후면 전체 역사, 국면 세계면 그 국면 블록에서만 뽑는다.
    날짜는 임의 라벨(영업일 연속) — 시그널은 거래일 수만 보므로 의미 없음."""
    warm = _sample_days(full_returns, WARMUP_TDAYS, rng)
    pool = regime_returns if regime_returns is not None else full_returns
    evald = _sample_days(pool, EVAL_TDAYS, rng)

    rets = np.concatenate([warm, evald])
    prices = pd.Series(100.0 * np.cumprod(1.0 + rets),
                       index=pd.bdate_range("2001-01-01", periods=len(rets)))
    start = prices.index[WARMUP_TDAYS]
    gym = Gym("평행세계", difficulty=0, volatility=0, ticker="SYNTH",
              start=start.strftime("%Y-%m-%d"),
              end=prices.index[-1].strftime("%Y-%m-%d"))
    return LoadedGym(gym=gym, prices=prices)


def run_gate2() -> bool:
    prices = get_prices("QQQ", DATA_START, DATA_END)
    full_returns = prices.pct_change().dropna()
    regime_returns = {
        name: pd.concat([full_returns.loc[s:e] for s, e in spans])
        for name, spans in REGIME_SPANS.items()
    }

    # 입장 명단: 도전권 보유자(현챔피언) + 스페셜리스트 + 벤치 1위(참고)
    graduates = load_graduates()
    champion = graduates[0]
    specialists = [g for g in graduates if g["specialist"]]
    # 하드 필터 통과자가 0명이면 벤치 1위가 없다 — 없이 진행 (크래시 방지)
    bench_top = next((g for g in graduates[1:] if not g["specialist"]), None)
    entrants = [champion] + specialists
    if bench_top is not None:
        bench_top = {**bench_top, "label": f"참고(벤치1위·{bench_top['label']})"}
        entrants = [champion, bench_top] + specialists
    else:
        print("⚠️ 하드 필터 통과자 0명 — 벤치 1위 없이 관문을 진행한다.")

    print("=== 챔피언로드 관문 ② 배틀 프론티어: 평행세계 운빨 검사 ===")
    print(f"재료: QQQ {DATA_START}~{DATA_END} (사천왕 구간 봉인 유지)")
    print(f"세계: 전천후 {N_WORLDS_ALL}개 · 국면(스페셜리스트 본판정) {N_WORLDS_REGIME}개"
          f" · 블록 {BLOCK_DAYS}일 · 평가 {EVAL_TDAYS}일/세계 · 시드 {SEED}\n")

    rows, all_pass = [], False
    for g in entrants:
        # 본판정 세계 선택: 스페셜리스트는 전문 국면, 나머지는 전천후
        regime = next((k for k in REGIME_SPANS if k in g["label"]), None)
        pool = regime_returns.get(regime)
        n_worlds = N_WORLDS_REGIME if pool is not None else N_WORLDS_ALL

        rng = np.random.default_rng(SEED)           # 동일 시드 = 모두 같은 세계들에서 시험
        scores, defenses = [], []
        for _ in range(n_worlds):
            world = make_world(full_returns, rng, pool)
            res = _score_position(
                combine_positions(positions_with_params(world.prices, g["params"]),
                                  g["weights"]), world)
            scores.append(score_vs_dca(res, fight_dca(world)))
            defenses.append(res.max_drawdown > res.market_drawdown)

        win_rate = float(np.mean([s > 0 for s in scores]))
        defense_rate = float(np.mean(defenses))
        med = float(np.median(scores))
        p5 = float(np.percentile(scores, 5))
        passed = win_rate >= WIN_RATE_BAR and defense_rate >= DEFENSE_BAR
        if g is champion:
            all_pass = passed
        rows.append((g, regime or "전천후", n_worlds, win_rate, defense_rate,
                     med, p5, passed))

    print(f"{'트레이더':<14} {'라벨':<22} {'시험장':<8} {'세계':>4} {'승률':>6}"
          f" {'방어':>6} {'중앙값':>7} {'하위5%':>7}  판정")
    for g, arena, n, wr, dr, med, p5, passed in rows:
        mark = "🎫 통과" if passed else "🪑 벤치"
        print(f"{g['name']:<14} {g['label']:<22} {arena:<8} {n:>4} {wr:>6.0%}"
              f" {dr:>6.0%} {med * 100:>+7.1f} {p5 * 100:>+7.1f}  {mark}")

    print(f"\n판정 기준: 라이벌전 승률 ≥ {WIN_RATE_BAR:.0%} AND 방어 승률 ≥ {DEFENSE_BAR:.0%}")
    print("벤치 ≠ 사망 — 상폐가 아니면 뒤진 게 아니다. 명단 보존, 재도전 가능.")
    print("\n관문 ③ 사천왕(post-COVID hold-out): 🔒 봉인 — 사용자 승인 후 최후의 1회만")
    return all_pass


if __name__ == "__main__":
    sys.exit(0 if run_gate2() else 1)
