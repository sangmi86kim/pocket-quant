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

실행: 시즌 어댑터로 진입 (예: python -m app.league.v1.battle_frontier_lineup)
"""
import sys

# Windows cp949 콘솔에서 이모지 크래시 방지 (3.7+)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")    # type: ignore[union-attr]
    except Exception:
        pass

import numpy as np
import pandas as pd

from app.pocket.models import Gym
from app.pocket.battle import (_score_position, fight_dca, score_vs_dca,
                                         terminal_balance)
from app.pocket.signals import combine_positions, positions_with_params
from app.world.data_loader import LoadedGym, get_prices
from app.league.operations.regime_picks import update_regime_picks as _update_regime_picks

SEED_KRW = 1_000_000   # 표시·판정용 시드 (06-13 — 한 세계당 100만원)

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
    # 합성 세계의 날짜는 실제 달력이 아니라 순서 라벨이다. 외부 정보원(VIX/TNX 등)을
    # 이 날짜로 다시 붙이면 합성 QQQ 경로와 다른 역사 조각이 섞이므로, 외부 의존
    # 시그널은 signals.py에서 이 표식을 보고 기권한다.
    prices.attrs["synthetic"] = True
    start = prices.index[WARMUP_TDAYS]
    gym = Gym("평행세계", difficulty=0, volatility=0, ticker="SYNTH",
              start=start.strftime("%Y-%m-%d"),
              end=prices.index[-1].strftime("%Y-%m-%d"))
    return LoadedGym(gym=gym, prices=prices)


def run_gate2(graduates: list) -> bool:
    """챔피언로드 ② 평행세계 운빨 검사 — graduates는 시즌 어댑터가 준비해 주입한다.

    graduate dict 형식:
      - 시그널 후보: {"name","label","weights","params","mean5","specialist"}
      - NPC 후보  : 위 키 + "evaluator"(loaded, seed_krw) -> (returns, terminal)
                    NPC도 평행세계 출전, 매년 1등 카운트에 자연 포함. 도전권
                    판정에서는 제외.
    """
    prices = get_prices("QQQ", DATA_START, DATA_END)
    full_returns = prices.pct_change().dropna()
    regime_returns = {
        name: pd.concat([full_returns.loc[s:e] for s, e in spans])
        for name, spans in REGIME_SPANS.items()
    }

    # 입장 명단: 도전권 보유자(현챔피언) + 스페셜리스트 + 벤치 1위(참고)
    champion = graduates[0]
    specialists = [g for g in graduates if g["specialist"]]
    npcs = [g for g in graduates if "evaluator" in g]
    # 하드 필터 통과자가 0명이면 벤치 1위가 없다 — 없이 진행 (크래시 방지)
    bench_top = next((g for g in graduates[1:]
                      if not g["specialist"] and "evaluator" not in g), None)
    entrants = [champion] + specialists
    if bench_top is not None:
        bench_top = {**bench_top, "label": f"참고(벤치1위·{bench_top['label']})"}
        entrants = [champion, bench_top] + specialists + npcs
    else:
        print("⚠️ 하드 필터 통과자 0명 — 벤치 1위 없이 관문을 진행한다.")
        entrants = entrants + npcs

    print("=== 챔피언로드 관문 ② 배틀 프론티어: 평행세계 운빨 검사 ===")
    print(f"재료: QQQ {DATA_START}~{DATA_END} (사천왕 구간 봉인 유지)")
    print(f"세계: 전천후 {N_WORLDS_ALL}개 · 국면(스페셜리스트 본판정) {N_WORLDS_REGIME}개"
          f" · 블록 {BLOCK_DAYS}일 · 평가 {EVAL_TDAYS}일/세계 · 시드 {SEED}\n")

    rows, all_pass = [], False
    # 시험장(arena)별 잔고 매트릭스 — {arena: {후보이름: [세계별 잔고]}}.
    # 같은 arena·같은 SEED라 모든 후보가 동일한 세계 시퀀스 → 세계별 1등 비교 가능.
    bal_by_arena: dict[str, dict[str, list[int]]] = {}
    dca_by_arena: dict[str, list[int]] = {}    # arena별 성실이 잔고 (첫 후보 때만 채움)

    for g in entrants:
        # 본판정 세계 선택: 스페셜리스트는 전문 국면, 나머지는 전천후
        regime = next((k for k in REGIME_SPANS if k in g["label"]), None)
        pool = regime_returns[regime] if regime else None
        arena = regime or "전천후"
        n_worlds = N_WORLDS_REGIME if pool is not None else N_WORLDS_ALL

        rng = np.random.default_rng(SEED)           # 동일 시드 = 모두 같은 세계들에서 시험
        scores, defenses, bals = [], [], []
        record_dca = arena not in dca_by_arena
        if record_dca:
            dca_by_arena[arena] = []
        for _ in range(n_worlds):
            world = make_world(full_returns, rng, pool)
            dca_res = fight_dca(world)
            if "evaluator" in g:
                rets, term = g["evaluator"](world, SEED_KRW)
                scores.append(0.0)
                defenses.append(False)
                bals.append(term)
            else:
                res = _score_position(
                    combine_positions(positions_with_params(world.prices, g["params"]),
                                      g["weights"]), world)
                scores.append(score_vs_dca(res, dca_res))
                defenses.append(res.max_drawdown > res.market_drawdown)
                bals.append(terminal_balance(res, SEED_KRW))
            if record_dca:
                dca_by_arena[arena].append(terminal_balance(dca_res, SEED_KRW))
        bal_by_arena.setdefault(arena, {})[g["name"]] = bals

        win_rate = float(np.mean([s > 0 for s in scores]))
        defense_rate = float(np.mean(defenses))
        med = float(np.median(scores))
        p5 = float(np.percentile(scores, 5))
        passed = "evaluator" not in g and win_rate >= WIN_RATE_BAR and defense_rate >= DEFENSE_BAR
        if g is champion:
            all_pass = passed
        rows.append((g, arena, n_worlds, win_rate, defense_rate,
                     med, p5, passed))

    print(f"{'트레이더':<14} {'라벨':<22} {'시험장':<8} {'세계':>4} {'승률':>6}"
          f" {'방어':>6} {'중앙값':>7} {'하위5%':>7}  판정")
    for g, arena, n, wr, dr, med, p5, passed in rows:
        if "evaluator" in g:
            mark = "🤖 NPC (참고)"
            wr_s, dr_s, med_s, p5_s = "  -  ", "  -  ", "   -  ", "   -  "
        else:
            mark = "🎫 통과" if passed else "🪑 벤치"
            wr_s, dr_s = f"{wr:>6.0%}", f"{dr:>6.0%}"
            med_s, p5_s = f"{med * 100:>+7.1f}", f"{p5 * 100:>+7.1f}"
        print(f"{g['name']:<14} {g['label']:<22} {arena:<8} {n:>4} {wr_s}"
              f" {dr_s} {med_s} {p5_s}  {mark}")

    print(f"\n판정 기준: 라이벌전 승률 ≥ {WIN_RATE_BAR:.0%} AND 방어 승률 ≥ {DEFENSE_BAR:.0%}")
    print("벤치 ≠ 사망 — 상폐가 아니면 뒤진 게 아니다. 명단 보존, 재도전 가능.")

    # ── 시험장(arena)별 세계 1등 카운트 + 평균 잔고 (사용자 안 06-13) ──
    # ⚠️ 합성 세계라 Regime_Scanner 일별 판정 적용 불가 — 풀명을 그대로 라벨로.
    # PocketQuant bear = Regime_Scanner bear와 의미 일치, rebound는 별도 카테고리.
    gate2 = []
    print("\n=== 평행세계 시험장별 세계 1등 카운트 (100만원 시드 → 종료 잔고) ===")
    for arena, cand_bals in bal_by_arena.items():
        n_worlds = len(next(iter(cand_bals.values())))
        wins_count: dict[str, int] = {n: 0 for n in cand_bals}
        for i in range(n_worlds):
            winner = max(cand_bals, key=lambda n, i=i: cand_bals[n][i])
            wins_count[winner] += 1
        dca_mean = int(np.mean(dca_by_arena[arena]))
        cand_mean = {n: int(np.mean(b)) for n, b in cand_bals.items()}
        gate2.append({
            "arena": arena, "n_worlds": n_worlds,
            "wins_count": wins_count,
            "후보별_평균잔고": cand_mean,
            "성실이_평균잔고": dca_mean,
            "후보별_평균잔고차": {n: cand_mean[n] - dca_mean for n in cand_bals},
        })
        print(f"\n  [시험장: {arena}] {n_worlds}세계 · 성실이 평균 {dca_mean:,}원")
        for n, w in sorted(wins_count.items(), key=lambda kv: -kv[1]):
            diff = cand_mean[n] - dca_mean
            sign = "+" if diff >= 0 else ""
            print(f"    {n:<14} 1등 {w:>3}/{n_worlds:<3} "
                  f"  평균 {cand_mean[n]:>10,}원 ({sign}{diff:,})")

    out = _update_regime_picks("gate2_worlds", gate2)
    print(f"\nsaved: {out}")

    print("\n관문 ③ 사천왕(post-COVID hold-out): 🔒 봉인 — 사용자 승인 후 최후의 1회만")
    return all_pass


if __name__ == "__main__":
    # 단독 실행 금지 — graduates는 시즌 어댑터가 준비한다.
    # v1 시즌:   python app/league/v1/battle_frontier_lineup.py (자체 평가 경로)
    # v1.x 시즌: python app/league/v1x/battle_frontier.py       (자체 main)
    raise SystemExit(
        "[battle_frontier] 본 코어는 graduates 인자가 필요합니다. "
        "시즌 어댑터로 진입하세요 (예: app/league/v1/battle_frontier_lineup.py)."
    )
