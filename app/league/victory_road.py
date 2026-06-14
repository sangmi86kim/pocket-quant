"""
victory_road.py - 챔피언로드: 리그 졸업생의 검증 관문 (사천왕 직전 동굴)

[위치]
체육관 6관(NSGA-III 리그) 졸업 → ★챔피언로드★ → 사천왕(hold-out, 봉인) → 챔피언(배포)

[관문 ① 리그 본선 — OOS 연도 시험 (이 파일이 구현)]
리그 졸업생은 '고정된 트레이더'(가중치+파라미터 확정)다. 그러므로 검증은
"훈련 체육관에 안 들어간 깨끗한 연도"에 내보내 라이벌 성실이(DCA)와
1년 단위 라이벌전을 시키는 것:

  훈련 체육관이 먹은 해: 2000~02(닷컴) 2008~10(GFC+회복장) 2015~17(횡보+상승) 2020(코로나)
  깨끗한 OOS 연도 11개 : 2003 2004 2005 2006 2007 2011 2012 2013 2014 2018 2019
  봉인(사천왕)         : 2020-07 이후 — 여기서 절대 안 씀

  ⚠️ OOS 11년은 평시 위주다(위기의 해는 훈련 체육관이 가져감) — 이 관문은
     "평시에 보험료를 얼마나 적게 내는가" 성격의 시험. 위기 OOS는 부족하므로
     관문 ②(배틀 프론티어, 부트스트랩 가짜 역사)가 보완한다.
  ※ 지표 워밍업(400일)이 훈련 연도와 겹치는 건 누수가 아니다 — 지표 초기화일
     뿐, 후보 선발에 그 구간 '성적'을 쓴 게 아니므로.

[도전권 판정 — 트레이더 슬로건: "상폐가 아니면 뒤진 게 아니다"]
  ① 라이벌전: OOS 연평균 score_vs_dca > 0  (성실이보다 강해야 함 — 핵심)
  ② 방어    : OOS 이어붙임 MDD가 B&H보다 얕음
  둘 다 통과 = 사천왕 도전권 획득. 효율(샤프 vs B&H)은 참고 표기.
  ⚠️ 미통과 = 사망이 아니라 '벤치'다. 시장에 있는 한 복리는 일하고 있고,
  명단은 DB에 보존되며, 다음 리그/배틀 프론티어에서 재도전한다.
  여기서 하는 판단은 "누가 죽었나"가 아니라 "누구에게 도전권을 주나"뿐.

[관문 ② 배틀 프론티어] 평행세계 운빨 검사 — app.league.battle_frontier
[관문 ③ 사천왕] post-COVID hold-out — 봉인. 최후의 1회만.

실행: 시즌 어댑터로 진입 (예: python -m app.league.v1.champion_road_lineup)
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
from app.pocket import battle
from app.pocket.battle import (_score_position, fight_dca, score_vs_dca,
                                         terminal_balance)
from app.pocket.signals import combine_positions, positions_with_params
from app.world.data_loader import LoadedGym, WARMUP_DAYS, get_prices
from app.world.regime import REGIME_LABELS, dominant_regime
from app.league.operations.regime_picks import update_regime_picks as _update_regime_picks

SEED_KRW = 1_000_000   # 표시·판정용 시드 (06-13 — 매년 새로 들고 들어감)

TICKER = "QQQ"
OOS_YEARS = [2003, 2004, 2005, 2006, 2007, 2011, 2012, 2013, 2014, 2018, 2019]


# graduate 명단 만들기는 시즌 어댑터의 책임 — NSGA-III sqlite 로드 + summarize_front는
# v1 시즌의 일이므로 `app/league/v1/champion_road_lineup.py`로 이주했다. 본 코어는
# graduates를 받기만 하고, NPC 4인방은 graduate dict의 `"evaluator"` 키 분기로 처리.


# ── 평가 ──────────────────────────────────────────
def _loaded_window(prices: pd.Series, year: int) -> LoadedGym:
    """1년짜리 임시 체육관 (본 게임과 동일하게 워밍업 버퍼 포함)."""
    start, end = f"{year}-01-01", f"{year}-12-31"
    gym = Gym(f"{year} OOS", difficulty=0, volatility=0,
              ticker=TICKER, start=start, end=end)
    s = pd.Timestamp(start) - pd.Timedelta(days=WARMUP_DAYS)
    return LoadedGym(gym=gym, prices=prices.loc[s:pd.Timestamp(end)])


def _daily_returns(loaded: LoadedGym, weights, params) -> pd.Series:
    """OOS 구간 일별 수익 (이어붙임용) — battle._score_position과 동일 공식."""
    pos = combine_positions(positions_with_params(loaded.prices, params), weights).shift(1)
    ret = pos * loaded.prices.pct_change() - pos.diff().abs() * battle.TRADE_COST
    mask = (ret.index >= pd.Timestamp(loaded.gym.start)) \
         & (ret.index <= pd.Timestamp(loaded.gym.end))
    return ret[mask].dropna()


def _perf(returns: pd.Series) -> tuple[float, float, float]:
    eq = (1 + returns).cumprod()
    cagr = float(eq.iloc[-1] ** (battle.TRADING_DAYS / len(returns)) - 1)
    mdd = float((eq / eq.cummax() - 1).min())
    std = returns.std()
    sharpe = float(returns.mean() / std * np.sqrt(battle.TRADING_DAYS)) if std > 0 else 0.0
    return cagr, mdd, sharpe


def run_gate1(graduates: list) -> bool:
    """챔피언로드 ① 시험장 — graduates는 시즌 어댑터가 준비해 주입한다 (필수 인자).

    graduate dict 형식:
      - 시그널 후보: {"name","label","weights","params","academy","specialist"}
      - NPC 후보  : 위 키 + "evaluator"(loaded, seed_krw) -> (returns, terminal)
                    NPC는 도전권 판정에서 제외, 표시·연도별 1등 매트릭스에만 참여.
    """
    prices = get_prices(TICKER, "1999-03-10", "2026-06-09")
    loadeds = {y: _loaded_window(prices, y) for y in OOS_YEARS}
    dca = {y: fight_dca(lg) for y, lg in loadeds.items()}

    print(f"=== 챔피언로드 관문 ① 리그 본선: OOS {len(OOS_YEARS)}개 연도 "
          f"({OOS_YEARS[0]}~{OOS_YEARS[-1]}, 훈련 체육관 미사용 해) ===")
    print(f"도전자 {len(graduates)}명 (NPC 포함)\n")

    # B&H 이어붙임 (모든 후보 공통 비교선)
    bh_all = pd.concat([loadeds[y].prices.pct_change()[
        (loadeds[y].prices.index >= pd.Timestamp(f"{y}-01-01"))
        & (loadeds[y].prices.index <= pd.Timestamp(f"{y}-12-31"))].dropna()
        for y in OOS_YEARS])
    bc, bm, bs = _perf(bh_all)

    rows, survivors = [], []
    balances: dict[str, dict[int, int]] = {}    # {후보 이름: {year: 종료 잔고}}
    for g in graduates:
        scores, parts = [], []
        balances[g["name"]] = {}
        is_npc = "evaluator" in g
        for y in OOS_YEARS:
            if is_npc:
                # NPC 경로: evaluator가 직접 (returns, terminal) 반환
                rets, term = g["evaluator"](loadeds[y], SEED_KRW)
                parts.append(rets)
                balances[g["name"]][y] = term
                # NPC는 도전권 판정에서 제외 — score는 표시용 0 채움
                scores.append(0.0)
            else:
                # 시그널 가중치 경로
                res = _score_position(
                    combine_positions(positions_with_params(loadeds[y].prices, g["params"]),
                                      g["weights"]), loadeds[y])
                scores.append(score_vs_dca(res, dca[y]))
                parts.append(_daily_returns(loadeds[y], g["weights"], g["params"]))
                balances[g["name"]][y] = terminal_balance(res, SEED_KRW)
        avg = float(np.mean(scores))
        wins = sum(s > 0 for s in scores)
        worst = float(min(scores))
        sc, sm, ss = _perf(pd.concat(parts))
        rival_ok = avg > 0
        defense_ok = sm > bm
        ticket = (not is_npc) and rival_ok and defense_ok   # NPC는 도전권 후보 아님
        if ticket and not g["specialist"]:
            survivors.append(g["name"])
        rows.append((g, avg, wins, worst, sc, sm, ss, ticket, is_npc))

    def _academy_score(g: dict):
        academy = g.get("academy") or {}
        return academy.get("score")

    print(f"{'트레이더':<14} {'라벨':<12} {'평균':>6} {'승':>5} {'최악':>7}"
          f" {'CAGR':>7} {'MDD':>7} {'샤프':>5} {'학교':>7}  판정")
    for g, avg, wins, worst, sc, sm, ss, ticket, is_npc in rows:
        academy_score = _academy_score(g)
        academy_str = f"{academy_score * 100:+.1f}" if academy_score is not None else "-"
        if is_npc:
            mark = "🤖 NPC (참고)"
            avg_str = "  -  "
            wins_str = "   -"
            worst_str = "    -  "
        else:
            if g["specialist"]:
                mark = "📋 참고 (본판정=관문②)" if not ticket else "📋 참고 (관문①도 통과)"
            else:
                mark = "🎫 도전권" if ticket else "🪑 벤치"
            avg_str = f"{avg * 100:>+6.1f}"
            wins_str = f"{wins:>3}/{len(OOS_YEARS)}"
            worst_str = f"{worst * 100:>+7.1f}"
        print(f"{g['name']:<14} {g['label']:<12} {avg_str} {wins_str}"
              f" {worst_str} {sc:>+7.1%} {sm:>7.1%} {ss:>5.2f} {academy_str:>7}  {mark}")

    print(f"\nB&H 기준선: CAGR {bc:+.1%}  MDD {bm:.1%}  샤프 {bs:.2f}")
    print(f"도전권 조건: ①OOS 평균 score_vs_dca > 0  ②이어붙임 MDD가 B&H({bm:.1%})보다 얕음")
    print("벤치 ≠ 사망 — 상폐가 아니면 뒤진 게 아니다. 명단 보존, 다음 리그/관문②에서 재도전.")

    # ── 매년 100만원 시드 잔고 표 + 연도별 1등 + 국면 라벨 (사용자 안 06-13) ──
    # 옵티마이저는 점수(score_vs_dca 다목적)로 탐색, 사람용 표시·판정만 잔고차.
    # 국면 라벨은 Regime_Scanner와 동일 정의(market.regime) — 외부 도구 입력원.
    year_head = " ".join(f"{y - 2000:>5}" for y in OOS_YEARS)   # '03 04 05...
    print("\n=== 시험장 11년 × 100만원 시드 (단위: 만원, 종료 잔고) ===")
    print(f"  {'후보':<14} {'라벨':<12} {year_head}  {'합계':>6}")
    for g in graduates:
        cells = " ".join(f"{balances[g['name']][y] / 10000:>5.0f}" for y in OOS_YEARS)
        tot = sum(balances[g["name"]].values()) // 10000
        print(f"  {g['name']:<14} {g['label']:<12} {cells}  {tot:>5,}")
    dca_bals = {y: terminal_balance(dca[y], SEED_KRW) for y in OOS_YEARS}
    dca_cells = " ".join(f"{dca_bals[y] / 10000:>5.0f}" for y in OOS_YEARS)
    dca_tot = sum(dca_bals.values()) // 10000
    print(f"  {'성실이':<14} {'(DCA)':<12} {dca_cells}  {dca_tot:>5,}")

    # 성실이도 1등 후보로 등록 (사용자 안 06-13: "성실이도 챔피언로드 보내")
    balances["성실이"] = dca_bals

    # ── 연도별 1등 + 국면 → gate1_oos (Regime Scanner 입력원) ──
    gate1 = []
    for y in OOS_YEARS:
        regime_en = dominant_regime(prices, f"{y}-01-01", f"{y}-12-31")
        regime = REGIME_LABELS[regime_en]
        win_name = max(balances, key=lambda n: balances[n][y])
        win_bal = balances[win_name][y]
        gate1.append({"year": y, "regime": regime, "regime_en": regime_en,
                      "winner": win_name, "잔고": win_bal,
                      "성실이": dca_bals[y], "차": win_bal - dca_bals[y]})

    print("\n=== 시험장 연도별 1등 + 국면 ===")
    for e in gate1:
        sign = "+" if e["차"] >= 0 else ""
        print(f"  {e['year']} {e['regime']:<4} {e['winner']:<14} "
              f"{e['잔고']:>10,}원  (성실이 {e['성실이']:>10,}원, {sign}{e['차']:,})")

    out = _update_regime_picks("gate1_oos", gate1)
    print(f"\nsaved: {out}")

    # 과적합 갭: 인샘플 점수가 OOS를 예측하는가
    pairs = [(_academy_score(g), avg) for g, avg, *_ in rows
             if _academy_score(g) is not None]
    if len(pairs) >= 3:
        ins, oos = zip(*pairs)
        corr = float(np.corrcoef(ins, oos)[0, 1])
        print(f"\n과적합 진단: 학교 점수 ↔ OOS 평균 상관 = {corr:+.2f} "
              f"({'인샘플 순위가 OOS에서도 유지됨' if corr > 0.3 else '인샘플 성적은 OOS를 거의 예측 못함 — 과적합 신호'})")

    print(f"\n=== 관문 ① 결과: 사천왕 도전권 {len(survivors)}/{len(graduates)}명 ===")
    if survivors:
        print("  " + ", ".join(survivors))
    print("\n관문 ② 배틀 프론티어(평행세계 운빨 검사): 시즌 어댑터로 진입")
    print("관문 ③ 사천왕(post-COVID hold-out): 🔒 봉인 — 최후의 1회만")
    return len(survivors) > 0


if __name__ == "__main__":
    # 단독 실행 금지 — graduates는 시즌 어댑터가 준비한다.
    # v1 시즌:   python app/league/v1/champion_road_lineup.py
    # v1.x 시즌: python app/league/v1x/champion_road_lineup.py
    raise SystemExit(
        "[victory_road] 본 코어는 graduates 인자가 필요합니다. "
        "시즌 어댑터로 진입하세요 (예: app/league/v1/champion_road_lineup.py)."
    )
