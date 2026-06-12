"""
elite_four.py - 챔피언로드 관문 ③: 사천왕 (봉인된 최종전 — 1회용)

[봉인 해제 기록]
post-COVID hold-out(2020-07-01 ~ )은 프로젝트 시작부터 봉인해 온 최종 시험지다.
훈련(체육관)·검증(관문 ①·②) 어디에도 이 구간을 쓰지 않았고, cGAN 학습 계획에서도
제외했다. 2026-06-11 사용자 승인으로 개봉 — **이 시험은 1회용이다.**
이 결과를 보고 적합도/가중치/파라미터를 고치면 hold-out이 세 번째 훈련셋이
되므로 반칙. 결과는 결과대로 기록한다.

[도전자]
관문 ①(깨끗한 OOS 연도)·②(평행세계 운빨 검사)를 모두 통과한 유일한 트레이더:
현 챔피언 = 동일가중 VOL + REV_RSI + REV_BB, 시그널 파라미터 기본값.

[판정 기준 — 개봉 전에 못박음 (관문 ①과 동일)]
  ① 라이벌전: 연도별 score_vs_dca 평균 > 0  (성실이보다 강함)
  ② 방어    : 전 구간 이어붙임 MDD가 B&H보다 얕음
출력: 콘솔 + reports/elite_four_report.html (자산곡선 포함)

실행: 프로젝트 루트에서  python tests/elite_four.py
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows cp949 콘솔에서 이모지 크래시 방지 (3.7+)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")    # type: ignore[union-attr]
    except Exception:
        pass

import numpy as np
import pandas as pd

from app.backend.core.models import Gym, Report, Strategy
from app.backend.engine import battle
from app.backend.engine.battle import (_dca_position, _score_position, fight_dca,
                                       score_vs_dca, terminal_balance)
from app.backend.genes.signals import ALL_GENES, combine_positions, positions_with_params
from app.backend.market.data import LoadedGym, WARMUP_DAYS, get_prices
from app.backend.market.regime import REGIME_LABELS, dominant_regime
from app.service import _update_regime_picks

_ROOT = Path(__file__).resolve().parent.parent
OUT = _ROOT / "reports" / "elite_four_report.html"

HOLDOUT_START, DATA_END = "2020-07-01", "2026-06-09"
TICKER = "QQQ"
SEED_KRW = 1_000_000   # 표시·판정용 시드 (06-13 — 매년 100만원)

# 도전자: 관문 ①·② 통과자 (고정 — 여기서 다른 후보를 끼워넣지 않는다)
CHAMPION_WEIGHTS = [1.0 if g in ("VOL", "REV_RSI", "REV_BB") else 0.0 for g in ALL_GENES]

# 연 단위 라운드 (마지막은 데이터 끝까지)
ROUNDS = [
    ("2020 하반기", "2020-07-01", "2020-12-31"),
    ("2021", "2021-01-01", "2021-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),   # 금리 인상 약세장 — 봉인 구간의 위기 시험
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
    ("2026 (~06-09)", "2026-01-01", DATA_END),
]


def _loaded_window(prices: pd.Series, start: str, end: str) -> LoadedGym:
    gym = Gym(f"{start}~{end}", difficulty=0, volatility=0,
              ticker=TICKER, start=start, end=end)
    s = pd.Timestamp(start) - pd.Timedelta(days=WARMUP_DAYS)
    return LoadedGym(gym=gym, prices=prices.loc[s:pd.Timestamp(end)])


def _champion_returns(loaded: LoadedGym) -> pd.Series:
    """챔피언의 구간 일별 수익 (battle._score_position과 동일 공식, 비용 0.1%)."""
    pos = combine_positions(positions_with_params(loaded.prices),
                            CHAMPION_WEIGHTS).shift(1)
    ret = pos * loaded.prices.pct_change() - pos.diff().abs() * battle.TRADE_COST
    mask = (ret.index >= pd.Timestamp(loaded.gym.start)) \
         & (ret.index <= pd.Timestamp(loaded.gym.end))
    return ret[mask].dropna()


def _dca_returns(loaded: LoadedGym) -> pd.Series:
    """성실이의 구간 일별 수익 (무비용 — 토스 자동 모으기)."""
    pos = _dca_position(loaded).shift(1)
    ret = pos * loaded.prices.pct_change()
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


def _equity_svg(curves: dict) -> str:
    """자산곡선 SVG (시작 100). curves = {이름: (색, equity 시리즈)}."""
    W, H, PAD = 920, 380, 50
    lo = min(float(eq.min()) for _c, eq in curves.values())
    hi = max(float(eq.max()) for _c, eq in curves.values())
    n = max(len(eq) for _c, eq in curves.values())

    def sx(i):
        return PAD + i / (n - 1) * (W - 2 * PAD)

    def sy(v):
        return H - PAD - (v - lo) / (hi - lo) * (H - 2 * PAD)

    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">',
             f'<rect width="{W}" height="{H}" fill="#fbfcfe" rx="8"/>',
             f'<line x1="{PAD}" y1="{sy(100):.1f}" x2="{W - PAD}" y2="{sy(100):.1f}"'
             f' stroke="#ddd" stroke-dasharray="4"/>']
    for label, (color, eq) in curves.items():
        step = max(1, len(eq) // 400)                 # ~400포인트로 다운샘플
        pts = " ".join(f"{sx(i):.1f},{sy(float(v)):.1f}"
                       for i, v in list(enumerate(eq))[::step])
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{W - PAD + 4}" y="{sy(float(eq.iloc[-1])):.1f}"'
                     f' font-size="12" fill="{color}">{label} {float(eq.iloc[-1]):.0f}</text>')
    parts.append(f'<text x="{PAD - 4}" y="{sy(100) + 4:.1f}" text-anchor="end"'
                 f' font-size="10" fill="#888">100</text>')
    parts.append("</svg>")
    return "".join(parts)


def run_gate3() -> bool:
    prices = get_prices(TICKER, "1999-03-10", DATA_END)

    print("=== 챔피언로드 관문 ③: 사천왕 (봉인 해제 — 1회용) ===")
    print(f"구간: {HOLDOUT_START} ~ {DATA_END} (훈련·검증 미사용 봉인 구간)")
    print("도전자: 현챔피언 (동일가중 VOL+REV_RSI+REV_BB, 기본 파라미터)\n")

    # 라운드(연 단위) 라이벌전
    rows, scores = [], []
    print(f"{'라운드':<14} {'챔피언':>8} {'성실이':>8} {'B&H':>8}"
          f" {'챔MDD':>8} {'B&H MDD':>8} {'score':>7}")
    for name, start, end in ROUNDS:
        lw = _loaded_window(prices, start, end)
        res = _score_position(
            combine_positions(positions_with_params(lw.prices), CHAMPION_WEIGHTS), lw)
        dca = fight_dca(lw)
        s = score_vs_dca(res, dca)
        scores.append(s)
        rows.append((name, res, dca, s))
        print(f"{name:<14} {res.total_return:>+7.1%} {dca.total_return:>+7.1%}"
              f" {res.market_return:>+7.1%} {res.max_drawdown:>8.1%}"
              f" {res.market_drawdown:>8.1%} {s * 100:>+7.1f}")

    # 전 구간 이어붙임 (자산곡선·방어 판정용)
    full = _loaded_window(prices, HOLDOUT_START, DATA_END)
    champ_ret = _champion_returns(full)
    dca_ret = _dca_returns(full)
    bh_ret = full.prices.pct_change()[champ_ret.index].dropna()
    cc, cm, cs = _perf(champ_ret)
    dc, dm, ds = _perf(dca_ret)
    bc, bm, bs = _perf(bh_ret)

    avg_score = float(np.mean(scores))
    rival_ok = avg_score > 0
    defense_ok = cm > bm
    passed = rival_ok and defense_ok

    print(f"\n=== 전 구간 ({HOLDOUT_START}~{DATA_END}) ===")
    print(f"  챔피언: 공격력 {cc:+.1%}/년 (CAGR) · 최대 데미지 {cm:.1%} (MDD) · 컨트롤 {cs:.2f} (샤프)")
    print(f"  성실이: 공격력 {dc:+.1%}/년 (CAGR) · 최대 데미지 {dm:.1%} (MDD) · 컨트롤 {ds:.2f} (샤프)")
    print(f"  B&H   : 공격력 {bc:+.1%}/년 (CAGR) · 최대 데미지 {bm:.1%} (MDD) · 컨트롤 {bs:.2f} (샤프)")
    print(f"\n=== 판정 (개봉 전 사전 등록 기준) ===")
    print(f"  ① 라이벌전 (연평균 score > 0)      : {'PASS' if rival_ok else 'FAIL'} ({avg_score * 100:+.1f})")
    print(f"  ② 방어 (최대 데미지 {cm:.1%} vs B&H {bm:.1%}) : {'PASS' if defense_ok else 'FAIL'}")
    print(f"\n{'👑 사천왕 격파 — 챔피언 확정 (배포 근거 완성)' if passed else '🪑 사천왕 벽 — 결과는 결과대로 기록 (재튜닝 금지, 다음 알파에서 재도전)'}")

    # ── 매년 100만원 시드 잔고 + 국면 라벨 → gate3_holdout (사용자 안 06-13) ──
    # 사천왕은 도전자 1명(챔피언)뿐이라 "후보 1등"보다 "연도별 잔고 + 국면"이 본질.
    # Regime Scanner가 추후 "이 국면에서 챔피언이 약했다"는 신호를 읽을 수 있게 기록.
    gate3 = []
    for name, start, end in ROUNDS:
        regime_en = dominant_regime(prices, start, end)
        regime = REGIME_LABELS[regime_en]
        # rows 매칭 (name으로)
        _, res, dca_res, s = next(r for r in rows if r[0] == name)
        c_bal = terminal_balance(res, SEED_KRW)
        d_bal = terminal_balance(dca_res, SEED_KRW)
        gate3.append({"year": name, "regime": regime, "regime_en": regime_en,
                      "winner": "현챔피언", "잔고": c_bal, "성실이": d_bal,
                      "차": c_bal - d_bal, "성실이_이김": c_bal > d_bal,
                      "score": float(s)})

    print("\n=== 사천왕 라운드별 잔고 + 국면 (100만원 시드 → 종료 잔고) ===")
    for e in gate3:
        sign = "+" if e["차"] >= 0 else ""
        mark = "✅" if e["성실이_이김"] else "❌"
        print(f"  {e['year']:<14} {e['regime']:<4} 챔피언 {e['잔고']:>10,}원  "
              f"성실이 {e['성실이']:>10,}원  ({sign}{e['차']:,})  {mark}")

    out = _update_regime_picks("gate3_holdout", gate3)
    print(f"saved: {out}")

    # 오박사 코너 — LM Studio가 켜져 있으면 hold-out 성적표를 직접 브리핑.
    # 부재 시엔 둔치 고정 대사 (리포트 생성은 오박사 없이도 항상 성공해야 한다).
    from app.oak import professor_briefing
    oak_report = Report(
        strategy=Strategy(genes=["VOL", "REV_RSI", "REV_BB"], name="현챔피언"),
        results=[r[1] for r in rows])
    oak_text = professor_briefing(oak_report) \
        or "그래. 그럴 수 있어. 시장이 원래 그래. (소주 한 모금)"

    _write_html(rows, (cc, cm, cs), (dc, dm, ds), (bc, bm, bs),
                avg_score, rival_ok, defense_ok, passed,
                champ_ret, dca_ret, bh_ret, oak_text)
    print(f"\nHTML 리포트: {OUT}")
    return passed


def _write_html(rows, champ_perf, dca_perf, bh_perf, avg_score,
                rival_ok, defense_ok, passed, champ_ret, dca_ret, bh_ret,
                oak_text: str = "") -> None:
    import html as _html
    cc, cm, cs = champ_perf
    dc, dm, ds = dca_perf
    bc, bm, bs = bh_perf
    oak_html = _html.escape(oak_text)

    round_rows = ""
    for name, res, dca, s in rows:
        cls = "pos" if s > 0 else "neg"
        round_rows += (f"<tr><td>{name}</td><td>{res.total_return * 100:+.1f}%</td>"
                       f"<td>{dca.total_return * 100:+.1f}%</td>"
                       f"<td>{res.market_return * 100:+.1f}%</td>"
                       f"<td>{res.max_drawdown * 100:.1f}%</td>"
                       f"<td>{res.market_drawdown * 100:.1f}%</td>"
                       f'<td class="{cls}">{s * 100:+.1f}</td></tr>')

    svg = _equity_svg({
        "챔피언": ("#2b6fb3", (1 + champ_ret).cumprod() * 100),
        "성실이": ("#d4720c", (1 + dca_ret).cumprod() * 100),
        "B&H": ("#9aa4af", (1 + bh_ret).cumprod() * 100),
    })

    verdict_html = ("👑 <b>사천왕 격파 — 챔피언 확정</b> (배포 근거 완성)" if passed else
                    "🪑 <b>사천왕 벽</b> — 결과는 결과대로 기록. 이 결과를 보고 재튜닝하면 반칙(hold-out 오염). 다음 알파(시그널 풀/오버레이)로 재도전.")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    html_doc = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<title>PocketQuant — 사천왕전 (봉인 해제)</title>
<style>
 body {{ font-family: 'Segoe UI', 'Malgun Gothic', sans-serif; max-width: 980px;
        margin: 24px auto; padding: 0 16px; color: #222; }}
 h1 {{ font-size: 26px; }} h2 {{ font-size: 19px; margin-top: 34px;
      border-bottom: 2px solid #eee; padding-bottom: 6px; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 12px 0; }}
 th, td {{ border: 1px solid #e3e7ec; padding: 6px 10px; text-align: right; }}
 th {{ background: #f4f6f9; }} td:first-child, th:first-child {{ text-align: left; }}
 .pos {{ color: #1d8a4f; font-weight: 600; }} .neg {{ color: #c0392b; }}
 .dim {{ color: #8a94a0; font-size: 12px; }}
 .verdict {{ border: 2px solid {'#1d8a4f' if passed else '#c0392b'}; border-radius: 10px;
            padding: 16px 20px; font-size: 16px; background: {'#f2faf5' if passed else '#fdf3f2'}; }}
 .warn {{ background: #fff7e8; border: 1px solid #f0d9a8; border-radius: 8px; padding: 12px 16px; }}
</style></head><body>

<h1>👑 사천왕전 — 봉인 해제 (1회용)</h1>
<p class="dim">구간 {HOLDOUT_START} ~ {DATA_END} (훈련·검증 일절 미사용) ·
도전자: 현챔피언(동일가중 VOL+REV_RSI+REV_BB) · 관문 ①·② 통과 후 도전 ·
개봉 {stamp}</p>

<div class="verdict">{verdict_html}<br>
<span class="dim">사전 등록 기준 — ① 연도별 성실이전 평균 {avg_score * 100:+.1f}
({'PASS' if rival_ok else 'FAIL'}) · ② 최대 데미지(MDD) {cm * 100:.1f}% vs B&H {bm * 100:.1f}%
({'PASS' if defense_ok else 'FAIL'})</span></div>

<h2>📈 자산곡선 (시작 100 · 봉인 구간 전체)</h2>
{svg}
<p class="dim">챔피언은 비용 0.1%/편도 부담, 성실이(일별 적립)는 토스 자동 모으기 기준 무비용.</p>

<h2>🥊 연 단위 라운드</h2>
<table><tr><th>라운드</th><th>챔피언 획득 골드<br><span class="dim">(수익)</span></th>
<th>성실이 획득 골드<br><span class="dim">(수익)</span></th><th>B&H 획득 골드<br><span class="dim">(수익)</span></th>
<th>챔피언 최대 데미지<br><span class="dim">(MDD)</span></th><th>B&H 최대 데미지<br><span class="dim">(MDD)</span></th>
<th>라이벌전 점수<br><span class="dim">(score_vs_dca)</span></th></tr>
{round_rows}</table>
<p class="dim">라이벌전 점수 = 0.4×수익차 + 0.4×낙폭개선 + 0.2×샤프차 (성실이 대비, ×100. 양수 = 성실이보다 강함)</p>

<h2>📊 전 구간 종합 스탯</h2>
<table><tr><th></th><th>공격력 <span class="dim">(CAGR, 연평균 성장)</span></th>
<th>최대 데미지 <span class="dim">(MDD, HP가 가장 깊게 패인 순간)</span></th>
<th>컨트롤 <span class="dim">(샤프, 멀미 대비 수익)</span></th></tr>
<tr><td>챔피언</td><td>{cc * 100:+.1f}%/년</td><td>{cm * 100:.1f}%</td><td>{cs:.2f}</td></tr>
<tr><td>성실이 (일별 DCA)</td><td>{dc * 100:+.1f}%/년</td><td>{dm * 100:.1f}%</td><td>{ds:.2f}</td></tr>
<tr><td>B&H (그냥 들고 있기)</td><td>{bc * 100:+.1f}%/년</td><td>{bm * 100:.1f}%</td><td>{bs:.2f}</td></tr></table>
<p class="dim">읽는 법: 챔피언은 6년 동안 매년 {cc * 100:+.1f}%씩 자산을 키웠고(공격력),
가장 깊게 맞은 순간이 {cm * 100:.1f}%였으며(최대 데미지 — B&H는 {bm * 100:.1f}%까지 맞음),
출렁임 대비 효율(컨트롤)은 {cs:.2f}로 셋 중 가장 안정적으로 벌었다.</p>

<h2>🍶 오박사 코너</h2>
<div style="display:flex; gap:18px; align-items:flex-start;">
  <img src="../character/dr_oh.png" alt="오박사" width="280"
       style="border-radius:10px; flex-shrink:0;">
  <div style="white-space:pre-wrap; line-height:1.7;">{oak_html}</div>
</div>

<h2>⚠️ 이 시험의 규칙</h2>
<div class="warn">
이 구간은 <b>1회용</b>이었다 — 이제 개봉됐으므로 더 이상 깨끗한 시험지가 아니다.
이 결과를 보고 적합도·가중치·파라미터를 고치면 hold-out이 훈련셋이 되는 반칙.
앞으로의 개선(시그널 풀 확장, Regime 오버레이, 아카데미)은 이 구간을 다시
'참고'할 수는 있어도 '최종 판정'으로 쓸 수는 없다 — 다음 최종 시험지는
지금부터 쌓이는 미래 데이터다.
</div>

<footer class="dim" style="margin:40px 0 16px">PocketQuant · 챔피언로드 관문 ③ ·
python tests/elite_four.py 로 재생성 (단, 판정의 의미는 첫 개봉 1회뿐)</footer>
</body></html>"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html_doc, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(0 if run_gate3() else 1)
