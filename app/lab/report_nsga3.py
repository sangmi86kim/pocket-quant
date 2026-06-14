"""
report_nsga3.py - NSGA-III 스터디 결과를 HTML 리포트로 내보내기

저장된 스터디(sqlite)를 읽어 사람이 보기 좋은 한 장짜리 리포트를 만든다:
  요약 카드 · 필터 스윕 · 라벨 후보 · Pareto 산점도(SVG) · 통과 후보 전체 표 · 관찰/경고

출력: reports/nsga3_report.html  (reports/는 gitignore — 로컬 열람용)
실행: 프로젝트 루트에서  python -m app.lab.report_nsga3
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import optuna

import app.academy.training.nsga3 as nsga3
from app.academy.engine.battle import fight_dca
from app.academy.genes.signals import ALL_GENES
from app.academy.gym import all_gyms
from app.world.data import load_gyms

_ROOT = Path(__file__).resolve().parents[2]
STORAGE = f"sqlite:///{(_ROOT / 'optuna_pocketquant.db').as_posix()}"
STUDY = "nsga3_v2_weights"
OUT = _ROOT / "reports" / "nsga3_report.html"

OBJ_LABELS = {"bear": "하락장", "rebound": "회복장", "crash_v": "급락V",
              "bull": "상승장", "chop": "횡보장"}


def _weights_str(params: dict) -> str:
    w = [params[f"w_{g}"] for g in ALL_GENES]
    total = sum(w) or 1.0
    parts = [(g, x / total * 100) for g, x in zip(ALL_GENES, w)]
    return " · ".join(f"{g} {pct:.0f}%" for g, pct in parts if pct >= 1)


def _params_str(p: dict) -> str:
    if "DD_LIMIT" not in p:
        return "파라미터 기본값 고정"
    return (f"DD {p['DD_LIMIT']:.2f} / MA {p['MA_WINDOW']} / MOM {p['MOM_LOOKBACK']} / "
            f"RSI&lt;{p['RSI_OVERSOLD']} / BB k{p['BB_K']:.2f} / "
            f"VOL {p['VOL_CALM']:.3f}~{p['VOL_CALM'] + p['VOL_SPREAD']:.3f}")


def _vec_cells(values: list[float]) -> str:
    cells = []
    for v in values[:5]:
        cls = "pos" if v > 0 else "neg"
        cells.append(f'<td class="{cls}">{v * 100:+.1f}</td>')
    cells.append(f"<td>{values[5]:.3f}</td>")
    return "".join(cells)


def _scatter_svg(points, ref, labeled, x_idx, y_idx, x_name, y_name,
                 x_scale=100.0, y_scale=100.0) -> str:
    """front 산점도 SVG. points=[(values, passed?)], ref=기준점 values, labeled={이름:row}."""
    width, height, pad = 460, 340, 46
    xs = [p[0][x_idx] * x_scale for p in points] + [ref[x_idx] * x_scale]
    ys = [p[0][y_idx] * y_scale for p in points] + [ref[y_idx] * y_scale]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)

    def sx(v):
        return pad + (v - x_lo) / (x_hi - x_lo or 1) * (width - 2 * pad)

    def sy(v):
        return height - pad - (v - y_lo) / (y_hi - y_lo or 1) * (height - 2 * pad)

    dots = []
    for values, passed in points:
        x, y = sx(values[x_idx] * x_scale), sy(values[y_idx] * y_scale)
        if passed:
            dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#2b6fb3" opacity="0.9"/>')
        else:
            dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" fill="#b9c4cf" opacity="0.45"/>')

    colors = {"Defensive": "#1d8a4f", "Balanced": "#d4720c",
              "Aggressive": "#b3372b", "Low-turnover": "#6b4fb3"}
    for name, row in labeled.items():
        x, y = sx(row["values"][x_idx] * x_scale), sy(row["values"][y_idx] * y_scale)
        c = colors.get(name, "#333")
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{c}" stroke="#fff" stroke-width="1.5"/>'
                    f'<text x="{x + 8:.1f}" y="{y + 4:.1f}" font-size="11" fill="{c}">{name}</text>')

    rx, ry = sx(ref[x_idx] * x_scale), sy(ref[y_idx] * y_scale)
    dots.append(f'<text x="{rx - 8:.1f}" y="{ry + 6:.1f}" font-size="18" fill="#c0392b">★</text>'
                f'<text x="{rx + 8:.1f}" y="{ry + 4:.1f}" font-size="11" fill="#c0392b">현 챔피언</text>')

    grid_zero = ""
    if x_lo < 0 < x_hi:
        grid_zero += f'<line x1="{sx(0):.1f}" y1="{pad}" x2="{sx(0):.1f}" y2="{height - pad}" stroke="#ddd" stroke-dasharray="4"/>'
    if y_lo < 0 < y_hi:
        grid_zero += f'<line x1="{pad}" y1="{sy(0):.1f}" x2="{width - pad}" y2="{sy(0):.1f}" stroke="#ddd" stroke-dasharray="4"/>'

    return f"""<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
<rect width="{width}" height="{height}" fill="#fbfcfe" rx="8"/>
<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#888"/>
<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#888"/>
{grid_zero}{''.join(dots)}
<text x="{width / 2}" y="{height - 10}" text-anchor="middle" font-size="12" fill="#444">{x_name}</text>
<text x="14" y="{height / 2}" text-anchor="middle" font-size="12" fill="#444" transform="rotate(-90 14 {height / 2})">{y_name}</text>
<text x="{pad}" y="{height - pad + 16}" font-size="10" fill="#888">{x_lo:.1f}</text>
<text x="{width - pad}" y="{height - pad + 16}" text-anchor="end" font-size="10" fill="#888">{x_hi:.1f}</text>
<text x="{pad - 4}" y="{height - pad}" text-anchor="end" font-size="10" fill="#888">{y_lo:.1f}</text>
<text x="{pad - 4}" y="{pad + 4}" text-anchor="end" font-size="10" fill="#888">{y_hi:.1f}</text>
</svg>"""


def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=STUDY, storage=STORAGE)
    front = study.best_trials
    loaded = load_gyms(all_gyms())
    dca = {lg.gym.name: fight_dca(lg) for lg in loaded}
    summary = nsga3.summarize_front(study, loaded_gyms=loaded, dca=dca)
    passed_numbers = {r["number"] for r in summary["passed"]}

    ref_d = nsga3.reference_vector(loaded, dca)
    ref = [min(ref_d["dotcom"], ref_d["gfc"]), ref_d["rebound"], ref_d["crash_v"],
           ref_d["bull"], ref_d["chop"], ref_d["turnover"]]
    ref_mean = sum(ref[:5]) / 5

    sweep_rows = ""
    for tol in (0.0, 0.02, 0.05, 0.10):
        s = nsga3.summarize_front(study, tolerance=tol, loaded_gyms=loaded, dca=dca)
        sweep_rows += (f"<tr><td>전 국면 ≥ {-tol * 100:+.0f}</td>"
                       f"<td>{len(s['passed'])}개</td></tr>")

    label_rows = ""
    descr = {"Defensive": "하락장 최강 - 보험 전문",
             "Balanced": "5국면 평균 최고 - 올라운더",
             "Aggressive": "회복+상승 최강 - 공격수",
             "Low-turnover": "매매 최소 - 수수료 절약형"}
    for name, row in summary["labels"].items():
        label_rows += (f'<tr><td><b>{name}</b><br><span class="dim">{descr[name]}</span></td>'
                       f'<td>#{row["number"]}</td>{_vec_cells(row["values"])}'
                       f'<td class="pos"><b>{row["mean5"] * 100:+.1f}</b></td>'
                       f'<td class="small">{_weights_str(row["params"])}<br>'
                       f'<span class="dim">{_params_str(row["params"])}</span></td></tr>')

    passed_rows = ""
    for r in sorted(summary["passed"], key=lambda r: -r["mean5"]):
        mean_cls = "pos" if r["mean5"] > 0 else "neg"
        passed_rows += (f'<tr><td>#{r["number"]}</td>{_vec_cells(r["values"])}'
                        f'<td class="{mean_cls}">{r["mean5"] * 100:+.1f}</td>'
                        f'<td class="small">{_weights_str(r["params"])}</td></tr>')

    points = [(list(t.values), t.number in passed_numbers) for t in front]
    chart1 = _scatter_svg(points, ref, summary["labels"], 0, 3,
                          "하락장(bear) 점수 ×100 - 보험 성능 →",
                          "상승장(bull) 점수 ×100 - 보험료 ↑")
    mean_points = [([sum(v[:5]) / 5] + v[1:], p) for v, p in points]
    ref_mean_vec = [ref_mean] + ref[1:]
    chart2 = _scatter_svg(mean_points, ref_mean_vec, {
        n: {"values": [r["mean5"]] + r["values"][1:]} for n, r in summary["labels"].items()
    }, 5, 0, "턴오버 (일평균 매매 비율)", "5국면 평균 점수 ×100", x_scale=1.0)

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    html_doc = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<title>PocketQuant - NSGA-III 리그 결과</title>
<style>
 body {{ font-family: 'Segoe UI', 'Malgun Gothic', sans-serif; max-width: 1080px;
        margin: 24px auto; padding: 0 16px; color: #222; background: #fff; }}
 h1 {{ font-size: 26px; }} h2 {{ font-size: 19px; margin-top: 36px;
      border-bottom: 2px solid #eee; padding-bottom: 6px; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 12px 0; }}
 th, td {{ border: 1px solid #e3e7ec; padding: 6px 9px; text-align: right; }}
 th {{ background: #f4f6f9; }} td:first-child, th:first-child {{ text-align: left; }}
 .pos {{ color: #1d8a4f; font-weight: 600; }} .neg {{ color: #c0392b; }}
 .dim {{ color: #8a94a0; font-weight: 400; font-size: 12px; }}
 .small {{ font-size: 12px; text-align: left; }}
 .cards {{ display: flex; gap: 14px; flex-wrap: wrap; }}
 .card {{ flex: 1; min-width: 230px; border: 1px solid #e3e7ec; border-radius: 10px;
         padding: 14px 18px; background: #fbfcfe; }}
 .card .big {{ font-size: 24px; font-weight: 700; }}
 .charts {{ display: flex; gap: 16px; flex-wrap: wrap; }}
 .charts > div {{ flex: 1; min-width: 380px; }}
 .warn {{ background: #fff7e8; border: 1px solid #f0d9a8; border-radius: 8px;
         padding: 12px 16px; }}
 footer {{ color: #999; font-size: 12px; margin: 40px 0 16px; }}
</style></head><body>

<h1>PocketQuant - NSGA-III 리그 결과</h1>
<p class="dim">스터디 <b>{STUDY}</b> · 가상 트레이더 {len(study.trials)}명 참가
(트라이얼 1개 = 포켓퀀트 6마리를 어떤 비중·세팅으로 굴릴지 정한 트레이더 1명) ·
시드 42 · 체육관 6개(QQQ) · 라이벌 = 성실이(일별 DCA, 수수료 0원) · 생성 {stamp}</p>

<div class="cards">
 <div class="card">Pareto front<br><span class="big">{len(front)}</span> 명
   <br><span class="dim">서로 우열 못 가리는 트레이더</span></div>
 <div class="card">하드 필터 통과<br><span class="big">{len(summary['passed'])}</span> 명
   <br><span class="dim">전 국면 ≥ -5 · 턴오버 ≤ 0.10</span></div>
 <div class="card">현 챔피언 평균<br><span class="big">{ref_mean * 100:+.1f}</span>
   <br><span class="dim">VOL+REV_RSI+REV_BB (동일가중)</span></div>
 <div class="card">신인왕 Balanced 평균<br><span class="big pos">{max((r['mean5'] for r in summary['passed']), default=0) * 100:+.1f}</span>
   <br><span class="dim">5국면 중 4개에서 챔피언 역전</span></div>
</div>

<h2>기준점: 현 단일목적 챔피언</h2>
<table><tr><th>전략</th><th>하락장</th><th>회복장</th><th>급락V</th><th>상승장</th><th>횡보장</th><th>턴오버</th><th>평균</th></tr>
<tr><td>VOL+REV_RSI+REV_BB (동일가중·기본 파라미터)</td>{_vec_cells(ref)}<td>{ref_mean * 100:+.1f}</td></tr></table>
<p class="dim">점수 = 그 국면에서 라이벌 성실이(DCA) 대비 얼마나 나았나 ×100. 양수 = 성실이보다 강함.</p>

<h2>라벨 트레이더 4명 (배포 후보 라인업)</h2>
<table><tr><th>라벨</th><th>trial</th><th>하락장</th><th>회복장</th><th>급락V</th><th>상승장</th><th>횡보장</th><th>턴오버</th><th>평균</th><th>구성 (가중치/파라미터)</th></tr>
{label_rows}</table>

<h2>Pareto 지도</h2>
<div class="charts">
 <div>{chart1}<p class="dim">보험 성능(하락장) ↔ 보험료(상승장) 트레이드오프.
   오른쪽 위가 이상향이지만 거기엔 아무도 없다(만능 없음).</p></div>
 <div>{chart2}<p class="dim">매매 많이 할수록(오른쪽) 평균 점수가 좋아지는가?
   파란 점 = 필터 통과 후보.</p></div>
</div>

<h2>하드 필터 스윕 (턴오버 ≤ 0.10 고정)</h2>
<table><tr><th>조건</th><th>통과</th></tr>{sweep_rows}</table>
<p class="dim">"전 국면 ≥ 0"(모든 체육관에서 성실이 승) = 0명 - 트레이더 3000명 중에도 성실이를 전 국면에서 이기는 만능은 없다.</p>

<h2>필터 통과 트레이더 {len(summary['passed'])}명 전체 (평균 점수 순)</h2>
<table><tr><th>trial</th><th>하락장</th><th>회복장</th><th>급락V</th><th>상승장</th><th>횡보장</th><th>턴오버</th><th>평균</th><th>가중치</th></tr>
{passed_rows}</table>

<h2>읽을 때 주의</h2>
<div class="warn">
<b>전부 인샘플 점수다.</b> 훈련 체육관(이미 본 시험지) 성적이므로 "우승 후보"가 아니라
"본선 진출자"로 읽을 것. 다음 검증: ① OOS 리그 본선(처음 보는 미래) ② 합성 스트레스
③ 봉인 hold-out(post-COVID, 최후 1회).
</div>

<footer>PocketQuant · python -m app.lab.report_nsga3 로 재생성 ·
데이터: optuna_pocketquant.db (sqlite)</footer>
</body></html>"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html_doc, encoding="utf-8")
    print(f"리포트 저장: {OUT}")


if __name__ == "__main__":
    main()
