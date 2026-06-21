"""졸업 시험 — 학교 top30을 실QQQ 6체육관에 응시시켜 졸업 성적표(md + 박스플랏)를 낸다.

[위치] 졸업시험은 **진단 전용(아이큐 테스트)** 이다 — 선발/게이트가 아니다. 선발은 학교
합성장 median 점수로 끝났고(training), 이 산출물은 "그 졸업생들이 진짜 시장 6국면에서
어떻게 서는가"를 눈으로 보는 성적표일 뿐이다. 진짜 줄세움은 리그(OOS·평행세계)에서 한다.

[입력] training이 뱉은 top30 json (`classroom_top30_*_v2.json`). 거기 담긴 trial별
`params`를 디코딩해 실QQQ 6체육관에 다시 돌린다 — stale할 수 있는 점수 키엔 안 의존한다.

[잣대] 후보 1명의 졸업 점수 = 6체육관 종료잔고의 **중앙값**(median 목적과 정합).
반(교실)별로 그 점수의 분포를 박스플랏으로 그리고, 성실이(DCA)를 기준선으로 같이 둔다.

[실행] .venv/Scripts/python.exe -m app.academy.exam.graduate
"""
import json
from html import escape
from pathlib import Path

import numpy as np

from app.academy.exam import all_gyms, gym_key
from app.academy.exam.grade import evaluate_balances
from app.academy.training.candidate import decode_params
from app.pocket.battle import fight_dca
from app.world.data_loader import load_gyms


ROOT = Path(__file__).resolve().parents[3]
TRAIN_RESULTS = ROOT / "app" / "academy" / "training" / "results"
REPORTS_DIR = ROOT / "app" / "academy" / "reports"
SEED_KRW = 1_000_000

# 실행 옵션은 모듈 상수로 둔다(argparse 없이). None이면 가장 최근 top30 파일을 자동 선택.
TOP30_PATH: Path | None = None

# 체육관키 → 성적표 열에 쓸 짧은 이름(콘솔 인코딩 영향 없는 표시용 라벨).
GYM_LABEL = {
    "dotcom": "닷컴",
    "gfc": "리먼",
    "rebound": "회복",
    "crash_v": "코로나",
    "bull": "상승",
    "chop": "횡보",
}
GROUP_COLORS = {
    "TPE": "#2f6fdd",
    "CMA-ES": "#2c9c69",
    "GP": "#9a6ad6",
    "NSGA": "#d48a1f",
    "성실이": "#c44545",
}


def _latest_top30() -> Path:
    if TOP30_PATH is not None:
        return TOP30_PATH
    cands = sorted(TRAIN_RESULTS.glob("classroom_top30_*_v2.json"))
    if not cands:
        raise FileNotFoundError(f"top30 파일 없음: {TRAIN_RESULTS}/classroom_top30_*_v2.json")
    return cands[-1]


def _candidate_gym_balances(item: dict, gyms, dca) -> dict[str, float]:
    """후보 1명을 6체육관에 응시 → {체육관키: 전략 종료잔고}."""
    weights, params = decode_params(item["params"])
    balances = evaluate_balances(weights, params, gyms, dca, seed_krw=SEED_KRW)
    return {gym_key(name): row["strat"] for name, row in balances.items()}


def _stats(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    return {
        "n": int(len(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def run() -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    top30_path = _latest_top30()
    top30 = json.loads(top30_path.read_text(encoding="utf-8"))
    stamp = top30.get("stamp", "latest")

    gyms = load_gyms(all_gyms())
    dca = {lg.gym.name: fight_dca(lg) for lg in gyms}
    gym_keys = [gym_key(lg.gym.name) for lg in gyms]   # all_gyms 순서 유지

    # 성실이(DCA)는 후보와 무관하게 체육관마다 한 값 — 한 번만 잰다.
    from app.pocket.battle import terminal_balance
    dca_by_gym = {gym_key(lg.gym.name): terminal_balance(dca[lg.gym.name], SEED_KRW)
                  for lg in gyms}

    classrooms = []
    for classroom in top30["classrooms"]:
        group = classroom["name"].replace("NSGA-III", "NSGA")
        members = []
        for item in classroom["topk"]:
            per_gym = _candidate_gym_balances(item, gyms, dca)
            members.append({
                "trial": item.get("trial"),
                "per_gym": per_gym,
                "score": float(np.median([per_gym[k] for k in gym_keys])),
            })
        classrooms.append({"group": group, "members": members})

    payload = {
        "stamp": stamp,
        "top30_source": str(top30_path),
        "seed_krw": SEED_KRW,
        "gym_keys": gym_keys,
        "dca_by_gym": dca_by_gym,
        "dca_score": float(np.median([dca_by_gym[k] for k in gym_keys])),
        "classrooms": classrooms,
    }
    svg_path = _boxplot_svg(payload, stamp)
    _write_markdown(payload, stamp, svg_path)
    return payload


def _boxplot_svg(payload: dict, stamp: str) -> Path:
    # 세로 박스플랏: x축=반, y축=후보별 6체육관 median 종료잔고(만원). 성실이는 기준 점.
    groups = [c["group"] for c in payload["classrooms"]]
    values = {c["group"]: np.array([m["score"] for m in c["members"]], dtype=float) / 10000.0
              for c in payload["classrooms"]}
    dca_v = payload["dca_score"] / 10000.0
    groups = groups + ["성실이"]
    values["성실이"] = np.array([dca_v], dtype=float)

    all_v = np.concatenate(list(values.values()))
    ymin, ymax = float(np.min(all_v)), float(np.max(all_v))
    pad = max((ymax - ymin) * 0.10, 1.0)
    ymin -= pad
    ymax += pad

    n = len(groups)
    col_w, top, bottom, left, right, plot_h = 118, 80, 78, 88, 36, 420
    width = left + right + col_w * n
    height = top + plot_h + bottom

    def sy(v: float) -> float:
        return top + (ymax - v) / (ymax - ymin) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-size="22" font-weight="700" fill="#1f2933">졸업 시험 — 실QQQ 6체육관 (반별 분포)</text>',
        f'<text x="{left}" y="58" font-size="13" fill="#52606d">unit: 만원, 점수=후보별 6체육관 median 종료잔고 · box=p25/p75 · line=median · 진단 전용</text>',
    ]
    axis_y = top + plot_h
    for tick in np.linspace(ymin, ymax, 6):
        y = sy(float(tick))
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#eef2f7" stroke-width="1"/>')
        lines.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#52606d">{tick:.0f}</text>')
    # 성실이 기준선 — 가로 점선
    yd = sy(dca_v)
    lines.append(f'<line x1="{left}" y1="{yd:.1f}" x2="{width-right}" y2="{yd:.1f}" stroke="#c44545" stroke-width="1.5" stroke-dasharray="6 4" opacity="0.7"/>')
    lines.append(f'<line x1="{left}" y1="{top-18}" x2="{left}" y2="{axis_y}" stroke="#9aa5b1" stroke-width="1"/>')
    lines.append(f'<line x1="{left}" y1="{axis_y}" x2="{width-right}" y2="{axis_y}" stroke="#9aa5b1" stroke-width="1"/>')

    box_w = 46
    for i, group in enumerate(groups):
        arr = values[group]
        cx = left + i * col_w + col_w / 2
        q0, q1, q2, q3, q4 = np.percentile(arr, [0, 25, 50, 75, 100])
        color = GROUP_COLORS.get(group, "#4b5563")
        lines.append(f'<text x="{cx:.1f}" y="{axis_y+24:.1f}" text-anchor="middle" font-size="14" fill="#1f2933">{escape(group)}</text>')
        lines.append(f'<text x="{cx:.1f}" y="{axis_y+42:.1f}" text-anchor="middle" font-size="11" fill="#7b8794">n={len(arr)}</text>')
        lines.append(f'<line x1="{cx:.1f}" y1="{sy(q4):.1f}" x2="{cx:.1f}" y2="{sy(q0):.1f}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<line x1="{cx-9:.1f}" y1="{sy(q4):.1f}" x2="{cx+9:.1f}" y2="{sy(q4):.1f}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<line x1="{cx-9:.1f}" y1="{sy(q0):.1f}" x2="{cx+9:.1f}" y2="{sy(q0):.1f}" stroke="{color}" stroke-width="2"/>')
        if len(arr) == 1 or abs(q1 - q3) < 1e-9:
            lines.append(f'<circle cx="{cx:.1f}" cy="{sy(q2):.1f}" r="7" fill="{color}" opacity="0.88"/>')
        else:
            box_h = max(sy(q1) - sy(q3), 2)
            lines.append(f'<rect x="{cx-box_w/2:.1f}" y="{sy(q3):.1f}" width="{box_w}" height="{box_h:.1f}" fill="{color}" opacity="0.24" stroke="{color}" stroke-width="2"/>')
            lines.append(f'<line x1="{cx-box_w/2-2:.1f}" y1="{sy(q2):.1f}" x2="{cx+box_w/2+2:.1f}" y2="{sy(q2):.1f}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{cx:.1f}" y="{sy(q4)-10:.1f}" text-anchor="middle" font-size="11" fill="#52606d">{q2:.0f}</text>')

    lines.append("</svg>")
    path = REPORTS_DIR / f"graduation_{stamp}_boxplot.svg"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_markdown(payload: dict, stamp: str, svg_path: Path) -> None:
    gym_keys = payload["gym_keys"]
    headers = [GYM_LABEL.get(k, k) for k in gym_keys]
    lines = [
        "# 🎓 졸업 시험 성적표 — 실QQQ 6체육관",
        "",
        f"> **진단 전용** (선발 아님 — 선발은 학교 합성장 median 점수로 끝). "
        f"top30 출처: `{Path(payload['top30_source']).name}` · 시드 100만원 · 잣대=종료잔고 median",
        f"> stamp: {stamp}",
        "",
        f"![졸업 boxplot]({svg_path.name})",
        "",
        "## 반별 종합 (후보별 6체육관 median 종료잔고 분포, 만원)",
        "",
        "| 반 | n | median | p25 | p75 | min | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    ranked = sorted(
        payload["classrooms"],
        key=lambda c: float(np.median([m["score"] for m in c["members"]])),
        reverse=True,
    )
    for c in ranked:
        s = _stats([m["score"] for m in c["members"]])
        lines.append(
            f"| {c['group']} | {s['n']} | {s['median']/10000:.0f} | {s['p25']/10000:.0f} | "
            f"{s['p75']/10000:.0f} | {s['min']/10000:.0f} | {s['max']/10000:.0f} |"
        )
    lines.append(
        f"| 성실이(DCA) | 1 | {payload['dca_score']/10000:.0f} | · | · | · | · |"
    )
    lines += [
        "",
        "## 반별 × 체육관 (반 median 종료잔고, 만원) — 약점 진단",
        "",
        "| 반 | " + " | ".join(headers) + " |",
        "|---|" + "---:|" * len(headers),
    ]
    for c in ranked:
        cells = []
        for k in gym_keys:
            med = float(np.median([m["per_gym"][k] for m in c["members"]]))
            cells.append(f"{med/10000:.0f}")
        lines.append(f"| {c['group']} | " + " | ".join(cells) + " |")
    dca_cells = [f"{payload['dca_by_gym'][k]/10000:.0f}" for k in gym_keys]
    lines.append("| 성실이(DCA) | " + " | ".join(dca_cells) + " |")
    lines.append("")
    lines.append("> 반 median이 성실이보다 낮은 체육관 = 그 반의 약점 과목.")
    lines.append("")

    md_path = REPORTS_DIR / f"graduation_{stamp}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    payload = run()
    print(f"reports_dir={REPORTS_DIR}", flush=True)
    print(f"graduation_{payload['stamp']}.md", flush=True)
    ranked = sorted(
        payload["classrooms"],
        key=lambda c: float(np.median([m["score"] for m in c["members"]])),
        reverse=True,
    )
    print("\n[졸업 종합 median, 만원]", flush=True)
    for c in ranked:
        med = float(np.median([m["score"] for m in c["members"]])) / 10000
        print(f"  {c['group']:<8} n={len(c['members']):>3} median={med:>8.0f}", flush=True)
    print(f"  {'성실이':<8} n=  1 median={payload['dca_score']/10000:>8.0f}", flush=True)


if __name__ == "__main__":
    main()
