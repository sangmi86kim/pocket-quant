"""v2 리그 재경기 — 교실별 top30 분포 + 기준선.

사천왕 hold-out은 호출하지 않는다. 아레나는 세 곳만 본다:
  1) 공식 6체육관 exam 합산
  2) victory_road OOS 11년 합산
  3) battle_frontier 전천후 200세계 평균잔고

출력은 app/league/results/에 저장한다.
"""
import json
from html import escape
from pathlib import Path

import numpy as np

from app.academy.exam import all_gyms
from app.academy.exam.grade import evaluate_balances
from app.academy.training.candidate import decode_params
from app.league import battle_frontier as BF
from app.league import elite_four as EF
from app.league import victory_road as VR
from app.league.operations.npcs import (
    _BH_ENTRY_FRACS,
    _app_deletion_squad,
    _dca,
    _piggy_bank,
    _savings,
)
from app.pocket.battle import _score_position, fight_dca, terminal_balance
from app.pocket.signals import combine_positions, positions_with_params
from app.world.data_loader import get_prices, load_gyms


ROOT = Path(__file__).resolve().parents[3]
TOP30_PATH = ROOT / "app" / "academy" / "training" / "results" / "classroom_top30_20260615_v2.json"
RESULTS_DIR = ROOT / "app" / "league" / "results"
JSON_OUT = RESULTS_DIR / "season_v2_top30_league.json"
MD_OUT = RESULTS_DIR / "season_v2_top30_league.md"
SEED_KRW = 1_000_000
ARENAS = (
    ("exam", "공식 6체육관 합산"),
    ("oos", "OOS 11년 합산"),
    ("world", "평행세계 200세계 평균"),
    ("holdout", "사천왕 hold-out 합산"),
)


def _uses_default_signal_params(params: dict) -> bool:
    return all(str(k).startswith("w_") for k in params)


def _candidate_exam(weights: list[float], params: dict, gyms, dca,
                    base_positions: dict[str, list] | None = None) -> int:
    if not _uses_default_signal_params(params) or base_positions is None:
        rows = evaluate_balances(weights, params, gyms, dca, seed_krw=SEED_KRW)
        return int(sum(row["strat"] for row in rows.values()))
    total = 0
    for loaded in gyms:
        pos = combine_positions(base_positions[loaded.gym.name], weights)
        total += terminal_balance(_score_position(pos, loaded), SEED_KRW)
    return int(total)


def _candidate_oos(weights: list[float], params: dict, oos_loaded: dict[int, object],
                   base_positions: dict[int, list] | None = None) -> int:
    total = 0
    for year in VR.OOS_YEARS:
        loaded = oos_loaded[year]
        positions = (base_positions[year] if _uses_default_signal_params(params)
                     and base_positions is not None else
                     positions_with_params(loaded.prices, params))
        pos = combine_positions(positions, weights)
        total += terminal_balance(_score_position(pos, loaded), SEED_KRW)
    return int(total)


def _candidate_world(weights: list[float], params: dict, worlds: list[object],
                     base_positions: list[list] | None = None) -> int:
    balances = []
    for i, world in enumerate(worlds):
        positions = (base_positions[i] if _uses_default_signal_params(params)
                     and base_positions is not None else
                     positions_with_params(world.prices, params))
        pos = combine_positions(positions, weights)
        balances.append(terminal_balance(_score_position(pos, world), SEED_KRW))
    return int(np.mean(balances))


def _candidate_holdout(weights: list[float], params: dict, holdout_loaded: dict[str, object],
                       base_positions: dict[str, list] | None = None) -> int:
    total = 0
    for name, _start, _end in EF.ROUNDS:
        loaded = holdout_loaded[name]
        positions = (base_positions[name] if _uses_default_signal_params(params)
                     and base_positions is not None else
                     positions_with_params(loaded.prices))
        pos = combine_positions(positions, weights)
        total += terminal_balance(_score_position(pos, loaded), SEED_KRW)
    return int(total)


def _baseline_row(name: str, label: str, evaluator, exam_gyms, oos_loaded, worlds,
                  holdout_loaded) -> dict:
    exam = sum(evaluator(loaded, SEED_KRW)[1] for loaded in exam_gyms)
    oos = sum(evaluator(oos_loaded[year], SEED_KRW)[1] for year in VR.OOS_YEARS)
    world = int(np.mean([evaluator(world, SEED_KRW)[1] for world in worlds]))
    holdout = sum(evaluator(holdout_loaded[name], SEED_KRW)[1]
                  for name, _s, _e in EF.ROUNDS)
    return {
        "name": name,
        "group": name,
        "kind": "baseline",
        "label": label,
        "exam": int(exam),
        "oos": int(oos),
        "world": int(world),
        "holdout": int(holdout),
    }


def _app_deletion_member_row(frac: float, exam_gyms, oos_loaded, worlds, holdout_loaded,
                             idx: int) -> dict:
    # _app_deletion_squad는 300명 중앙값 대표를 돌려주는 함수라, 멤버별 분포는
    # 같은 랜덤 진입일 배열을 쓰되 각 멤버의 frac으로 직접 계산한다.
    from app.league.operations.npcs import TRADE_COST

    def term(loaded) -> float:
        prices = loaded.prices.loc[loaded.gym.start:loaded.gym.end]
        rets = prices.pct_change().dropna()
        n = len(rets)
        if n == 0:
            return float(SEED_KRW)
        arr = rets.to_numpy().copy()
        entry = min(int(frac * n), n - 1)
        arr[:entry] = 0.0
        arr[entry] -= TRADE_COST
        return SEED_KRW * float(np.prod(1.0 + arr))

    exam = sum(term(loaded) for loaded in exam_gyms)
    oos = sum(term(oos_loaded[year]) for year in VR.OOS_YEARS)
    world = int(np.mean([term(world) for world in worlds]))
    holdout = sum(term(holdout_loaded[name]) for name, _s, _e in EF.ROUNDS)
    return {
        "name": f"어플삭제단-{idx:03d}",
        "group": "어플삭제단",
        "kind": "baseline_member",
        "label": "B&H random entry",
        "exam": float(exam),
        "oos": float(oos),
        "world": int(world),
        "holdout": float(holdout),
    }


def _load_worlds() -> list[object]:
    prices = get_prices("QQQ", BF.DATA_START, BF.DATA_END)
    full_returns = prices.pct_change().dropna()
    rng = np.random.default_rng(BF.SEED)
    return [BF.make_world(full_returns, rng) for _ in range(BF.N_WORLDS_ALL)]


def _stats(rows: list[dict], key: str) -> dict:
    arr = np.array([row[key] for row in rows], dtype=float)
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def _summaries(rows: list[dict]) -> dict:
    groups = sorted(set(row["group"] for row in rows))
    out = {}
    for group in groups:
        grows = [row for row in rows if row["group"] == group]
        out[group] = {arena: _stats(grows, arena) for arena, _title in ARENAS}
    return out


def _boxplot_svg(payload: dict, arena: str, title: str) -> Path:
    # 세로 박스플랏: 그룹은 x축 칸, 잔고(만원)는 y축. 중앙값 내림차순으로 왼→오 배치.
    groups = sorted(
        payload["summary"],
        key=lambda g: payload["summary"][g][arena]["median"],
        reverse=True,
    )
    values = {
        group: np.array([row[arena] for row in payload["rows"]
                         if row["group"] == group], dtype=float) / 10000.0
        for group in groups
    }
    all_values = np.concatenate(list(values.values()))
    ymin = float(np.min(all_values))
    ymax = float(np.max(all_values))
    pad = max((ymax - ymin) * 0.08, 1.0)
    ymin -= pad
    ymax += pad

    n = len(groups)
    col_w = 118
    top = 76
    bottom = 84
    left = 88
    right = 36
    plot_h = 430
    width = left + right + col_w * n
    height = top + plot_h + bottom

    def sy(v: float) -> float:
        return top + (ymax - v) / (ymax - ymin) * plot_h

    colors = {
        "TPE": "#2f6fdd",
        "CMA-ES": "#2c9c69",
        "GP": "#9a6ad6",
        "NSGA": "#d48a1f",
        "어플삭제단": "#58606a",
        "성실이": "#c44545",
        "저축왕": "#4a8f9f",
        "돼지저금통": "#777777",
    }
    tick_count = 6
    ticks = np.linspace(ymin, ymax, tick_count)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="32" font-size="22" font-weight="700" fill="#1f2933">{escape(title)} boxplot</text>',
        f'<text x="{left}" y="56" font-size="13" fill="#52606d">unit: 만원(10,000 KRW), whisker=min/max, box=p25/p75, line=median</text>',
    ]
    axis_y = top + plot_h
    for tick in ticks:
        y = sy(float(tick))
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#eef2f7" stroke-width="1"/>')
        lines.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#52606d">{tick:.0f}</text>')
    lines.append(f'<line x1="{left}" y1="{top-18}" x2="{left}" y2="{axis_y}" stroke="#9aa5b1" stroke-width="1"/>')
    lines.append(f'<line x1="{left}" y1="{axis_y}" x2="{width-right}" y2="{axis_y}" stroke="#9aa5b1" stroke-width="1"/>')

    box_w = 46
    for i, group in enumerate(groups):
        arr = values[group]
        cx = left + i * col_w + col_w / 2
        q0, q1, q2, q3, q4 = np.percentile(arr, [0, 25, 50, 75, 100])
        color = colors.get(group, "#4b5563")
        # 그룹 이름 + 표본 수는 x축 아래
        lines.append(f'<text x="{cx:.1f}" y="{axis_y+24:.1f}" text-anchor="middle" font-size="14" fill="#1f2933">{escape(group)}</text>')
        lines.append(f'<text x="{cx:.1f}" y="{axis_y+42:.1f}" text-anchor="middle" font-size="11" fill="#7b8794">n={len(arr)}</text>')
        # 위스커(min~max) 세로선 + 캡
        lines.append(f'<line x1="{cx:.1f}" y1="{sy(q4):.1f}" x2="{cx:.1f}" y2="{sy(q0):.1f}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<line x1="{cx-9:.1f}" y1="{sy(q4):.1f}" x2="{cx+9:.1f}" y2="{sy(q4):.1f}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<line x1="{cx-9:.1f}" y1="{sy(q0):.1f}" x2="{cx+9:.1f}" y2="{sy(q0):.1f}" stroke="{color}" stroke-width="2"/>')
        if len(arr) == 1 or abs(q1 - q3) < 1e-9:
            lines.append(f'<circle cx="{cx:.1f}" cy="{sy(q2):.1f}" r="7" fill="{color}" opacity="0.88"/>')
        else:
            box_top = sy(q3)
            box_h = max(sy(q1) - sy(q3), 2)
            lines.append(f'<rect x="{cx-box_w/2:.1f}" y="{box_top:.1f}" width="{box_w}" height="{box_h:.1f}" fill="{color}" opacity="0.24" stroke="{color}" stroke-width="2"/>')
            lines.append(f'<line x1="{cx-box_w/2-2:.1f}" y1="{sy(q2):.1f}" x2="{cx+box_w/2+2:.1f}" y2="{sy(q2):.1f}" stroke="{color}" stroke-width="3"/>')
        # 중앙값 라벨은 위스커 위에
        lines.append(f'<text x="{cx:.1f}" y="{sy(q4)-10:.1f}" text-anchor="middle" font-size="11" fill="#52606d">{q2:.0f}</text>')

    lines.append("</svg>")
    path = RESULTS_DIR / f"season_v2_top30_league_{arena}_boxplot.svg"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_boxplots(payload: dict) -> dict[str, Path]:
    return {arena: _boxplot_svg(payload, arena, title)
            for arena, title in ARENAS}


def _write_markdown(payload: dict) -> None:
    charts = _write_boxplots(payload)
    lines = ["# Season v2 Top30 League", ""]
    lines.append(f"- top30: `{payload['top30_source']}`")
    lines.append(f"- worlds: {payload['worlds']} / seed={BF.SEED}")
    lines.append("- hold-out: not used")
    lines.append("")
    for arena, title in ARENAS:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"![{arena} boxplot]({charts[arena].name})")
        lines.append("")
        lines.append("| group | n | mean | std | min | p25 | median | p75 | max |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        ranked = sorted(
            payload["summary"].items(),
            key=lambda kv: kv[1][arena]["median"],
            reverse=True,
        )
        for group, stats in ranked:
            s = stats[arena]
            lines.append(
                f"| {group} | {s['n']} | {s['mean']:.0f} | {s['std']:.0f} | "
                f"{s['min']:.0f} | {s['p25']:.0f} | {s['median']:.0f} | "
                f"{s['p75']:.0f} | {s['max']:.0f} |"
            )
        lines.append("")
    lines.append("## Top candidates by arena")
    lines.append("")
    lines.append("| arena | name | group | balance | label |")
    lines.append("|---|---|---|---:|---|")
    for arena, _title in ARENAS:
        best = max(payload["rows"], key=lambda row, a=arena: row[a])
        lines.append(
            f"| {arena} | {best['name']} | {best['group']} | "
            f"{best[arena]:.0f} | {best.get('label', '')} |"
        )
    lines.append("")
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    top30 = json.loads(TOP30_PATH.read_text(encoding="utf-8"))

    exam_gyms = load_gyms(all_gyms())
    exam_dca = {loaded.gym.name: fight_dca(loaded) for loaded in exam_gyms}
    prices = get_prices(VR.TICKER, "1999-03-10", "2026-06-09")
    oos_loaded = {year: VR._loaded_window(prices, year) for year in VR.OOS_YEARS}
    holdout_loaded = {name: EF._loaded_window(prices, start, end)
                      for name, start, end in EF.ROUNDS}
    worlds = _load_worlds()
    exam_positions = {loaded.gym.name: positions_with_params(loaded.prices)
                      for loaded in exam_gyms}
    oos_positions = {year: positions_with_params(loaded.prices)
                     for year, loaded in oos_loaded.items()}
    holdout_positions = {name: positions_with_params(loaded.prices)
                         for name, loaded in holdout_loaded.items()}
    world_positions = [positions_with_params(world.prices) for world in worlds]

    rows = []
    for classroom in top30["classrooms"]:
        for item in classroom["topk"]:
            weights, params = decode_params(item["params"])
            rows.append({
                "name": f"{classroom['name']}-t{item['trial']}",
                "group": classroom["name"].replace("NSGA-III", "NSGA"),
                "kind": "candidate",
                "trial": item["trial"],
                "label": item["weights"],
                "exam": _candidate_exam(weights, params, exam_gyms, exam_dca,
                                        exam_positions),
                "oos": _candidate_oos(weights, params, oos_loaded,
                                      oos_positions),
                "world": _candidate_world(weights, params, worlds,
                                          world_positions),
                "holdout": _candidate_holdout(weights, params, holdout_loaded,
                                              holdout_positions),
            })

    for idx, frac in enumerate(_BH_ENTRY_FRACS, start=1):
        rows.append(_app_deletion_member_row(frac, exam_gyms, oos_loaded, worlds,
                                             holdout_loaded, idx))

    rows.extend([
        _baseline_row("성실이", "DCA", _dca, exam_gyms, oos_loaded, worlds, holdout_loaded),
        _baseline_row("저축왕", "연3%", _savings, exam_gyms, oos_loaded, worlds, holdout_loaded),
        _baseline_row("돼지저금통", "현금0%", _piggy_bank, exam_gyms, oos_loaded, worlds, holdout_loaded),
    ])

    payload = {
        "top30_source": str(TOP30_PATH),
        "worlds": len(worlds),
        "seed_krw": SEED_KRW,
        "rows": rows,
        "summary": _summaries(rows),
    }
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(payload)
    return payload


def main() -> None:
    payload = run()
    print(f"json={JSON_OUT}", flush=True)
    print(f"md={MD_OUT}", flush=True)
    for arena, _title in ARENAS:
        print(f"\n[{arena} median]", flush=True)
        ranked = sorted(
            payload["summary"].items(),
            key=lambda kv: kv[1][arena]["median"],
            reverse=True,
        )
        for group, stats in ranked:
            s = stats[arena]
            print(
                f"  {group:<8} n={s['n']:>3} median={s['median']:>10.0f} "
                f"p25={s['p25']:>10.0f} p75={s['p75']:>10.0f} std={s['std']:>9.0f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
