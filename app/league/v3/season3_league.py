"""시즌3 리그 (v3) — 최종 topk 65명 본경기.

[학교 ↔ 리그 분리]
- exam(실QQQ 6체육관 졸업시험)은 **학교 졸업 산출물**로 분리됐다
  (`app/academy/exam/graduate.py`, 진단 전용). 리그는 더 이상 exam을 채점하지 않는다.
- 시즌3 리그는 실데이터 두 관문만 본다:
    ① OOS 11년       (victory_road) — 평시 중심
    ② 사천왕 hold-out (elite_four)   — 최종 판정
  battle_frontier(평행세계 200)는 시즌3 리그에서 일단 뺀다.

[출전 범위]
각 교실의 최종 `topk`만 출전한다. phase1.topk는 졸업 진단 비교용이다.
CMA-ES 30명 + GP 5명 + NSGA-III 30명 = 65명.

[레포트 구조]
- 종합 비교: OOS 11년 + 사천왕 7라운드 전체 median.
- 관문별 비교: OOS와 사천왕을 별도 median 박스플랏으로 비교.
- 국면별 비교: 일 단위 국면 비율을 붙이고, OOS/사천왕을 분리해 라운드별 박스플랏을 그린다.

실행: .venv/Scripts/python.exe -m app.league.v3.season3_league
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402
from matplotlib.patches import Patch                    # noqa: E402

from app.academy.training.candidate import decode_params  # noqa: E402
from app.league import elite_four as EF                  # noqa: E402
from app.league import victory_road as VR                # noqa: E402
from app.league.operations.npcs import npc_graduates     # noqa: E402
from app.pocket.battle import _score_position, terminal_balance  # noqa: E402
from app.pocket.signals import SIGNAL_NAMES, combine_positions, positions_with_params  # noqa: E402
from app.world.data_loader import LoadedGym, get_prices  # noqa: E402
from app.world.regime import REGIME_LABELS, classify_daily  # noqa: E402


matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[3]
TRAIN_RESULTS = ROOT / "app" / "academy" / "training" / "results"
REPORTS_DIR = ROOT / "reports" / "포켓퀀트리그"
GRAPH_DIR = REPORTS_DIR / "graph" / "season3"

SEED_KRW = 1_000_000
STAGES = (
    ("oos", "OOS 11년", "victory_road"),
    ("holdout", "사천왕 hold-out", "elite_four"),
)

# None이면 최신 호환 top30.
TOP30_PATH: Path | None = None
JSON_OUT = REPORTS_DIR / "season3_league.json"
MD_OUT = REPORTS_DIR / "season3_league.md"

GROUP_COLORS = {
    "CMA-ES": "#2c9c69",
    "GP": "#7e5ca8",
    "NSGA": "#d48a1f",
    "성실이": "#c44545",
    "어플삭제단": "#58606a",
    "저축왕": "#4a8f9f",
    "돼지저금통": "#777777",
}


def _missing_weights(item: dict) -> list[str]:
    params = item.get("params", {})
    return [f"w_{name}" for name in SIGNAL_NAMES if f"w_{name}" not in params]


def _cost_model_ready(top30: dict) -> bool:
    meta = top30.get("cost_model")
    return isinstance(meta, dict) and meta.get("complete") is True


def _top30_compatible(path: Path) -> bool:
    try:
        top30 = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not _cost_model_ready(top30):
        return False
    for classroom in top30.get("classrooms", []):
        for item in classroom.get("topk") or []:
            if _missing_weights(item):
                return False
    return True


def _latest_top30() -> Path:
    if TOP30_PATH is not None:
        if not _top30_compatible(TOP30_PATH):
            raise ValueError(f"top30 파일 사용 불가: {TOP30_PATH}")
        return TOP30_PATH
    paths = sorted(TRAIN_RESULTS.glob("classroom_top30_*_v2.json"))
    compatible = [path for path in paths if _top30_compatible(path)]
    if not compatible:
        raise FileNotFoundError("season3 비용 모델과 14신호 스키마를 만족하는 top30 파일 없음")
    return compatible[-1]


def _group_name(name: str) -> str:
    return name.replace("NSGA-III", "NSGA")


def _load_candidates(top30: dict) -> list[dict]:
    out = []
    for classroom in top30.get("classrooms", []):
        group = _group_name(classroom["name"])
        topk = classroom.get("topk") or []
        for rank, item in enumerate(topk, start=1):
            missing = _missing_weights(item)
            if missing:
                raise ValueError(f"{group} 후보 가중치 누락: {missing}")
            weights, params = decode_params(item["params"])
            out.append({
                "name": f"{group}-t{item.get('trial', rank)}",
                "group": group,
                "kind": "candidate",
                "trial": item.get("trial"),
                "rank": rank,
                "weights": weights,
                "params": params,
            })
    if not out:
        raise ValueError("최종 topk 후보가 비어 있음")
    return out


def _candidate_balance(player: dict, loaded: LoadedGym) -> int:
    positions = positions_with_params(loaded.prices, player["params"])
    position = combine_positions(positions, player["weights"])
    return terminal_balance(_score_position(position, loaded), SEED_KRW)


def _baseline_balance(player: dict, loaded: LoadedGym) -> int:
    _returns, balance = player["evaluator"](loaded, SEED_KRW)
    return int(balance)


def _regime_profile(prices, start: str, end: str) -> dict:
    daily = classify_daily(prices)
    mask = (daily.index >= start) & (daily.index <= end)
    sub = daily[mask]
    if len(sub) == 0:
        return {
            "primary": "횡보장",
            "mix": "횡보장 100%",
            "counts": {"sideways": 0},
            "shares": {"sideways": 1.0},
        }
    counts = sub.value_counts().to_dict()
    shares = {key: float(value / len(sub)) for key, value in counts.items()}
    primary = max(shares, key=lambda key: shares[key])
    order = ["bull", "bear", "sideways", "volatile"]
    parts = [
        f"{REGIME_LABELS[key]} {shares[key] * 100:.0f}%"
        for key in order if key in shares
    ]
    return {
        "primary": REGIME_LABELS[primary],
        "mix": " / ".join(parts),
        "counts": {key: int(value) for key, value in counts.items()},
        "shares": shares,
    }


def _rounds(prices) -> list[dict]:
    out = []
    for year in VR.OOS_YEARS:
        loaded = VR._loaded_window(prices, year)
        out.append({
            "key": f"oos:{year}",
            "stage": "oos",
            "label": str(year),
            "loaded": loaded,
            "regime": _regime_profile(loaded.prices, loaded.gym.start, loaded.gym.end),
        })
    for name, start, end in EF.ROUNDS:
        loaded = EF._loaded_window(prices, start, end)
        out.append({
            "key": f"holdout:{name}",
            "stage": "holdout",
            "label": name,
            "loaded": loaded,
            "regime": _regime_profile(loaded.prices, start, end),
        })
    return out


def _stats(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    return {
        "n": int(len(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def _score_players(players: list[dict], rounds: list[dict]) -> list[dict]:
    rows = []
    for player in players:
        balances = {}
        for round_info in rounds:
            loaded = round_info["loaded"]
            if "evaluator" in player:
                balance = _baseline_balance(player, loaded)
            else:
                balance = _candidate_balance(player, loaded)
            balances[round_info["key"]] = balance
        oos = [balances[r["key"]] for r in rounds if r["stage"] == "oos"]
        holdout = [balances[r["key"]] for r in rounds if r["stage"] == "holdout"]
        rows.append({
            "name": player["name"],
            "group": player["group"],
            "kind": player["kind"],
            "trial": player.get("trial"),
            "rank": player.get("rank"),
            "overall": float(np.median(list(balances.values()))),
            "oos": float(np.median(oos)),
            "holdout": float(np.median(holdout)),
            "balances": balances,
        })
    return rows


def _summary(rows: list[dict]) -> dict:
    groups = sorted(set(row["group"] for row in rows))
    out = {}
    for group in groups:
        subset = [row for row in rows if row["group"] == group]
        out[group] = {
            key: _stats([row[key] for row in subset])
            for key in ("overall", "oos", "holdout")
        }
    return out


def _candidate_groups(rows: list[dict]) -> list[str]:
    return sorted({row["group"] for row in rows if row["kind"] == "candidate"})


def _baseline_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["kind"] == "baseline"]


def _plot_box(payload: dict, key: str, title: str, path: Path) -> None:
    groups = sorted(
        payload["summary"],
        key=lambda group: payload["summary"][group][key]["median"],
        reverse=True,
    )
    data = [
        [row[key] / 10000 for row in payload["rows"] if row["group"] == group]
        for group in groups
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, patch_artist=True, showmeans=True)
    for patch, group in zip(bp["boxes"], groups):
        patch.set_facecolor(GROUP_COLORS.get(group, "#777777"))
        patch.set_alpha(0.45)
    for i, vals in enumerate(data, start=1):
        ax.scatter([i] * len(vals), vals, s=22, color="#222", alpha=0.55, zorder=3)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([f"{group}\n(n={len(vals)})" for group, vals in zip(groups, data)])
    ax.set_ylabel("종료잔고 (만원)")
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_stage_by_round(payload: dict, stage: str, path: Path) -> None:
    rounds = [row for row in payload["rounds"] if row["stage"] == stage]
    rows = payload["rows"]
    candidate_groups = _candidate_groups(rows)
    baselines = _baseline_rows(rows)
    width = 0.22
    offsets = np.linspace(-width, width, len(candidate_groups))
    base_x = np.arange(1, len(rounds) + 1)

    fig, ax = plt.subplots(figsize=(15, 6))
    legend_handles = []
    for offset, group in zip(offsets, candidate_groups):
        data = [
            [row["balances"][r["key"]] / 10000
             for row in rows if row["group"] == group]
            for r in rounds
        ]
        positions = base_x + offset
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.18,
            patch_artist=True,
            showfliers=False,
        )
        color = GROUP_COLORS.get(group, "#777777")
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
        for line in bp["medians"]:
            line.set_color("#111111")
        legend_handles.append(Patch(facecolor=color, alpha=0.35, label=group))

    marker = {"성실이": "o", "저축왕": "s", "돼지저금통": "^", "어플삭제단": "D"}
    for row in baselines:
        ys = [row["balances"][r["key"]] / 10000 for r in rounds]
        ax.plot(
            base_x,
            ys,
            marker=marker.get(row["group"], "o"),
            ms=4,
            lw=1.4,
            alpha=0.85,
            label=row["group"],
        )

    labels = [f"{r['label']}\n{r['regime']['primary']}" for r in rounds]
    ax.set_xticks(base_x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("종료잔고 (만원)")
    title = "OOS 국면별 비교" if stage == "oos" else "사천왕 국면별 비교"
    ax.set_title(f"{title} — 교실별 boxplot + baseline line")
    ax.grid(alpha=0.25, axis="y")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=legend_handles + handles, fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _candidate_median(rows: list[dict], group: str, round_key: str) -> float:
    values = [
        row["balances"][round_key]
        for row in rows if row["group"] == group and row["kind"] == "candidate"
    ]
    return float(np.median(values))


def _write_round_table(lines: list[str], payload: dict, stage: str) -> None:
    groups = _candidate_groups(payload["rows"])
    group_cols = " | ".join(f"{group} median" for group in groups)
    lines.append(f"| round | daily regime mix | {group_cols} | best baseline | balance |")
    lines.append("|---|---|" + "---:|" * len(groups) + "---|---:|")
    for round_info in [r for r in payload["rounds"] if r["stage"] == stage]:
        key = round_info["key"]
        baseline_values = [
            (row["group"], row["balances"][key])
            for row in payload["rows"] if row["kind"] == "baseline"
        ]
        best_name, best_balance = max(baseline_values, key=lambda item: item[1])
        medians = " | ".join(
            f"{_candidate_median(payload['rows'], group, key):.0f}"
            for group in groups
        )
        lines.append(
            f"| {round_info['label']} | {round_info['regime']['mix']} | "
            f"{medians} | {best_name} | {best_balance:.0f} |"
        )


def _write_markdown(payload: dict, charts: dict[str, Path]) -> None:
    def chart_ref(key: str) -> str:
        return charts[key].relative_to(MD_OUT.parent).as_posix()

    lines = [
        "# Season3 League",
        "",
        f"- top30: `{payload['top30_source']}`",
        f"- cost_model: `{payload['cost_model']['version']}`",
        f"- contestants: candidates {payload['candidate_count']}명 + baseline {payload['baseline_count']}명",
        "- order: OOS 11년 → 사천왕 hold-out",
        "",
        "## 종합 비교",
        "",
        f"![overall]({chart_ref('overall')})",
        "",
        "| group | n | overall median | oos median | holdout median |",
        "|---|---:|---:|---:|---:|",
    ]
    ranked = sorted(
        payload["summary"].items(),
        key=lambda kv: kv[1]["overall"]["median"],
        reverse=True,
    )
    for group, stats in ranked:
        lines.append(
            f"| {group} | {stats['overall']['n']} | "
            f"{stats['overall']['median']:.0f} | {stats['oos']['median']:.0f} | "
            f"{stats['holdout']['median']:.0f} |"
        )
    lines.extend([
        "",
        "## 관문별 비교",
        "",
        f"![oos]({chart_ref('oos')})",
        "",
        f"![holdout]({chart_ref('holdout')})",
        "",
        "## 국면별 비교 — OOS",
        "",
        f"![oos by round]({chart_ref('by_round_oos')})",
        "",
    ])
    _write_round_table(lines, payload, "oos")
    lines.extend([
        "",
        "## 국면별 비교 — 사천왕",
        "",
        f"![holdout by round]({chart_ref('by_round_holdout')})",
        "",
    ])
    _write_round_table(lines, payload, "holdout")
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    top30_path = _latest_top30()
    top30 = json.loads(top30_path.read_text(encoding="utf-8"))
    candidates = _load_candidates(top30)
    baselines = npc_graduates()
    for baseline in baselines:
        baseline["group"] = baseline["name"]
        baseline["kind"] = "baseline"

    prices = get_prices(VR.TICKER, "1999-03-10", EF.DATA_END)
    rounds = _rounds(prices)
    rows = _score_players(candidates + baselines, rounds)
    payload = {
        "top30_source": str(top30_path),
        "stamp": top30.get("stamp"),
        "cost_model": top30["cost_model"],
        "seed_krw": SEED_KRW,
        "stages": STAGES,
        "candidate_count": len(candidates),
        "baseline_count": len(baselines),
        "rounds": [{k: v for k, v in r.items() if k != "loaded"} for r in rounds],
        "rows": rows,
        "summary": _summary(rows),
    }
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    charts = {
        "overall": GRAPH_DIR / "season3_league_overall_boxplot.svg",
        "oos": GRAPH_DIR / "season3_league_oos_boxplot.svg",
        "holdout": GRAPH_DIR / "season3_league_holdout_boxplot.svg",
        "by_round_oos": GRAPH_DIR / "season3_league_oos_by_round_boxplot.svg",
        "by_round_holdout": GRAPH_DIR / "season3_league_holdout_by_round_boxplot.svg",
    }
    _plot_box(payload, "overall", "종합 비교 — 전체 18라운드 median", charts["overall"])
    _plot_box(payload, "oos", "OOS 11년 비교 — 연도별 median", charts["oos"])
    _plot_box(payload, "holdout", "사천왕 hold-out 비교 — 라운드별 median", charts["holdout"])
    _plot_stage_by_round(payload, "oos", charts["by_round_oos"])
    _plot_stage_by_round(payload, "holdout", charts["by_round_holdout"])
    _write_markdown(payload, charts)
    return payload


def main() -> None:
    payload = run()
    print(f"json={JSON_OUT}", flush=True)
    print(f"md={MD_OUT}", flush=True)
    print("\n[overall median]", flush=True)
    ranked = sorted(
        payload["summary"].items(),
        key=lambda kv: kv[1]["overall"]["median"],
        reverse=True,
    )
    for group, stats in ranked:
        print(
            f"  {group:<8} n={stats['overall']['n']:>2} "
            f"overall={stats['overall']['median']/10000:>8.1f}만 "
            f"oos={stats['oos']['median']/10000:>8.1f}만 "
            f"holdout={stats['holdout']['median']/10000:>8.1f}만",
            flush=True,
        )


if __name__ == "__main__":
    main()
