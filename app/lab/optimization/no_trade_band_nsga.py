"""NSGA 30명 No-trade band 민감도 — 거래 횟수와 turnover 진단."""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

from app.academy.exam import all_gyms, gym_key  # noqa: E402
from app.academy.training.candidate import decode_params  # noqa: E402
from app.pocket.battle import _dca_position, apply_no_trade_band  # noqa: E402
from app.pocket.signals import combine_positions, positions_with_params  # noqa: E402
from app.world.data_loader import load_gyms  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
NSGA_TOP30 = (ROOT / "app" / "academy" / "training" / "db"
              / "invalid_cost_model" / "20260622_115015"
              / "nsga_top30_20260622_115015_v2.json")
OUT_DIR = ROOT / "app" / "lab" / "reports" / "optimization" / "no_trade_band_nsga"
BANDS = [0.0, 0.01, 0.02, 0.05, 0.10]
EPS = 1e-12

GYM_LABEL = {
    "dotcom": "닷컴",
    "gfc": "리먼",
    "rebound": "회복",
    "crash_v": "코로나",
    "bull": "상승",
    "chop": "횡보",
}


def _uses_default_signal_params(params: dict) -> bool:
    return all(str(k).startswith("w_") for k in params)


def _window_turnover(turnover: pd.Series, loaded) -> pd.Series:
    mask = ((loaded.prices.index >= pd.Timestamp(loaded.gym.start))
            & (loaded.prices.index <= pd.Timestamp(loaded.gym.end)))
    return turnover[mask].dropna()


def _stats_from_position(position: pd.Series, loaded, band: float) -> dict:
    actual = apply_no_trade_band(position, band=band)
    turnover = _window_turnover(actual.shift(1).diff().abs(), loaded)
    traded = turnover[turnover > EPS]
    return {
        "trade_count": int((turnover > EPS).sum()),
        "days": int(len(turnover)),
        "trade_day_rate": float((turnover > EPS).mean()) if len(turnover) else 0.0,
        "daily_turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "total_turnover": float(turnover.sum()) if len(turnover) else 0.0,
        "trade_size_when_traded": float(traded.mean()) if len(traded) else 0.0,
    }


def _dca_stats(loaded) -> dict:
    turnover = _window_turnover(_dca_position(loaded).shift(1).diff().abs(), loaded)
    traded = turnover[turnover > EPS]
    return {
        "trade_count": int((turnover > EPS).sum()),
        "days": int(len(turnover)),
        "trade_day_rate": float((turnover > EPS).mean()) if len(turnover) else 0.0,
        "daily_turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "total_turnover": float(turnover.sum()) if len(turnover) else 0.0,
        "trade_size_when_traded": float(traded.mean()) if len(traded) else 0.0,
    }


def _load_rows() -> tuple[list[dict], dict]:
    payload = json.loads(NSGA_TOP30.read_text(encoding="utf-8"))
    items = payload["classrooms"][0]["topk"]
    gyms = load_gyms(all_gyms())
    base_positions = {lg.gym.name: positions_with_params(lg.prices) for lg in gyms}
    dca = {gym_key(lg.gym.name): _dca_stats(lg) for lg in gyms}

    rows = []
    for item in items:
        weights, params = decode_params(item["params"])
        for lg in gyms:
            positions = (base_positions[lg.gym.name]
                         if _uses_default_signal_params(params)
                         else positions_with_params(lg.prices, params))
            target = combine_positions(positions, weights)
            gkey = gym_key(lg.gym.name)
            for band in BANDS:
                stat = _stats_from_position(target, lg, band)
                rows.append({
                    "trial": item["trial"],
                    "gym": gkey,
                    "gym_label": GYM_LABEL.get(gkey, gkey),
                    "band": band,
                    **stat,
                })
    return rows, dca


def _boxplot(rows: list[dict], dca: dict, metric: str, title: str,
             ylabel: str, filename: str, as_pct: bool = False) -> Path:
    gyms = list(GYM_LABEL)
    bands = BANDS
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
    for ax, gkey in zip(axes.flat, gyms):
        data = []
        labels = []
        for band in bands:
            vals = [r[metric] for r in rows if r["gym"] == gkey and r["band"] == band]
            if as_pct:
                vals = [v * 100 for v in vals]
            data.append(vals)
            labels.append(f"{band:.0%}")
        ax.boxplot(data, tick_labels=labels, patch_artist=True,
                   boxprops=dict(facecolor="#d7e8f7", edgecolor="#2f6f9f"),
                   medianprops=dict(color="#1f2933", linewidth=1.5),
                   whiskerprops=dict(color="#52606d"),
                   capprops=dict(color="#52606d"))
        dca_value = dca[gkey][metric]
        if as_pct:
            dca_value *= 100
        ax.axhline(dca_value, color="#c0392b", linestyle="--", linewidth=1.3,
                   label="성실이")
        ax.set_title(GYM_LABEL[gkey], fontsize=12, fontweight="bold")
        ax.grid(axis="y", color="#eef2f7")
        ax.set_xlabel("No-trade band")
        ax.set_ylabel(ylabel)
    axes.flat[0].legend(loc="upper right")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = OUT_DIR / filename
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _summary_table(rows: list[dict], dca: dict) -> str:
    lines = []
    lines.append("| 체육관 | band | 거래횟수 median | 성실이 거래횟수 | turnover median | 성실이 대비 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for gkey in GYM_LABEL:
        dca_turn = dca[gkey]["daily_turnover"]
        for band in BANDS:
            sub = [r for r in rows if r["gym"] == gkey and r["band"] == band]
            count_med = float(np.median([r["trade_count"] for r in sub]))
            turn_med = float(np.median([r["daily_turnover"] for r in sub]))
            ratio = turn_med / dca_turn if dca_turn > 0 else float("nan")
            lines.append(
                f"| {GYM_LABEL[gkey]} | {band:.0%} | {count_med:.0f} | "
                f"{dca[gkey]['trade_count']} | {turn_med:.3%} | {ratio:.1f}x |"
            )
    return "\n".join(lines)


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, dca = _load_rows()
    trade_count_png = _boxplot(
        rows, dca, "trade_count",
        "NSGA 30명 No-trade band별 거래 횟수 — 점선=성실이",
        "거래 횟수", "nsga_no_trade_band_trade_count.png",
    )
    turnover_png = _boxplot(
        rows, dca, "daily_turnover",
        "NSGA 30명 No-trade band별 일평균 turnover — 점선=성실이",
        "일평균 turnover (%)", "nsga_no_trade_band_turnover.png",
        as_pct=True,
    )

    md = OUT_DIR / "nsga_no_trade_band_summary.md"
    md.write_text(
        "# NSGA 30명 No-trade band 거래량 진단\n\n"
        f"- source: `{NSGA_TOP30}`\n"
        "- 비용 모델 미완 학습 산출물 기반이므로 선발/판정용이 아니라 거래량 진단용이다.\n\n"
        f"![거래 횟수]({trade_count_png.name})\n\n"
        f"![일평균 turnover]({turnover_png.name})\n\n"
        "## 요약표\n\n"
        f"{_summary_table(rows, dca)}\n",
        encoding="utf-8",
    )
    print(md)
    print(trade_count_png)
    print(turnover_png)


if __name__ == "__main__":
    run()
