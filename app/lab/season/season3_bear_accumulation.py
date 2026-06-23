"""시즌3 하락장 평단가 분석 — 하락장에서 모은 물량이 다음 상승장에서 먹혔는지 본다."""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

from app.academy.training.candidate import decode_params  # noqa: E402
from app.league import elite_four as EF  # noqa: E402
from app.league import victory_road as VR  # noqa: E402
from app.pocket.battle import _dca_position, apply_no_trade_band  # noqa: E402
from app.pocket.signals import combine_positions, positions_with_params  # noqa: E402
from app.world.data_loader import get_prices  # noqa: E402
from app.world.regime import classify_daily  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
TOP30_PATH = (ROOT / "app" / "academy" / "training" / "db"
              / "classroom_top30_20260622_195214_v2.json")
OUT_DIR = ROOT / "app" / "lab" / "reports" / "season" / "season3_bear_accumulation"
SEED_KRW = 1_000_000
EPS = 1e-12


def _group_name(name: str) -> str:
    return name.replace("NSGA-III", "NSGA")


def _load_candidates() -> list[dict]:
    payload = json.loads(TOP30_PATH.read_text(encoding="utf-8"))
    out = []
    for classroom in payload["classrooms"]:
        base = _group_name(classroom["name"])
        for phase, topk in [
            ("1차", classroom.get("phase1", {}).get("topk") or []),
            ("보충", classroom.get("topk") or []),
        ]:
            group = f"{base}-{phase}"
            for rank, item in enumerate(topk, start=1):
                weights, params = decode_params(item["params"])
                out.append({
                    "name": f"{group}-t{item.get('trial', rank)}",
                    "group": group,
                    "weights": weights,
                    "params": params,
                })
    return out


def _league_windows(prices: pd.Series) -> list[dict]:
    out = []
    for year in VR.OOS_YEARS:
        loaded = VR._loaded_window(prices, year)
        out.append({
            "label": str(year),
            "stage": "빅토리 로드 (OOS)",
            "prices": loaded.prices,
            "start": loaded.gym.start,
            "end": loaded.gym.end,
        })
    for name, start, end in EF.ROUNDS:
        loaded = EF._loaded_window(prices, start, end)
        out.append({
            "label": name,
            "stage": "사천왕",
            "prices": loaded.prices,
            "start": start,
            "end": end,
        })
    return out


def _position_for(candidate: dict, prices: pd.Series) -> pd.Series:
    positions = positions_with_params(prices, candidate["params"])
    target = combine_positions(positions, candidate["weights"])
    return apply_no_trade_band(target).shift(1)


def _dca_position_for(prices: pd.Series, start: str, end: str) -> pd.Series:
    from app.pocket.models import Gym
    from app.world.data_loader import LoadedGym
    gym = Gym("DCA window", difficulty=0, volatility=0, ticker=VR.TICKER,
              start=start, end=end)
    return _dca_position(LoadedGym(gym=gym, prices=prices)).shift(1)


def _next_bull_return(prices: pd.Series, daily: pd.Series, bear_end: pd.Timestamp,
                      avg_cost: float) -> float | None:
    future = daily[daily.index > bear_end]
    if len(future) == 0:
        return None
    bull = future[future == "bull"]
    if len(bull) == 0:
        return None
    end_price = float(prices.reindex([bull.index[-1]], method="ffill").iloc[0])
    return end_price / avg_cost - 1.0


def _analyze_one(name: str, group: str, prices: pd.Series, position: pd.Series,
                 daily: pd.Series, start: str, end: str) -> dict | None:
    mask = (prices.index >= start) & (prices.index <= end)
    window_prices = prices[mask]
    window_pos = position.reindex(window_prices.index).ffill().fillna(0.0)
    window_daily = daily.reindex(window_prices.index).dropna()
    bear_days = window_daily[window_daily == "bear"].index
    if len(bear_days) == 0:
        return None

    buy = window_pos.diff().clip(lower=0.0).reindex(bear_days).fillna(0.0)
    buy = buy[buy > EPS]
    if len(buy) == 0:
        return None
    buy_prices = window_prices.reindex(buy.index, method="ffill")
    shares = buy / buy_prices
    avg_cost = float(buy.sum() / shares.sum())
    bear_end = pd.Timestamp(bear_days[-1])
    next_bull = _next_bull_return(window_prices, window_daily, bear_end, avg_cost)
    end_pos = float(window_pos.reindex([bear_end], method="ffill").iloc[0])
    return {
        "name": name,
        "group": group,
        "bear_buy_amount": float(buy.sum()),
        "bear_buy_count": int(len(buy)),
        "bear_avg_cost": avg_cost,
        "bear_end_position": end_pos,
        "next_bull_return": next_bull,
    }


def _rows() -> list[dict]:
    prices = get_prices(VR.TICKER, "1999-03-10", EF.DATA_END)
    candidates = _load_candidates()
    windows = _league_windows(prices)
    rows = []
    for window in windows:
        daily = classify_daily(window["prices"])
        for candidate in candidates:
            position = _position_for(candidate, window["prices"])
            row = _analyze_one(candidate["name"], candidate["group"], window["prices"],
                               position, daily, window["start"], window["end"])
            if row is not None:
                row.update({"stage": window["stage"], "window": window["label"]})
                rows.append(row)
        dca_pos = _dca_position_for(window["prices"], window["start"], window["end"])
        dca_row = _analyze_one("성실이", "성실이", window["prices"], dca_pos,
                               daily, window["start"], window["end"])
        if dca_row is not None:
            dca_row.update({"stage": window["stage"], "window": window["label"]})
            rows.append(dca_row)
    return rows


def _stats(rows: list[dict], key: str) -> dict:
    vals = [row[key] for row in rows if row.get(key) is not None]
    arr = np.array(vals, dtype=float)
    return {
        "n": int(len(arr)),
        "median": float(np.median(arr)) if len(arr) else float("nan"),
        "p25": float(np.percentile(arr, 25)) if len(arr) else float("nan"),
        "p75": float(np.percentile(arr, 75)) if len(arr) else float("nan"),
    }


def _summary(rows: list[dict]) -> dict:
    groups = sorted(set(row["group"] for row in rows))
    out = {}
    for group in groups:
        subset = [row for row in rows if row["group"] == group]
        out[group] = {
            "avg_cost": _stats(subset, "bear_avg_cost"),
            "next_bull_return": _stats(subset, "next_bull_return"),
            "bear_end_position": _stats(subset, "bear_end_position"),
        }
    return out


def _plot(rows: list[dict], key: str, title: str, filename: str) -> None:
    groups = sorted(set(row["group"] for row in rows))
    data = [[row[key] for row in rows
             if row["group"] == group and row.get(key) is not None]
            for group in groups]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.boxplot(data, patch_artist=True, showmeans=True, showfliers=False)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([f"{group}\n(n={len(vals)})" for group, vals in zip(groups, data)],
                       rotation=30, ha="right", fontsize=9)
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=130)
    plt.close(fig)


def _write_report(rows: list[dict], summary: dict) -> None:
    lines = [
        "# 시즌3 하락장 평단가 분석",
        "",
        f"- top30: `{TOP30_PATH.name}`",
        "- 질문: 하락장에서 산 물량이 다음 상승장에서 얼마나 먹혔나?",
        "- 기준: 일 단위 Regime Scanner 라벨의 bear 일자 매수만 집계",
        "",
        "![next bull](season3_bear_accumulation_next_bull.png)",
        "",
        "![end position](season3_bear_accumulation_position.png)",
        "",
        "| group | n | bear avg cost median | next bull return median | bear end position median |",
        "|---|---:|---:|---:|---:|",
    ]
    ranked = sorted(summary.items(),
                    key=lambda item: item[1]["next_bull_return"]["median"],
                    reverse=True)
    for group, stats in ranked:
        lines.append(
            f"| {group} | {stats['next_bull_return']['n']} | "
            f"{stats['avg_cost']['median']:.2f} | "
            f"{stats['next_bull_return']['median'] * 100:.1f}% | "
            f"{stats['bear_end_position']['median']:.2f} |"
        )
    (OUT_DIR / "season3_bear_accumulation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _rows()
    summary = _summary(rows)
    (OUT_DIR / "season3_bear_accumulation.json").write_text(
        json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot(rows, "next_bull_return", "하락장 매수 평단 → 다음 상승장 수익률",
          "season3_bear_accumulation_next_bull.png")
    _plot(rows, "bear_end_position", "하락장 종료 시 실제 포지션",
          "season3_bear_accumulation_position.png")
    _write_report(rows, summary)
    print(f"rows={len(rows)}")
    print(f"report={OUT_DIR / 'season3_bear_accumulation.md'}")


if __name__ == "__main__":
    main()
