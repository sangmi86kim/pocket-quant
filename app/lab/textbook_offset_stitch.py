"""레벨형 신호 합성 — raw vs offset vs offset+앵커 (실데이터 실험).

[왜] 시즌 3 교과서 점검. 교과서(`curriculum/textbook.py`)는 외부 정보원을 타입별로 다르게
복원한다. 가격형(QQQ/SPY/TLT/DIA)은 수익률로 잘라 누적 → 이음매 연속(OK). 레벨형(^VIX/^TNX)은
절대값 raw를 그대로 이어붙임 → 21일 블록 경계마다 점프(평온 13 → 위기 52). 그 가짜 절벽을
US10Y('60일 평균 대비 급락')가 진짜 금리인하로 착각해 오발동한다.

[세 방식 비교]
  raw    : 현재 교과서 (절대값 그대로 이어붙임) → 경계 절벽
  offset : 블록을 통째로 위아래로 밀어 앞 블록 끝에 이어붙임 → 경계 연속, 내부 모양 보존.
           단 앵커가 없어 블록 순변화가 누적되면 밴드를 벗어나 떠내려갈 수 있음.
  anchor : offset과 같되 매일 자기 평균(μ)으로 살짝 잡아당김(평균회귀) → 떠내려감 억제.

[측정] 시드 N권에 대해
  ⓐ 경계 점프 = 블록 이음매 |값[s]-값[s-1]| 평균
  ⓑ 떠내려감 = 권별 합성 레벨 min/max 평균 vs 실제 역사 밴드
  ⓒ 발동 횟수 = FEAR(VIX>30)·US10Y(60일평균 대비 -0.5%p) 평가창 일수

[출력] app/lab/outputs/textbook_offset_stitch/ 에 png 2장 + md 리포트.
실행: .venv/Scripts/python.exe -m app.lab.textbook_offset_stitch
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.academy.curriculum import textbook as tb
from app.pocket.signals import FEAR_THRESHOLD, US10Y_DROP_BP
from app.world.data_loader import get_prices

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ── 실험 설정 (argparse 대신 상수) ──
N_SEEDS = 200                     # 떠내려감 분포 안정용 권수
EXAMPLE_SEED = 0                  # 예시 그림으로 그릴 한 권
LEVEL_TICKERS = ("^VIX", "^TNX")  # 레벨형 = 이 둘만 (FEAR·US10Y가 씀)
BLOCK = tb.BLOCK_DAYS             # 21
WARMUP = tb.WARMUP_TDAYS          # 270 (성적 제외, 지표 예열)
THETA = 0.04                      # 앵커 세기 — 하루에 (μ-현재)의 4%를 평균으로 당김(반감기 ~17일)

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "textbook_offset_stitch")
METHODS = ("raw", "offset", "anchor")
COLOR = {"raw": "#c62828", "offset": "#f9a825", "anchor": "#2e7d32"}   # 빨강/주황/초록
LABEL = {"raw": "raw (현재: 절대값 이어붙임)",
         "offset": "offset (블록 평행이동 → 연속)",
         "anchor": "offset+앵커 (평균회귀로 밴드 유지)"}


def offset_stitch(vals: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """블록 내부 모양은 그대로, 블록을 통째로 밀어 앞 블록 끝에 이어붙인다.

    블록 b 전체에 (앞 블록 끝값 - 블록 b 첫값)을 더해 이음매를 연속으로 만든다.
    → 경계 점프 0, 내부 일별 등락(트렌드) 100% 보존. 앵커 없음 = 떠내려감은 그대로 둠.
    """
    out = np.array(vals, dtype=float)
    for s in range(block, len(out), block):
        out[s:s + block] += out[s - 1] - out[s]
    return out


def anchor_stitch(vals: np.ndarray, mu: float,
                  theta: float = THETA, block: int = BLOCK) -> np.ndarray:
    """offset과 같은 연속 이어붙임 + 매일 평균(μ)으로 살짝 당기는 평균회귀.

    out[t] = out[t-1] + delta[t] + theta*(μ - out[t-1])
      delta[t] = 블록 경계면 0(이어붙임), 아니면 raw의 그날 변화(내부 등락 보존).
    평균회귀 항이 누적 표류를 평소 밴드(μ) 쪽으로 되돌려 떠내려감을 막는다.
    """
    out = np.empty(len(vals), dtype=float)
    out[0] = vals[0]
    for t in range(1, len(vals)):
        delta = 0.0 if t % block == 0 else vals[t] - vals[t - 1]
        out[t] = out[t - 1] + delta + theta * (mu - out[t - 1])
    return out


def _clean_level(series: pd.Series) -> pd.Series:
    """레벨형 원천의 자잘한 holiday 결측만 메운다(VIX·금리는 연속 실데이터)."""
    return series.interpolate().ffill().bfill()


def _prepare():
    """make_world와 동일한 공통 인덱스·블록단위를 한 번만 만들어 둔다(이후 슬라이싱만)."""
    qqq = get_prices("QQQ", tb.DATA_START, tb.DATA_END)
    qqq_ret = (qqq / qqq.shift(1)).apply(np.log).dropna()
    common_idx = qqq_ret.index

    aligned = tb._load_external(tb.DEFAULT_EXTERNAL_STREAMS, tb.DATA_START,
                                tb.DATA_END, common_idx)
    levels = {}
    for tk in LEVEL_TICKERS:
        block_unit, stype = aligned[tk]
        assert stype == "level"
        levels[tk] = _clean_level(block_unit)
    n_days = tb.WARMUP_TDAYS + tb.EVAL_TDAYS
    return levels, n_days


def _build_world(levels, mu, n_days, seed):
    """한 권: 같은 블록 인덱스로 VIX·TNX를 raw / offset / anchor 세 방식 복원."""
    rng = np.random.default_rng(seed)
    idx = tb._sample_block_indices(n_days, len(next(iter(levels.values()))), rng)
    out = {}
    for tk, ser in levels.items():
        raw = np.asarray(ser.values)[idx]
        out[tk] = {"raw": raw,
                   "offset": offset_stitch(raw),
                   "anchor": anchor_stitch(raw, mu[tk])}
    return out


def _seam_jumps(vals: np.ndarray) -> np.ndarray:
    seams = np.arange(BLOCK, len(vals), BLOCK)
    return np.abs(vals[seams] - vals[seams - 1])


def _fear_fires(vix: np.ndarray) -> int:
    """평가창에서 FEAR(VIX>30) 발동 일수."""
    return int((vix[WARMUP:] > FEAR_THRESHOLD).sum())


def _us10y_fires(tnx: np.ndarray) -> int:
    """평가창에서 US10Y(60일 평균 대비 -0.5%p 하락) 발동 일수. 60일평균은 전체로 예열."""
    s = pd.Series(tnx)
    diff = (s - s.rolling(60, min_periods=20).mean()).to_numpy()
    return int((diff[WARMUP:] < -US10Y_DROP_BP).sum())


def run():
    levels, n_days = _prepare()
    real_band = {tk: (float(s.min()), float(s.max()), float(s.mean()))
                 for tk, s in levels.items()}
    mu = {tk: real_band[tk][2] for tk in LEVEL_TICKERS}   # 앵커 평균 = 자기 역사 평균

    # 누적 통계: 방식별 경계점프·min·max (티커별), 발동수(FEAR=VIX, US10Y=TNX)
    jump = {tk: {mth: [] for mth in METHODS} for tk in LEVEL_TICKERS}
    vmin = {tk: {mth: [] for mth in METHODS} for tk in LEVEL_TICKERS}
    vmax = {tk: {mth: [] for mth in METHODS} for tk in LEVEL_TICKERS}
    fear = {mth: [] for mth in METHODS}
    us10y = {mth: [] for mth in METHODS}

    example = None
    for seed in range(N_SEEDS):
        world = _build_world(levels, mu, n_days, seed)
        if seed == EXAMPLE_SEED:
            example = world
        for tk in LEVEL_TICKERS:
            for mth in METHODS:
                v = world[tk][mth]
                jump[tk][mth].append(_seam_jumps(v).mean())
                vmin[tk][mth].append(v.min())
                vmax[tk][mth].append(v.max())
        for mth in METHODS:
            fear[mth].append(_fear_fires(world["^VIX"][mth]))
            us10y[mth].append(_us10y_fires(world["^TNX"][mth]))

    os.makedirs(OUT_DIR, exist_ok=True)
    _fig_example(example, real_band)
    _fig_stats(jump, vmin, vmax, fear, us10y, real_band)
    _write_md(jump, vmin, vmax, fear, us10y, real_band)
    print(f"[저장] {OUT_DIR}")


def _fig_example(world, real_band):
    """예시 한 권: VIX·TNX의 세 방식 시계열 (결정용 — 읽히게)."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 9.5))
    x = np.arange(len(world["^VIX"]["raw"]))
    style = {"raw": dict(lw=1.3, alpha=0.9, ls="-"),
             "offset": dict(lw=1.6, alpha=0.95, ls="--"),
             "anchor": dict(lw=1.8, alpha=0.95, ls="-")}

    panels = [("^VIX", "VIX 공포지수 — FEAR는 '그날 값 > 30'에 발동", FEAR_THRESHOLD, "VIX 레벨"),
              ("^TNX", "^TNX 10년물 금리 — US10Y는 '60일 평균 대비 -0.5%p 급락'에 발동",
               None, "금리 (%)")]
    for ax, (tk, title, hline, ylab) in zip(axes, panels):
        lo, hi, _ = real_band[tk]
        ax.axhspan(lo, hi, color="#90caf9", alpha=0.13, zorder=0,
                   label=f"실제 역사 밴드 [{lo:.0f}–{hi:.0f}]")
        for mth in METHODS:
            ax.plot(x, world[tk][mth], color=COLOR[mth], label=LABEL[mth],
                    zorder=3, **style[mth])
        if hline is not None:
            ax.axhline(hline, ls="--", color="black", alpha=0.6, lw=1.0)
            ax.text(len(x) - 2, hline + 1.5, "FEAR 발동선 30", ha="right",
                    fontsize=9, color="black")
        ax.axvline(WARMUP - 0.5, color="navy", alpha=0.5, lw=1.4)
        ax.text(WARMUP + 4, ax.get_ylim()[1], "← 예열 | 성적구간 →", color="navy",
                fontsize=9, va="top")
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_ylabel(ylab, fontsize=11)
        ax.margins(x=0.01)

    # VIX 패널에 핵심 포인트 화살표 — offset이 0 아래로 떠내려감
    vix_off = world["^VIX"]["offset"]
    lo_i = int(np.argmin(vix_off))
    axes[0].annotate("offset 떠내려감\n(0 아래로)", xy=(lo_i, vix_off[lo_i]),
                     xytext=(lo_i + 40, vix_off[lo_i] - 2), color=COLOR["offset"],
                     fontsize=9, arrowprops=dict(arrowstyle="->", color=COLOR["offset"]))

    axes[1].set_xlabel(f"거래일 (1999~2020 실데이터를 21일 블록으로 셔플 · 남색선={WARMUP}일 예열 끝)",
                       fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle(f"레벨형 합성 세 방식 비교 — 예시 1권(seed={EXAMPLE_SEED})",
                 fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(os.path.join(OUT_DIR, "example_world.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def _grouped_bar(ax, groups, series_by_method, title, ylab, fmt="{:.1f}"):
    """공통: x그룹 × 방식별 막대."""
    xs = np.arange(len(groups))
    w = 0.26
    for i, mth in enumerate(METHODS):
        vals = series_by_method[mth]
        bars = ax.bar(xs + (i - 1) * w, vals, w, color=COLOR[mth], label=LABEL[mth])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(groups)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylab)


def _fig_stats(jump, vmin, vmax, fear, us10y, real_band):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.8))

    # ⓐ 경계 점프 — 티커별 × 방식
    _grouped_bar(a1, list(LEVEL_TICKERS),
                 {mth: [np.mean(jump[tk][mth]) for tk in LEVEL_TICKERS] for mth in METHODS},
                 "ⓐ 블록 경계 점프 평균\n(낮을수록 매끄러움)", "|이음매 값 변화|", "{:.2f}")
    a1.legend(fontsize=7)

    # ⓑ 떠내려감 — [VIX min, VIX max, TNX min, TNX max] × 방식, 실제 밴드 점선
    groups = ["VIX\nmin", "VIX\nmax", "TNX\nmin", "TNX\nmax"]
    series = {mth: [np.mean(vmin["^VIX"][mth]), np.mean(vmax["^VIX"][mth]),
                    np.mean(vmin["^TNX"][mth]), np.mean(vmax["^TNX"][mth])]
              for mth in METHODS}
    _grouped_bar(a2, groups, series,
                 "ⓑ 권별 레벨 min/max 평균\n(검은선=실제 역사 한계, 넘으면 떠내려감)", "레벨 값")
    for j, (tk, which) in enumerate([("^VIX", 0), ("^VIX", 1), ("^TNX", 0), ("^TNX", 1)]):
        ref = real_band[tk][which]   # min 또는 max
        a2.hlines(ref, j - 0.4, j + 0.4, color="black", ls="--", lw=1.1)
    a2.axhline(0, color="gray", lw=0.6)

    # ⓒ 발동 횟수 — FEAR(VIX), US10Y(TNX) × 방식
    _grouped_bar(a3, ["FEAR\n(VIX>30)", "US10Y\n(금리급락)"],
                 {mth: [np.mean(fear[mth]), np.mean(us10y[mth])] for mth in METHODS},
                 "ⓒ 평가창 발동 일수 평균\n(US10Y↓=가짜제거 / FEAR는 raw가 정상기준)", "발동 일수 / 권")
    a3.legend(fontsize=7)

    fig.suptitle(f"레벨형 세 방식 — {N_SEEDS}권 통계 (앵커 θ={THETA})", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "stats.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def _write_md(jump, vmin, vmax, fear, us10y, real_band):
    def m(x):
        return float(np.mean(x))

    L = [
        "# 레벨형 신호 합성 — raw vs offset vs offset+앵커 (실데이터 실험)\n",
        f"> 실제 QQQ/VIX/금리({tb.DATA_START}~{tb.DATA_END})로 합성 {N_SEEDS}권. 레벨형(^VIX/^TNX)을 "
        f"세 방식으로 복원해 비교. 앵커 평균회귀 세기 θ={THETA}, 평균 μ=자기 역사 평균.\n",
        "\n## 예시 한 권 — 세 방식이 어떻게 다른가\n\n",
        "![예시 1권](example_world.png)\n\n",
        "*VIX 패널(위): 빨강 raw는 블록 경계마다 절벽이 튄다. 주황 offset은 절벽은 없지만 "
        "0 아래로 떠내려간다(가짜 공포). 초록 앵커는 파란 밴드 안에 머문다. "
        "TNX 패널(아래): offset·앵커 둘 다 매끄럽고 밴드 안 — 금리는 offset으로 충분.*\n\n",
        "![통계](stats.png)\n\n",
        "*ⓐ 경계 점프는 offset·앵커 둘 다 0으로 사라짐. ⓑ VIX max를 보면 raw 61 → 앵커 43으로 "
        "진짜 위기 스파이크가 눌린다. ⓒ US10Y는 raw 186 → offset 11(가짜 175일 제거), FEAR는 "
        "raw 47이 정상 기준인데 offset 146(폭증)·앵커 13(과소).*\n",
        "\n## ⓐ 블록 경계 점프 (낮을수록 매끄러움)\n\n",
        "| 레벨형 | raw | offset | anchor |\n|---|--:|--:|--:|\n",
    ]
    for tk in LEVEL_TICKERS:
        L.append(f"| {tk} | {m(jump[tk]['raw']):.3f} | {m(jump[tk]['offset']):.4f} | "
                 f"{m(jump[tk]['anchor']):.4f} |\n")

    L.append("\n## ⓑ 떠내려감 — 권별 레벨 범위 vs 실제 역사 밴드\n\n")
    L.append("| 레벨형 | 실제 [min, max] | raw [min,max] | offset [min,max] | anchor [min,max] |\n")
    L.append("|---|--:|--:|--:|--:|\n")
    for tk in LEVEL_TICKERS:
        lo, hi, _ = real_band[tk]
        L.append(f"| {tk} | [{lo:.1f}, {hi:.1f}] | "
                 f"[{m(vmin[tk]['raw']):.1f}, {m(vmax[tk]['raw']):.1f}] | "
                 f"[{m(vmin[tk]['offset']):.1f}, {m(vmax[tk]['offset']):.1f}] | "
                 f"[{m(vmin[tk]['anchor']):.1f}, {m(vmax[tk]['anchor']):.1f}] |\n")

    L.append("\n## ⓒ 평가창 발동 일수 (권당 평균)\n\n")
    L.append("| 신호 | raw | offset | anchor |\n|---|--:|--:|--:|\n")
    L.append(f"| FEAR (VIX>30) | {m(fear['raw']):.1f} | {m(fear['offset']):.1f} | "
             f"{m(fear['anchor']):.1f} |\n")
    L.append(f"| US10Y (금리급락) | {m(us10y['raw']):.1f} | {m(us10y['offset']):.1f} | "
             f"{m(us10y['anchor']):.1f} |\n")

    L.append(
        "\n## 처방 결론\n\n"
        "| 레벨형 | 처방 | 근거 |\n|---|---|---|\n"
        f"| **^TNX (US10Y)** | **offset 채택** | 가짜 금리인하 {m(us10y['raw']) - m(us10y['offset']):.0f}일 제거"
        f"({m(us10y['raw']):.0f}→{m(us10y['offset']):.0f}), 밴드도 유지. 앵커는 과함. |\n"
        f"| **^VIX (FEAR)** | **raw 유지 권장** | FEAR는 절대 임계라 절벽이 무해. "
        f"offset은 떠내려가 폭증({m(fear['offset']):.0f}), 앵커는 위기 압축으로 과소({m(fear['anchor']):.0f}); "
        f"raw {m(fear['raw']):.0f}이 진짜에 가장 충실. |\n"
        "\n재현: `.venv/Scripts/python.exe -m app.lab.textbook_offset_stitch`\n"
    )
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("".join(L))


if __name__ == "__main__":
    run()
