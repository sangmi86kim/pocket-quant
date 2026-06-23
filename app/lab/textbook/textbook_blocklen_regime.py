"""국면 격차 손보기 — 블록 길이별 합성 국면 분포가 실제 나스닥에 얼마나 가까워지나.

[왜] 합성 교과서가 변동장 4.7배(0.7→3.3%)·상승장 -8pp로 실제보다 더 출렁댄다
(synth_vs_real_regime). 원인: 21일 블록을 무작위로 섞으면 고변동(위기) 블록이 상승장
한복판에 흩뿌려져 '추세 애매 + 고변동' = 변동장으로 오분류된다(regime.py 판정). 실제는
위기가 닷컴·금융위기에 뭉쳐 있어 깔끔하게 하락장으로 잡힌다.

[가설] 블록을 길게 하면 위기 블록이 내부적으로 '고변동+하락'을 유지해 bear로 잡히고,
추세도 안 끊겨 bull이 살아난다 → 변동장↓·상승장↑로 실제에 수렴.

[실험] 블록 길이 L ∈ {21,42,63,84,126}로 합성 N권씩 만들어 국면 분포·연속 상승장 길이를
실제 QQQ와 비교. 실제와의 거리(Σ|합성%-실제%|)가 최소인 길이를 찾는다. 단 길수록 블록
다양성(권당 블록 수)이 줄어 과적합 위험 — 그 트레이드오프도 같이 본다.

[출력] app/lab/outputs/textbook/textbook_blocklen_regime/ 에 png 2장 + md 리포트.
실행: .venv/Scripts/python.exe -m app.lab.textbook.textbook_blocklen_regime
"""

import collections
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.academy.curriculum import textbook as tb
from app.world.data_loader import get_prices
from app.world.regime import REGIME_LABELS, classify_daily

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

N_WORLDS = 50
BLOCK_LENGTHS = (21, 42, 63, 84, 126)
LABEL_ORDER = ["bull", "bear", "sideways", "volatile"]
LABEL_COLOR = {"bull": "#2e7d32", "bear": "#c62828",
               "sideways": "#9e9e9e", "volatile": "#f9a825"}
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "textbook", "textbook_blocklen_regime")


def _dist(counter):
    tot = sum(counter.values()) or 1
    return {k: 100.0 * counter.get(k, 0) / tot for k in LABEL_ORDER}


def _mean_bull_run(labels):
    runs, cur = [], 0
    for v in labels:
        if v == "bull":
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return sum(runs) / len(runs) if runs else 0.0


def _synth_dist(block_len):
    """블록 길이를 바꿔 합성 N권의 국면 분포·평균 상승장 길이."""
    orig = tb.BLOCK_DAYS
    tb.BLOCK_DAYS = block_len          # _sample_block_indices·offset이 참조하는 전역
    try:
        counter = collections.Counter()
        runs = []
        for s in range(N_WORLDS):
            g = tb.make_world(seed=s)
            lab = classify_daily(g.prices)
            lab = lab[lab.index >= pd.Timestamp(g.gym.start)]
            counter.update(lab.values)
            runs.append(_mean_bull_run(list(lab.values)))
    finally:
        tb.BLOCK_DAYS = orig
    return _dist(counter), sum(runs) / len(runs)


def run():
    real = classify_daily(get_prices("QQQ", tb.DATA_START, tb.DATA_END))
    real_d = _dist(collections.Counter(real.values))
    real_run = _mean_bull_run(list(real.values))

    results = {}    # L -> (dist, bull_run, dist_to_real, blocks_per_world)
    n_days = tb.WARMUP_TDAYS + tb.EVAL_TDAYS
    for L in BLOCK_LENGTHS:
        d, run_len = _synth_dist(L)
        dist_to_real = sum(abs(d[k] - real_d[k]) for k in LABEL_ORDER)
        results[L] = (d, run_len, dist_to_real, math.ceil(n_days / L))
        print(f"L={L:3d}  " + "  ".join(f"{REGIME_LABELS[k]} {d[k]:4.1f}%" for k in LABEL_ORDER)
              + f"  | 실제거리 {dist_to_real:4.1f}  블록/권 {math.ceil(n_days / L)}")
    print("실제      " + "  ".join(f"{REGIME_LABELS[k]} {real_d[k]:4.1f}%" for k in LABEL_ORDER))

    os.makedirs(OUT_DIR, exist_ok=True)
    _fig_dist(real_d, results)
    _fig_converge(real_d, real_run, results)
    _write_md(real_d, real_run, results)
    print(f"[저장] {OUT_DIR}")


def _fig_dist(real_d, results):
    """국면 분포 — 실제 + 블록길이별 그룹 막대."""
    groups = ["실제"] + [f"L={L}" for L in BLOCK_LENGTHS]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    xs = np.arange(len(LABEL_ORDER))
    w = 0.8 / len(groups)
    series = [real_d] + [results[L][0] for L in BLOCK_LENGTHS]
    for i, (g, d) in enumerate(zip(groups, series)):
        off = (i - (len(groups) - 1) / 2) * w
        alpha = 1.0 if g == "실제" else 0.55
        ax.bar(xs + off, [d[k] for k in LABEL_ORDER], w,
               color=[LABEL_COLOR[k] for k in LABEL_ORDER], alpha=alpha,
               edgecolor="black", linewidth=0.4,
               hatch=None if g == "실제" else "//")
        ax.bar(xs[:1] + off, [0], w, label=g,
               color="gray", alpha=alpha, hatch=None if g == "실제" else "//")
    ax.set_xticks(xs)
    ax.set_xticklabels([REGIME_LABELS[k] for k in LABEL_ORDER])
    ax.set_ylabel("비율 (%)")
    ax.set_title("국면 분포 — 실제 나스닥(진하게) vs 블록 길이별 합성(빗금)\n"
                 "블록이 길수록 변동장↓·상승장↑로 실제에 수렴하는지", fontsize=11)
    ax.legend(ncol=len(groups), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "regime_by_blocklen.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def _fig_converge(real_d, real_run, results):
    """수렴: 블록 길이(x) vs 상승장·변동장%, 실제거리, 블록/권 트레이드오프."""
    Ls = list(BLOCK_LENGTHS)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))

    a1.plot(Ls, [results[L][0]["bull"] for L in Ls], "o-", color=LABEL_COLOR["bull"],
            label="상승장 (합성)")
    a1.plot(Ls, [results[L][0]["volatile"] for L in Ls], "o-", color=LABEL_COLOR["volatile"],
            label="변동장 (합성)")
    a1.axhline(real_d["bull"], ls="--", color=LABEL_COLOR["bull"], alpha=0.7,
               label=f"실제 상승 {real_d['bull']:.0f}%")
    a1.axhline(real_d["volatile"], ls="--", color=LABEL_COLOR["volatile"], alpha=0.7,
               label=f"실제 변동 {real_d['volatile']:.1f}%")
    a1.set_xlabel("블록 길이 (거래일)")
    a1.set_ylabel("비율 (%)")
    a1.set_title("상승장·변동장이 실제선(점선)에 수렴")
    a1.legend(fontsize=8)

    a2.plot(Ls, [results[L][2] for L in Ls], "s-", color="#1565c0", label="실제와의 거리 Σ|차이|")
    a2.set_xlabel("블록 길이 (거래일)")
    a2.set_ylabel("실제거리 (낮을수록 닮음)", color="#1565c0")
    a2.tick_params(axis="y", labelcolor="#1565c0")
    best = min(Ls, key=lambda L: results[L][2])
    a2.axvline(best, ls=":", color="#1565c0", alpha=0.6)
    a2.annotate(f"최소 L={best}", xy=(best, results[best][2]), fontsize=9, color="#1565c0")
    a3 = a2.twinx()
    a3.plot(Ls, [results[L][3] for L in Ls], "^--", color="#999", label="블록/권(다양성)")
    a3.set_ylabel("권당 블록 수 (적을수록 과적합 위험)", color="#999")
    a3.tick_params(axis="y", labelcolor="#999")
    a2.set_title("실제거리 ↓ vs 다양성 ↓ 트레이드오프")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "convergence.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def _write_md(real_d, real_run, results):
    best = min(BLOCK_LENGTHS, key=lambda L: results[L][2])
    L = [
        "# 국면 격차 손보기 — 블록 길이별 합성 국면 분포 (실데이터 실험)\n",
        f"> 블록 길이 {BLOCK_LENGTHS}별로 합성 {N_WORLDS}권씩, 실제 QQQ({tb.DATA_START}~{tb.DATA_END})와 "
        "국면 분포 비교. 분류기 `world/regime.py`.\n",
        "\n![국면 분포](regime_by_blocklen.png)\n\n![수렴/트레이드오프](convergence.png)\n",
        "\n## 블록 길이별 국면 분포\n\n",
        "| 블록 | " + " | ".join(REGIME_LABELS[k] for k in LABEL_ORDER)
        + " | 실제거리 | 블록/권 |\n",
        "|---|" + "--:|" * (len(LABEL_ORDER) + 2) + "\n",
        "| **실제** | " + " | ".join(f"{real_d[k]:.1f}%" for k in LABEL_ORDER) + " | 0.0 | — |\n",
    ]
    for bl in BLOCK_LENGTHS:
        d, _, dist, bpw = results[bl]
        star = " ⭐" if bl == best else ""
        L.append(f"| L={bl}{star} | " + " | ".join(f"{d[k]:.1f}%" for k in LABEL_ORDER)
                 + f" | {dist:.1f} | {bpw} |\n")

    bd = results[best][0]
    vol_span = max(results[bl][0]["volatile"] for bl in BLOCK_LENGTHS) - \
        min(results[bl][0]["volatile"] for bl in BLOCK_LENGTHS)
    L.append(
        f"\n## 결론 — 블록 길이는 깨끗한 레버가 아니다\n\n"
        f"1. **추세 갭(상승장)은 길이로 잡힌다.** L={best}일에서 상승 {bd['bull']:.1f}%·하락 {bd['bear']:.1f}%·"
        f"횡보 {bd['sideways']:.1f}%가 실제(58/23/18)에 거의 수렴 — 실제거리 최소 {results[best][2]:.1f}.\n"
        f"2. **변동장(4.7배)은 길이로 안 잡힌다.** 모든 길이에서 {min(results[bl][0]['volatile'] for bl in BLOCK_LENGTHS):.1f}"
        f"~{max(results[bl][0]['volatile'] for bl in BLOCK_LENGTHS):.1f}%로 평평(변동폭 {vol_span:.1f}pp), "
        f"실제 {real_d['volatile']:.1f}% 근처도 못 감. 짧은 합성창(774일)에서 변동성 백분위가 정의상 상위 15%를 "
        "만들고, 그중 추세 애매한 전환 구간이 변동장으로 잡히는 구조적 문제 → 길이로는 못 푼다.\n"
        f"3. **다양성 붕괴.** 실제거리 최소인 L={best}는 권당 블록이 {results[best][3]}개뿐(현 21일은 37개) "
        "→ 챔피언이 7조각만 외우는 과적합 위험.\n"
        "\n**→ 권고: 블록 길이만으로 끝내지 말 것.** 추세 갭은 적당한 길이로 일부 줄이되, 변동장·다양성을 같이 "
        "잡으려면 일지 §III-2의 다른 레버가 필요하다 — **국면 비율 맞춤 샘플링**(블록을 국면 라벨로 뽑아 실제 "
        "58/23/18/0.7에 맞춤) + **폭락장 분리**(닷컴·금융위기 블록은 일반 교과서서 빼고 별도 시험장). 다음 실험 후보.\n"
        "\n재현: `.venv/Scripts/python.exe -m app.lab.textbook.textbook_blocklen_regime`\n"
    )
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("".join(L))


if __name__ == "__main__":
    run()
