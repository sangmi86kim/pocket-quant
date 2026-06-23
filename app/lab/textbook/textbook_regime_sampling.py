"""국면 비율 맞춤 샘플링 — 블록을 국면 라벨로 뽑아 실제 분포에 맞춘다 (실데이터 실험).

[왜] 블록 길이로는 국면 격차가 안 풀렸다(textbook_blocklen_regime): 추세 갭은 L=126에서만
잡히는데 그땐 다양성 붕괴, 변동장 4.7배는 아예 안 잡힘. 다른 레버 = 일지 §III-2의
'국면 비율 맞춤 샘플링'.

[방법] 21일 블록을 그대로 두되(다양성 유지=권당 37블록), 무작위가 아니라 국면 비율에 맞춰 뽑는다.
  0. 실제 QQQ를 일별 국면 분류 → 각 21일 블록의 우세 국면으로 라벨.
  1. 블록을 국면 풀(상승/하락/횡보/변동)로 나눔.
  2. 실제 분포(58/23/18/0.7)에 맞춰 풀에서 블록 수 배분해 뽑음.
  3. 이어붙이는 순서 두 가지:
     - random  : 무작위 (비율만 맞춤)
     - clustered: 국면별로 뭉쳐 배치 (전환 최소화 → 변동장 오분류 줄이는지 검증)

[비교] baseline(균등 무작위) vs strat_random vs strat_clustered. 출력 국면 분포·실제거리.

[출력] app/lab/outputs/textbook/textbook_regime_sampling/ 에 png 2장 + md 리포트.
실행: .venv/Scripts/python.exe -m app.lab.textbook.textbook_regime_sampling
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
BLOCK = tb.BLOCK_DAYS
LABEL_ORDER = ["bull", "bear", "sideways", "volatile"]
LABEL_COLOR = {"bull": "#2e7d32", "bear": "#c62828",
               "sideways": "#9e9e9e", "volatile": "#f9a825"}
MODES = ["baseline", "strat_random", "strat_clustered"]
MODE_KO = {"baseline": "균등(현재)", "strat_random": "비율맞춤·무작위",
           "strat_clustered": "비율맞춤·뭉치기"}
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "textbook", "textbook_regime_sampling")


def _dist(counter):
    tot = sum(counter.values()) or 1
    return {k: 100.0 * counter.get(k, 0) / tot for k in LABEL_ORDER}


def _prepare():
    """공통 인덱스, 블록 국면 풀, 목표 블록 수를 한 번만 만든다."""
    qqq = get_prices("QQQ", tb.DATA_START, tb.DATA_END)
    qqq_ret = (qqq / qqq.shift(1)).apply(np.log).dropna()
    common_idx = qqq_ret.index
    rets = qqq_ret.to_numpy()

    # 실제 국면을 공통 인덱스에 정렬 (워밍업 전 구간은 NaN)
    labels = classify_daily(qqq).reindex(common_idx).to_numpy()
    real_d = _dist(collections.Counter(label for label in labels if isinstance(label, str)))

    # 블록 우세 국면 풀 — 21일 전부 분류된 시작점만
    pools = {k: [] for k in LABEL_ORDER}
    for i in range(len(common_idx) - BLOCK + 1):
        seg = labels[i:i + BLOCK]
        if any(not isinstance(x, str) for x in seg):
            continue
        dom = collections.Counter(seg).most_common(1)[0][0]
        pools[dom].append(i)

    n_blocks = math.ceil((tb.WARMUP_TDAYS + tb.EVAL_TDAYS) / BLOCK)
    counts = {k: int(round(real_d[k] / 100 * n_blocks)) for k in LABEL_ORDER}
    counts["bull"] += n_blocks - sum(counts.values())   # 반올림 오차는 상승장에 흡수
    return rets, pools, counts, n_blocks, real_d


def _world_labels(rets, starts, n_days):
    """블록 시작점들 → 합성 QQQ → 평가창 국면 라벨."""
    idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n_days]
    price = 100.0 * np.exp(np.cumsum(rets[idx]))
    dates = pd.bdate_range("2001-01-01", periods=n_days)
    synth = pd.Series(price, index=dates)
    lab = classify_daily(synth)
    return lab[lab.index >= dates[tb.WARMUP_TDAYS]]


def _sample_starts(mode, pools, counts, n_blocks, rng):
    """모드별 블록 시작점 목록."""
    if mode == "baseline":
        allp = [i for k in LABEL_ORDER for i in pools[k]]
        return list(rng.choice(allp, size=n_blocks, replace=False))
    chosen = {}
    for k in LABEL_ORDER:
        c = counts[k]
        if c <= 0 or not pools[k]:
            chosen[k] = []
            continue
        chosen[k] = list(rng.choice(pools[k], size=c, replace=c > len(pools[k])))
    if mode == "strat_clustered":
        return [s for k in LABEL_ORDER for s in chosen[k]]   # 국면별로 뭉침
    starts = [s for k in LABEL_ORDER for s in chosen[k]]      # strat_random
    rng.shuffle(starts)
    return starts


def run():
    rets, pools, counts, n_blocks, real_d = _prepare()
    n_days = tb.WARMUP_TDAYS + tb.EVAL_TDAYS
    print("블록 풀 크기: " + "  ".join(f"{REGIME_LABELS[k]} {len(pools[k])}" for k in LABEL_ORDER))
    print("목표 블록 수: " + "  ".join(f"{REGIME_LABELS[k]} {counts[k]}" for k in LABEL_ORDER)
          + f"  (합 {n_blocks})")

    results = {}
    for mode in MODES:
        counter = collections.Counter()
        for s in range(N_WORLDS):
            rng = np.random.default_rng(s)
            starts = _sample_starts(mode, pools, counts, n_blocks, rng)
            counter.update(_world_labels(rets, starts, n_days).values)
        d = _dist(counter)
        dist = sum(abs(d[k] - real_d[k]) for k in LABEL_ORDER)
        results[mode] = (d, dist)
        print(f"{MODE_KO[mode]:14s} " + "  ".join(f"{REGIME_LABELS[k]} {d[k]:4.1f}%" for k in LABEL_ORDER)
              + f"  | 실제거리 {dist:4.1f}")
    print("실제           " + "  ".join(f"{REGIME_LABELS[k]} {real_d[k]:4.1f}%" for k in LABEL_ORDER))

    os.makedirs(OUT_DIR, exist_ok=True)
    _fig_dist(real_d, results)
    _write_md(real_d, results, pools, counts, n_blocks)
    print(f"[저장] {OUT_DIR}")


def _fig_dist(real_d, results):
    groups = ["실제"] + [MODE_KO[m] for m in MODES]
    series = [real_d] + [results[m][0] for m in MODES]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                  gridspec_kw={"width_ratios": [3, 1.1]})
    xs = np.arange(len(LABEL_ORDER))
    w = 0.8 / len(groups)
    for i, (g, d) in enumerate(zip(groups, series)):
        off = (i - (len(groups) - 1) / 2) * w
        alpha = 1.0 if g == "실제" else 0.55
        ax.bar(xs + off, [d[k] for k in LABEL_ORDER], w,
               color=[LABEL_COLOR[k] for k in LABEL_ORDER], alpha=alpha,
               edgecolor="black", linewidth=0.4, hatch=None if g == "실제" else "//")
        ax.bar([-9], [0], w, label=g, color="gray", alpha=alpha,
               hatch=None if g == "실제" else "//")
    ax.set_xlim(-0.6, len(LABEL_ORDER) - 0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([REGIME_LABELS[k] for k in LABEL_ORDER])
    ax.set_ylabel("비율 (%)")
    ax.set_title("국면 분포 — 실제 vs 샘플링 방식 (블록 21일 유지=다양성 그대로)", fontsize=11)
    ax.legend(fontsize=8)

    # 실제거리 막대
    ax2.bar([MODE_KO[m] for m in MODES], [results[m][1] for m in MODES],
            color=["#9e9e9e", "#1565c0", "#2e7d32"], edgecolor="black", linewidth=0.5)
    for i, m in enumerate(MODES):
        ax2.text(i, results[m][1], f"{results[m][1]:.1f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("실제와의 거리 Σ|차이| (낮을수록 닮음)")
    ax2.set_title("실제거리")
    ax2.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "regime_sampling.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def _write_md(real_d, results, pools, counts, n_blocks):
    best = min(MODES, key=lambda m: results[m][1])
    L = [
        "# 국면 비율 맞춤 샘플링 — 블록을 국면 라벨로 뽑기 (실데이터 실험)\n",
        f"> 21일 블록을 국면 라벨로 뽑아 실제 비율에 맞춤. 합성 {N_WORLDS}권, 블록 길이는 21일 유지"
        f"(권당 {n_blocks}블록=다양성 그대로). 블록 길이 레버(다양성 붕괴)와 대비.\n",
        "\n![국면 분포](regime_sampling.png)\n",
        "\n## 블록 국면 풀 / 목표 배분\n\n",
        "| 국면 | 풀 크기(실제 블록) | 목표 블록/권 |\n|---|--:|--:|\n",
    ]
    for k in LABEL_ORDER:
        L.append(f"| {REGIME_LABELS[k]} | {len(pools[k])} | {counts[k]} |\n")

    L.append("\n## 방식별 국면 분포\n\n")
    L.append("| 방식 | " + " | ".join(REGIME_LABELS[k] for k in LABEL_ORDER) + " | 실제거리 |\n")
    L.append("|---|" + "--:|" * (len(LABEL_ORDER) + 1) + "\n")
    L.append("| **실제** | " + " | ".join(f"{real_d[k]:.1f}%" for k in LABEL_ORDER) + " | 0.0 |\n")
    for m in MODES:
        d, dist = results[m]
        star = " ⭐" if m == best else ""
        L.append(f"| {MODE_KO[m]}{star} | " + " | ".join(f"{d[k]:.1f}%" for k in LABEL_ORDER)
                 + f" | {dist:.1f} |\n")

    base, sr, sc = results["baseline"], results["strat_random"], results["strat_clustered"]
    L.append(
        "\n## 결론 — 국면 비율 맞춤 샘플링도 답이 아니다\n\n"
        f"1. **비율맞춤·무작위 ≈ 균등** (실제거리 {sr[1]:.1f} vs {base[1]:.1f}, 상승장 둘 다 ~{sr[0]['bull']:.0f}%). "
        "블록을 실제 비율(22/8/7)로 골라 넣어도 출력 분포가 안 바뀐다 — 균등 추출도 이미 실제 역사(대부분 "
        "상승장)에서 뽑아 입력 비율이 비슷하기 때문. **입력 블록 구성이 출력 국면을 결정하지 않는다.**\n"
        f"2. **뭉치기는 더 나쁨** (실제거리 {sc[1]:.1f}, 변동장 {sc[0]['volatile']:.1f}%). 국면별로 몰면 "
        "상승→하락 거대 전환 한 번이 긴 '추세 애매+고변동' 구간을 만들어 변동장·하락장을 오히려 키운다.\n"
        f"3. **근본 원인**: 국면은 MA200(약 200일=10블록) 같은 **장기 문맥** 속성인데 블록은 21일. "
        "이어붙이면 그 장기 추세가 깨지고, 출력 국면은 **블록 선택이 아니라 이음매 전환 동역학**이 지배한다. "
        "그래서 블록을 아무리 골라 넣어도(비율·순서) 출력 분포를 못 바꾼다.\n"
        "\n**→ 두 실험 종합**: 블록 부트스트랩은 짧은 블록으론 실제 국면 분포를 못 살린다. 길이를 크게(L≈126) "
        "하면 추세는 살지만 다양성 붕괴, 선택/순서로는 통제 불가. 남은 길 = **폭락장 분리**(흩뿌려진 위기만 "
        "따로 빼 변동장 줄이기, 미검증)거나, 일지의 **'졸업시험을 합격선이 아닌 진단지로 격하'**(합성 분포를 "
        "억지로 안 맞추고 합성의 한계를 인정). 다음 세션 결정거리.\n"
        "\n재현: `.venv/Scripts/python.exe -m app.lab.textbook.textbook_regime_sampling`\n"
    )
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("".join(L))


if __name__ == "__main__":
    run()
