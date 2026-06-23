"""폭락장 분리 — 고변동 블록을 일반 교과서서 빼면 변동장 오분류가 줄어드나 (실데이터 실험).

[왜] 블록 길이·국면 비율 샘플링 둘 다 국면 격차를 못 풀었다. 마지막 싼 레버 = 일지 §III-2의
'폭락장 분리': 닷컴·금융위기 같은 고변동 블록을 일반 교과서서 빼 별도 시험장으로. 가설은
'흩뿌려진 위기 블록이 상승장 한복판에 끼어 추세 애매+고변동 = 변동장 오분류를 만든다'였으므로,
위기 블록을 빼면 일반 교과서의 변동장이 줄어야 한다.

[방법] 21일 블록의 실현변동성으로 폭락 판정(상위 CRASH_PCT%). 세 모드 비교:
  전체(현재)   : 모든 블록에서 추출
  폭락제외     : 하위 80% 블록만 (일반 교과서 = 평시장 시험)
  폭락만       : 상위 20% 블록만 (폭락 시험 = 방어 시험장)
각 모드 합성 N권의 국면 분포를 실제 나스닥과 비교. 핵심 = 폭락제외의 변동장이 떨어지나.

[출력] app/lab/outputs/textbook/textbook_crash_split/ 에 png 1장 + md 리포트.
실행: .venv/Scripts/python.exe -m app.lab.textbook.textbook_crash_split
"""

import collections
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
CRASH_PCT = 80            # 실현변동성 상위 (100-80)=20%를 '폭락 블록'으로
LABEL_ORDER = ["bull", "bear", "sideways", "volatile"]
LABEL_COLOR = {"bull": "#2e7d32", "bear": "#c62828",
               "sideways": "#9e9e9e", "volatile": "#f9a825"}
MODES = ["전체(현재)", "폭락제외(일반시험)", "폭락만(폭락시험)"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "textbook", "textbook_crash_split")


def _dist(counter):
    tot = sum(counter.values()) or 1
    return {k: 100.0 * counter.get(k, 0) / tot for k in LABEL_ORDER}


def _prepare():
    qqq = get_prices("QQQ", tb.DATA_START, tb.DATA_END)
    qqq_ret = (qqq / qqq.shift(1)).apply(np.log).dropna()
    rets = qqq_ret.to_numpy()
    n = len(rets)

    # 블록별 실현변동성(연율화) → 상위 20%를 폭락 블록으로
    vols = np.array([rets[i:i + BLOCK].std() for i in range(n - BLOCK + 1)]) * np.sqrt(252)
    thr = float(np.percentile(vols, CRASH_PCT))
    pools = {
        "전체(현재)": np.arange(n - BLOCK + 1),
        "폭락제외(일반시험)": np.where(vols < thr)[0],
        "폭락만(폭락시험)": np.where(vols >= thr)[0],
    }
    real_d = _dist(collections.Counter(classify_daily(qqq).values))
    return rets, pools, real_d, thr


def _world_labels(rets, starts, n_days):
    idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n_days]
    price = 100.0 * np.exp(np.cumsum(rets[idx]))
    dates = pd.bdate_range("2001-01-01", periods=n_days)
    lab = classify_daily(pd.Series(price, index=dates))
    return lab[lab.index >= dates[tb.WARMUP_TDAYS]]


def run():
    rets, pools, real_d, thr = _prepare()
    n_days = tb.WARMUP_TDAYS + tb.EVAL_TDAYS
    n_blocks = int(np.ceil(n_days / BLOCK))
    print(f"폭락 임계(실현변동성 상위 {100 - CRASH_PCT}%): {thr:.3f}")
    print("풀 크기: " + "  ".join(f"{m} {len(pools[m])}" for m in MODES))

    results = {}
    for m in MODES:
        counter = collections.Counter()
        for s in range(N_WORLDS):
            rng = np.random.default_rng(s)
            starts = list(rng.choice(pools[m], size=n_blocks, replace=n_blocks > len(pools[m])))
            counter.update(_world_labels(rets, starts, n_days).values)
        d = _dist(counter)
        dist = sum(abs(d[k] - real_d[k]) for k in LABEL_ORDER)
        results[m] = (d, dist)
        print(f"{m:18s} " + "  ".join(f"{REGIME_LABELS[k]} {d[k]:4.1f}%" for k in LABEL_ORDER)
              + f"  | 실제거리 {dist:4.1f}")
    print("실제               " + "  ".join(f"{REGIME_LABELS[k]} {real_d[k]:4.1f}%" for k in LABEL_ORDER))

    os.makedirs(OUT_DIR, exist_ok=True)
    _fig(real_d, results)
    _write_md(real_d, results, pools, thr, n_blocks)
    print(f"[저장] {OUT_DIR}")


def _fig(real_d, results):
    groups = ["실제"] + MODES
    series = [real_d] + [results[m][0] for m in MODES]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                  gridspec_kw={"width_ratios": [3, 1.2]})
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
    ax.set_title("국면 분포 — 실제 vs 폭락 분리 모드", fontsize=11)
    ax.legend(fontsize=8)

    # 변동장만 집중 비교 (핵심 지표)
    vols = [real_d["volatile"]] + [results[m][0]["volatile"] for m in MODES]
    cols = ["#2e7d32", "#9e9e9e", "#1565c0", "#c62828"]
    ax2.bar(["실제"] + MODES, vols, color=cols, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(vols):
        ax2.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax2.axhline(real_d["volatile"], ls="--", color="#2e7d32", alpha=0.6)
    ax2.set_ylabel("변동장 비율 (%)")
    ax2.set_title("변동장 — 폭락제외가 실제선(점선)에 가까워지나")
    ax2.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "crash_split.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def _write_md(real_d, results, pools, thr, n_blocks):
    reg = results["폭락제외(일반시험)"][0]
    base = results["전체(현재)"][0]
    L = [
        "# 폭락장 분리 — 고변동 블록을 일반 교과서서 빼기 (실데이터 실험)\n",
        f"> 21일 블록을 실현변동성 상위 {100 - CRASH_PCT}%(임계 {thr:.2f})로 폭락 판정. 세 모드 합성 "
        f"{N_WORLDS}권씩, 국면 분포를 실제 QQQ와 비교. 블록 21일 유지(권당 {n_blocks}블록=다양성 보존).\n",
        "\n![국면 분포 + 변동장](crash_split.png)\n",
        "\n## 모드별 국면 분포\n\n",
        "| 모드 | 풀 블록수 | " + " | ".join(REGIME_LABELS[k] for k in LABEL_ORDER) + " | 실제거리 |\n",
        "|---|--:|" + "--:|" * (len(LABEL_ORDER) + 1) + "\n",
        "| **실제** | — | " + " | ".join(f"{real_d[k]:.1f}%" for k in LABEL_ORDER) + " | 0.0 |\n",
    ]
    for m in MODES:
        d, dist = results[m]
        L.append(f"| {m} | {len(pools[m])} | "
                 + " | ".join(f"{d[k]:.1f}%" for k in LABEL_ORDER) + f" | {dist:.1f} |\n")

    crash = results["폭락만(폭락시험)"][0]
    L.append(
        f"\n## 결론 — 변동장은 못 고치나, 두 코헤어런트 시험은 얻는다\n\n"
        f"1. **변동장은 폭락 분리로도 안 고쳐짐(오히려 늘어남).** 전체 {base['volatile']:.1f}% → "
        f"폭락제외 **{reg['volatile']:.1f}%**(실제 {real_d['volatile']:.1f}%). 위기 블록을 빼도 변동장이 "
        "안 줄고 되레 늘었다 — 변동장 판정의 변동성 기준이 **상대(그 세계 내 상위 15%)**라, 위기를 빼 "
        "잔잔해진 세계에서도 똑같이 상위 15%가 생기고 그게 추세 애매한 잔물결에 붙어 변동장이 된다. "
        "**변동장 4.7배는 블록 조작(길이·선택·폭락분리)으로 못 푸는 구조적 산물.**\n"
        f"2. **그래도 얻는 것 — 두 코헤어런트 시험.** 폭락제외 = 상승 {reg['bull']:.1f}%(추세 시험), "
        f"폭락만 = 하락 {crash['bear']:.1f}%(방어 시험). 흩뿌려 섞지 않고 추세장/위기장을 **갈라** "
        "각각 코헤어런트하게 평가 → 일지 §III-2의 진짜 값은 '분포 맞추기'가 아니라 이 **이원화**다.\n"
        "\n**→ 국면 격차 3실험 종합**: 블록 부트스트랩은 실제 국면 분포(특히 변동장)를 못 살린다 — "
        "길이·국면샘플링·폭락분리 셋 다 변동장을 못 잡았다(구조적 한계). 그러니 합성 분포를 억지로 "
        "맞추려 말고: ⓐ **합성을 진단지로 격하**(일지 §III-2, 합성 한계 인정) + ⓑ 필요하면 **폭락 분리로 "
        "추세/방어 2트랙** 평가. 근본 우회(cGAN 등)는 검증 부담 큰 별도 트랙.\n"
        "\n재현: `.venv/Scripts/python.exe -m app.lab.textbook.textbook_crash_split`\n"
    )
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("".join(L))


if __name__ == "__main__":
    run()
