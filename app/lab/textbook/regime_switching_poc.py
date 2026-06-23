"""생성 모델 PoC — Regime-Switching(마르코프 국면전환)으로 합성 지수 생성.

[왜] 블록 부트스트랩은 실제 국면 분포를 못 살린다(길이·샘플링·폭락분리 3실험 모두 변동장
4.7배 못 잡음 — 구조적 한계). 대안 후보 중 가장 가볍고 해석 가능한 게 regime-switching:
국면(상승/하락/횡보/변동)을 상태로 두고, 국면별 수익률 분포 + 전이확률을 실제에서 추정해
생성한다. 국면 지속성을 명시하므로 추세가 안 끊기고, 국면 비율을 전이행렬로 직접 통제한다.

[PoC 질문] 이렇게 생성한 합성을 다시 분류하면 국면 분포가 블록 부트스트랩보다 실제에
가까운가? 특히 변동장(블록 방식이 4.7배로 못 잡던 것)이 잡히나?

[방법]
  1. 실제 QQQ 일별 국면 라벨(regime.py) → 국면별 일일 로그수익률 평균μ·표준편차σ 추정.
  2. 라벨 시퀀스에서 전이행렬 P(4x4)·국면 비율 π 추정.
  3. 마르코프 체인으로 상태열 생성 → 상태별 정규분포에서 수익률 샘플 → 누적 → 합성 가격.
  4. 합성 N권을 재분류해 국면 분포를 실제·블록부트스트랩과 비교.

[한계] 정규분포 가정(팻테일 미반영), 변동장 판정은 상대변동성이라 재분류서 여전히 흔들릴 수
있음 — 그걸 보는 게 PoC.

[출력] app/lab/outputs/textbook/regime_switching_poc/ 에 png 1장 + md 리포트.
실행: .venv/Scripts/python.exe -m app.lab.textbook.regime_switching_poc
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
STATES = ["bull", "bear", "sideways", "volatile"]
SIDX = {s: i for i, s in enumerate(STATES)}
LABEL_COLOR = {"bull": "#2e7d32", "bear": "#c62828",
               "sideways": "#9e9e9e", "volatile": "#f9a825"}
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "textbook", "regime_switching_poc")


def _dist(counter):
    tot = sum(counter.values()) or 1
    return {k: 100.0 * counter.get(k, 0) / tot for k in STATES}


def _fit():
    """실제 QQQ에서 국면별 μ·σ, 전이행렬 P, 국면비율 π 추정."""
    qqq = get_prices("QQQ", tb.DATA_START, tb.DATA_END)
    ret = (qqq / qqq.shift(1)).apply(np.log)
    labels = classify_daily(qqq)
    ret_a = ret.reindex(labels.index)
    lab = labels.to_numpy()

    mu = np.zeros(4)
    sigma = np.zeros(4)
    for s in STATES:
        r = ret_a[labels == s].dropna().to_numpy()
        mu[SIDX[s]] = r.mean()
        sigma[SIDX[s]] = r.std()

    P = np.zeros((4, 4))
    for a, b in zip(lab[:-1], lab[1:]):
        P[SIDX[a], SIDX[b]] += 1
    P = P / P.sum(axis=1, keepdims=True)
    pi = np.array([np.mean(lab == s) for s in STATES])
    real_d = _dist(collections.Counter(lab))
    return mu, sigma, P, pi, real_d


def _gen_world(mu, sigma, P, pi, n_days, seed):
    """마르코프 체인으로 합성 가격 한 권 생성."""
    rng = np.random.default_rng(seed)
    s = rng.choice(4, p=pi)
    rets = np.empty(n_days)
    for t in range(n_days):
        rets[t] = rng.normal(mu[s], sigma[s])
        s = rng.choice(4, p=P[s])
    price = 100.0 * np.exp(np.cumsum(rets))
    return pd.Series(price, index=pd.bdate_range("2001-01-01", periods=n_days))


def _classify_eval(prices, n_days):
    lab = classify_daily(prices)
    return lab[lab.index >= prices.index[tb.WARMUP_TDAYS]]


def run():
    mu, sigma, P, pi, real_d = _fit()
    n_days = tb.WARMUP_TDAYS + tb.EVAL_TDAYS
    print("국면별 일일 μ(연율%)·σ(연율%):")
    for s in STATES:
        print(f"  {REGIME_LABELS[s]}: μ {mu[SIDX[s]]*252*100:+6.1f}%  σ {sigma[SIDX[s]]*np.sqrt(252)*100:5.1f}%")

    # regime-switching 합성
    rs_counter = collections.Counter()
    example = None
    for seed in range(N_WORLDS):
        w = _gen_world(mu, sigma, P, pi, n_days, seed)
        if seed == 0:
            example = w
        rs_counter.update(_classify_eval(w, n_days).values)
    rs_d = _dist(rs_counter)

    # 블록 부트스트랩 비교 (현재 make_world)
    bb_counter = collections.Counter()
    for seed in range(N_WORLDS):
        g = tb.make_world(seed=seed)
        bb_counter.update(_classify_eval(g.prices, n_days).values)
    bb_d = _dist(bb_counter)

    def dist_to_real(d):
        return sum(abs(d[k] - real_d[k]) for k in STATES)

    print("\n          " + "  ".join(f"{REGIME_LABELS[k]}" for k in STATES) + "  | 실제거리")
    for name, d in [("실제", real_d), ("블록부트스트랩", bb_d), ("Regime-Switching", rs_d)]:
        print(f"{name:16s} " + "  ".join(f"{d[k]:4.1f}%" for k in STATES)
              + f"  | {dist_to_real(d):4.1f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    _fig(real_d, bb_d, rs_d, example)
    _write_md(real_d, bb_d, rs_d, mu, sigma, dist_to_real)
    print(f"[저장] {OUT_DIR}")


def _fig(real_d, bb_d, rs_d, example):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5),
                                  gridspec_kw={"width_ratios": [2.2, 1.3]})
    groups = [("실제", real_d, 1.0, None),
              ("블록부트스트랩", bb_d, 0.5, "//"),
              ("Regime-Switching", rs_d, 0.5, "xx")]
    xs = np.arange(len(STATES))
    w = 0.8 / len(groups)
    for i, (g, d, alpha, hatch) in enumerate(groups):
        off = (i - (len(groups) - 1) / 2) * w
        ax.bar(xs + off, [d[k] for k in STATES], w,
               color=[LABEL_COLOR[k] for k in STATES], alpha=alpha,
               edgecolor="black", linewidth=0.4, hatch=hatch)
        ax.bar([-9], [0], w, label=g, color="gray", alpha=alpha, hatch=hatch)
    ax.set_xlim(-0.6, len(STATES) - 0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([REGIME_LABELS[k] for k in STATES])
    ax.set_ylabel("비율 (%)")
    ax.set_title("국면 분포 — 실제 vs 블록부트스트랩 vs Regime-Switching", fontsize=11)
    ax.legend(fontsize=8)

    ax2.plot(np.arange(len(example)), example.to_numpy(), color="#37474f", lw=1.2)
    ax2.axvline(tb.WARMUP_TDAYS - 0.5, color="navy", alpha=0.4, lw=1.1)
    ax2.set_title("Regime-Switching 합성 예시 1권", fontsize=11)
    ax2.set_xlabel("거래일")
    ax2.set_ylabel("합성 지수")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "regime_switching_poc.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def _write_md(real_d, bb_d, rs_d, mu, sigma, dist_to_real):
    L = [
        "# 생성 모델 PoC — Regime-Switching(마르코프 국면전환)\n",
        f"> 실제 QQQ({tb.DATA_START}~{tb.DATA_END})에서 국면별 수익률·전이확률 추정 → 마르코프 체인으로 "
        f"합성 {N_WORLDS}권 생성 → 재분류해 국면 분포 비교. 블록 부트스트랩이 못 살린 분포를 푸는지 확인.\n",
        "\n![국면 분포 + 예시](regime_switching_poc.png)\n",
        "\n## 추정된 국면별 일일 수익률 (연율 환산)\n\n",
        "| 국면 | 평균 μ | 변동성 σ |\n|---|--:|--:|\n",
    ]
    for s in STATES:
        L.append(f"| {REGIME_LABELS[s]} | {mu[SIDX[s]]*252*100:+.1f}% | "
                 f"{sigma[SIDX[s]]*np.sqrt(252)*100:.1f}% |\n")

    L.append("\n## 국면 분포 비교\n\n")
    L.append("| 출처 | " + " | ".join(REGIME_LABELS[k] for k in STATES) + " | 실제거리 |\n")
    L.append("|---|" + "--:|" * (len(STATES) + 1) + "\n")
    for name, d in [("**실제**", real_d), ("블록부트스트랩(현재)", bb_d), ("Regime-Switching", rs_d)]:
        L.append(f"| {name} | " + " | ".join(f"{d[k]:.1f}%" for k in STATES)
                 + f" | {dist_to_real(d):.1f} |\n")

    better = dist_to_real(rs_d) < dist_to_real(bb_d)
    L.append(
        f"\n## 결론 (PoC) — 순진한 Regime-Switching도 분포를 못 살린다\n\n"
        f"- **실제거리**: 블록부트스트랩 {dist_to_real(bb_d):.1f} → Regime-Switching {dist_to_real(rs_d):.1f} "
        f"({'개선' if better else '오히려 악화'}). 하락장 과대생성({rs_d['bear']:.1f}% vs 실제 {real_d['bear']:.1f}%), "
        f"변동장 {rs_d['volatile']:.1f}%로 여전(실제 {real_d['volatile']:.1f}%).\n"
        "- **PoC가 드러낸 핵심**: 생성한 **숨은 상태**(상승/하락 의도)와 가격을 다시 분류한 **관측 국면**이 "
        "다르다. 분류기(regime.py)가 MA50/200·60일수익률 같은 **지연된 장기 함수**라, 국면을 명시 생성해도 "
        "재분류 분포가 의도대로 안 나온다. 게다가 분류된 '하락 day'에서 추정한 μ(-70%/yr)가 극단이라 생성 시 "
        "하락을 더 부풀린다.\n"
        "- **함의**: 분포 미스매치의 상당 부분은 **생성 방법이 아니라 분류기 정의**(상대변동성·지연 MA) 탓. "
        "그래서 cGAN 같은 정교한 생성기도 **같은 벽**에 부딪힐 수 있다 — '진짜 같은 생성'과 '재분류 분포 일치'는 "
        "별개 문제. 일지의 **진단지 격하**(합성 분포에 집착 말기)가 더 현실적이라는 근거.\n"
        "\n**PoC 판정**: 순진한 RS는 블록 부트스트랩 대비 이득 없음. 살리려면 ⓐ 국면 지속(전이행렬을 더 끈적하게)·"
        "ⓑ 팻테일(t분포)·ⓒ GARCH 변동성 군집·ⓓ 분류기 자체 재검토가 필요. 큰 투자 전 '실전 OOS 전이' 검증이 "
        "관문 — 더 진짜 같은 합성이 실전을 더 잘 맞춘다는 보장은 아직 없다.\n"
        "\n재현: `.venv/Scripts/python.exe -m app.lab.textbook.regime_switching_poc`\n"
    )
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("".join(L))


if __name__ == "__main__":
    run()
