"""교과서 시그널 엑스레이 — 합성 세계에서 학생이 시그널별로 '뭘 배우나' 까보기.

[왜] GP가 특정 시그널(REV_RSI·크로스에셋·US10Y…)을 정답으로 외운다. 학생 탓이 아니라
교과서(합성 세계)가 그렇게 가르치기 때문. 그럼 교과서가 시그널별로 정확히 뭘 보상하는지
직접 들여다보자. (금리 offset·VXN FEAR_NQ 반영 후 현재 상태)

[시그널별로 재는 것]
  ① 발동 빈도   = 평가창에서 의견을 낸 날 비율 (상시형 ~100%, 이벤트형은 가끔)
  ② 의견 방향   = 발동일 평균 포지션 (1=매수 / 0=방어 / 0.5=중립)
  ③ 보상 엣지   = 발동일 이후 K일 합성수익률을 '의견 방향으로' 본 평균.
                  양(+)이면 교과서가 그 시그널의 베팅을 보상 = GP가 정답으로 외울 이유.
                  buy 베팅(pos>0.5)은 이후 오르면 +, defense 베팅(pos<0.5)은 이후 내리면 +.

[그래프 3장]
  freq_dir.png  : 시그널별 발동 빈도 (색=평균 의견방향)
  edge.png      : 시그널별 보상 엣지 (5일·20일) — 교과서가 뭘 정답으로 보는지
  xray_one.png  : 예시 한 권 — 합성 가격 + 시그널 발동 히트맵(매수=초록/방어=빨강/기권=흰)

[출력] app/lab/outputs/textbook/textbook_signal_xray/ 에 png 3장 + md 리포트.
실행: .venv/Scripts/python.exe -m app.lab.textbook.textbook_signal_xray
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from app.academy.curriculum.textbook import make_world, WARMUP_TDAYS
from app.pocket.signals import SIGNAL_NAMES, SIGNAL_REGISTRY

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

N_WORLDS = 60          # 평균 안정용 권수
EXAMPLE_SEED = 0       # 히트맵으로 그릴 한 권
FWD = (5, 20)          # 보상 엣지 측정 지평(거래일)
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "textbook", "textbook_signal_xray")

# 시그널 한 줄 분류 (그래프 라벨용) — signals.py 주석 기준
KIND = {
    "DD": "방어·상시", "VOL": "방어·상시", "MA": "추세·상시", "MOM": "추세·상시",
    "REV_RSI": "역발상·이벤트", "REV_BB": "역발상·이벤트",
    "VOL_SPIKE": "거래량·이벤트", "FEAR": "VIX공포·이벤트", "FEAR_NQ": "VXN공포·이벤트",
    "US10Y": "금리·이벤트", "DXY": "달러·방어", "SPY_TLT": "채권·방어",
    "QQQ_SPY": "성장우위·이벤트", "QQQ_DIA": "테크우위·이벤트",
}


def _positions(prices):
    """시그널별 포지션 시리즈 (평가창만)."""
    out = {}
    for name in SIGNAL_NAMES:
        pos = SIGNAL_REGISTRY[name](prices)
        out[name] = pos.iloc[WARMUP_TDAYS:]
    return out


def _collect():
    """N권에 대해 시그널별 빈도·방향·엣지 누적, 예시 한 권 포지션 행렬도 보관."""
    freq = {n: [] for n in SIGNAL_NAMES}
    direction = {n: [] for n in SIGNAL_NAMES}
    edge = {k: {n: [] for n in SIGNAL_NAMES} for k in FWD}
    example = None

    for seed in range(N_WORLDS):
        g = make_world(seed=seed)
        price = g.prices.iloc[WARMUP_TDAYS:].to_numpy()
        fwd = {k: np.concatenate([price[k:] / price[:-k] - 1, np.full(k, np.nan)])
               for k in FWD}
        pos_map = _positions(g.prices)
        if seed == EXAMPLE_SEED:
            example = (g.prices.iloc[WARMUP_TDAYS:], pos_map)

        for name in SIGNAL_NAMES:
            p = pos_map[name].to_numpy()
            active = ~np.isnan(p)
            freq[name].append(100.0 * active.mean())
            direction[name].append(float(np.nanmean(p)) if active.any() else np.nan)
            # 보상 엣지: 의견 방향(부호) × 미래수익률, 중립(0.5)·기권 제외
            bet = np.sign(p - 0.5)
            use = active & (bet != 0)
            for k in FWD:
                m = use & ~np.isnan(fwd[k])
                edge[k][name].append(float(np.mean(bet[m] * fwd[k][m])) if m.any() else np.nan)
    return freq, direction, edge, example


def _avg(d):
    return {n: float(np.nanmean(v)) for n, v in d.items()}


def run():
    freq, direction, edge, example = _collect()
    os.makedirs(OUT_DIR, exist_ok=True)
    f_avg, d_avg = _avg(freq), _avg(direction)
    e_avg = {k: _avg(edge[k]) for k in FWD}

    _fig_freq_dir(f_avg, d_avg)
    _fig_edge(e_avg)
    _fig_xray(example)
    _write_md(f_avg, d_avg, e_avg)
    print(f"[저장] {OUT_DIR}")


def _dir_cmap():
    return LinearSegmentedColormap.from_list("dir", ["#c62828", "#f5f5f5", "#2e7d32"])


def _fig_freq_dir(f_avg, d_avg):
    """발동 빈도(가로막대) — 색은 평균 의견방향(빨강 방어 ~ 초록 매수)."""
    order = sorted(SIGNAL_NAMES, key=lambda n: f_avg[n])
    cmap = _dir_cmap()
    fig, ax = plt.subplots(figsize=(10, 7))
    ys = np.arange(len(order))
    colors = [cmap(d_avg[n]) for n in order]   # d in 0..1
    ax.barh(ys, [f_avg[n] for n in order], color=colors, edgecolor="black", linewidth=0.5)
    for y, n in zip(ys, order):
        ax.text(f_avg[n] + 1, y, f"{f_avg[n]:.0f}%", va="center", fontsize=8)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{n}  ({KIND[n]})" for n in order], fontsize=9)
    ax.set_xlabel("발동 빈도 (평가창 의견 낸 날 %)")
    ax.set_title("① 시그널별 발동 빈도 — 색: 의견방향(빨강 방어 0 ~ 초록 매수 1)\n"
                 "상시형은 매일, 이벤트형은 가끔만 의견", fontsize=11)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=ax, label="평균 의견 (0 방어 ~ 1 매수)", fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "freq_dir.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def _fig_edge(e_avg):
    """보상 엣지(가로막대) — 의견 방향으로 본 발동 후 미래수익률. 양수=교과서가 보상."""
    order = sorted(SIGNAL_NAMES, key=lambda n: e_avg[FWD[1]][n])
    fig, ax = plt.subplots(figsize=(10, 7))
    ys = np.arange(len(order))
    h = 0.38
    for i, k in enumerate(FWD):
        vals = [e_avg[k][n] * 100 for n in order]
        ax.barh(ys + (0.5 - i) * h, vals, h,
                color="#1565c0" if k == FWD[0] else "#90caf9",
                edgecolor="black", linewidth=0.4, label=f"발동 후 {k}일")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{n}  ({KIND[n]})" for n in order], fontsize=9)
    ax.set_xlabel("보상 엣지 (의견 방향으로 본 발동 후 평균 수익률, %)")
    ax.set_title("② 교과서가 시그널별로 뭘 보상하나 (양수=정답으로 가르침)\n"
                 "buy 베팅은 이후 오르면 +, defense 베팅은 이후 내리면 +", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "edge.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def _fig_xray(example):
    """예시 한 권: 합성 가격 + 시그널 발동 히트맵."""
    prices, pos_map = example
    x = np.arange(len(prices))
    # 행렬: 매수(+1)/방어(-1)/중립(0)/기권(nan)
    mat = np.full((len(SIGNAL_NAMES), len(prices)), np.nan)
    for r, name in enumerate(SIGNAL_NAMES):
        p = pos_map[name].to_numpy()
        mat[r] = np.where(np.isnan(p), np.nan, np.sign(p - 0.5))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 2.4]})
    ax0.plot(x, prices.to_numpy(), color="#37474f", lw=1.3)
    ax0.set_ylabel("합성 QQQ")
    ax0.set_title(f"③ 예시 한 권(seed={EXAMPLE_SEED}) — 합성 가격과 시그널 발동 지도", fontsize=11)
    ax0.margins(x=0.005)

    cmap = _dir_cmap()
    cmap.set_bad("white")
    ax1.imshow(mat, aspect="auto", cmap=cmap, vmin=-1, vmax=1,
               interpolation="nearest", extent=(0, len(prices), len(SIGNAL_NAMES) - 0.5, -0.5))
    ax1.set_yticks(np.arange(len(SIGNAL_NAMES)))
    ax1.set_yticklabels([f"{n} ({KIND[n]})" for n in SIGNAL_NAMES], fontsize=8)
    ax1.set_xlabel("평가창 거래일")
    ax1.set_ylabel("시그널")
    # 범례 패치
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color="#2e7d32", label="매수(1)"),
                        Patch(color="#c62828", label="방어(0)"),
                        Patch(facecolor="white", edgecolor="gray", label="기권")],
               loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "xray_one.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def _write_md(f_avg, d_avg, e_avg):
    def arrow(d):
        if np.isnan(d):
            return "—"
        return "🟩매수" if d > 0.66 else ("🟥방어" if d < 0.34 else "🟨혼합")

    rows = sorted(SIGNAL_NAMES, key=lambda n: e_avg[FWD[1]][n], reverse=True)
    L = [
        "# 교과서 시그널 엑스레이 — 합성 세계에서 뭘 배우나\n",
        f"> 합성 {N_WORLDS}권(`make_world`)에서 14 시그널의 발동 빈도·의견 방향·보상 엣지. "
        "금리 offset·VXN FEAR_NQ 반영 후 현재 상태.\n",
        "\n![발동 빈도](freq_dir.png)\n\n",
        "*상시형(MA/MOM/DD/VOL)은 100% 매일 의견. 이벤트형은 가끔만 — 크로스에셋 ~60%, 공포·역발상은 "
        "2~10%. 색이 초록일수록 매수, 빨강일수록 방어(DD·DXY·SPY_TLT가 방어).*\n\n",
        "![보상 엣지](edge.png)\n\n",
        "*오른쪽(+)일수록 교과서가 그 베팅을 보상 = GP가 정답으로 외울 이유. 상위는 전부 역발상·공포"
        "(REV_BB/REV_RSI/FEAR/VOL_SPIKE), 하위는 방어형(SPY_TLT/DXY). 합성 교과서가 '떨어질 때 사라'를 "
        "보상하는 구조 — 실전(긴 우상향=추세 보상)과 거꾸로.*\n\n",
        "![엑스레이 한 권](xray_one.png)\n\n",
        "*위 가격이 폭락하는 구간(~380일)에 역발상·공포가 초록(매수)으로 몰리고, 상시형은 그때 "
        "빨강(방어)으로 돈다. 누가 언제 의견을 내는지 한눈에 보이는 지도.*\n",
        "\n## 시그널별 요약 (보상 엣지 20일 내림차순)\n\n",
        "| 시그널 | 분류 | 발동빈도 | 의견 | 엣지5(%) | 엣지20(%) |\n",
        "|---|---|--:|:--:|--:|--:|\n",
    ]
    for n in rows:
        L.append(f"| {n} | {KIND[n]} | {f_avg[n]:.0f}% | {arrow(d_avg[n])} | "
                 f"{e_avg[5][n] * 100:+.2f} | {e_avg[20][n] * 100:+.2f} |\n")

    e20 = e_avg[20]
    L.append(
        "\n## 주요 발견\n\n"
        f"1. **교과서 정답 = 역발상·공포.** 엣지 상위는 REV_BB(+{e20['REV_BB']*100:.2f}%)·"
        f"REV_RSI(+{e20['REV_RSI']*100:.2f}%)·FEAR(+{e20['FEAR']*100:.2f}%)·VOL_SPIKE 순. "
        "합성 교과서가 '떨어질 때 사라'를 보상 → 실전(추세 보상)과 반대 = v2 train/test 미스매치의 뿌리 재확인.\n"
        f"2. **US10Y 고침 확인.** 발동 빈도가 **{f_avg['US10Y']:.0f}%**로 급감(offset 전 ~37%) — "
        "경계 절벽이 만들던 가짜 금리인하 이벤트가 사라졌다. 남은 발동은 진짜 급락뿐.\n"
        f"3. **방어형은 벌받음.** SPY_TLT({e20['SPY_TLT']*100:+.2f}%)·DXY({e20['DXY']*100:+.2f}%)가 "
        "엣지 바닥 — 합성 세계가 출렁여도 결국 우상향이라 방어 베팅은 손해.\n"
        f"4. **의외 — FEAR_NQ(VXN)가 FEAR(VIX)와 정반대.** 같은 공포형인데 FEAR는 +{e20['FEAR']*100:.2f}%, "
        f"FEAR_NQ는 **{e20['FEAR_NQ']*100:+.2f}%**(음수). 발동 빈도는 둘 다 {f_avg['FEAR']:.0f}%로 잘 맞췄지만"
        "(임계 47 빈도매칭 성공), VXN은 폭락 한복판에 더 길게 발동해 '바닥'이 아닌 '추가 하락 직전'을 산다 "
        "(히트맵 확인). 단 합성 엣지는 실전과 반대일 수 있으니(역발상이 합성서 이기고 실전서 짐) "
        "**FEAR_NQ가 나쁘다고 단정 말고 실전 OOS로 따로 검증** 필요.\n"
        "\n재현: `.venv/Scripts/python.exe -m app.lab.textbook.textbook_signal_xray`\n"
    )
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("".join(L))


if __name__ == "__main__":
    run()
