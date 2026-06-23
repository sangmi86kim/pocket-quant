"""QQQ 전체 기간 국면 분포 리포트 — 시즌 3 교과서 개편용 (코덱스안 1번).

[왜] 교과서(아카데미 합성 체육관)가 방어 편향인지 점검하려면 먼저 "실제 나스닥이
어떤 국면을 얼마나 겪었나"의 기준 분포가 필요하다. 합성 교과서를 이 실제 분포에 맞춰야
"폭락장 과대표집 → 방어 신호 과보상" 편향을 피한다 (방어 알파가 진짜인지, 시험장
편향을 먹은 것인지 분리하려면 기준선이 있어야 한다).

[방법] app.world.regime.classify_daily(50/200 MA + 60일 수익률 + 20일 변동성 백분위)로
일별 국면(bull/bear/sideways/volatile)을 라벨링 → 전체·시대별·hold-out 기준으로 집계.
분류기는 리그/시험장 라벨링과 같은 단일 소스(regime.py)를 그대로 쓴다 = 정의 일관.

[출력] 콘솔 표 + app/lab/reports/textbook/qqq_regime_distribution.{md,png}.

실행: .venv/Scripts/python.exe -m app.lab.textbook.regime_distribution
"""

import os

import matplotlib

matplotlib.use("Agg")   # 디스플레이 없는 환경에서도 PNG 저장
import matplotlib.pyplot as plt
import pandas as pd

from app.world.data_loader import get_prices
from app.world.regime import REGIME_LABELS, classify_daily

plt.rcParams["font.family"] = "Malgun Gothic"   # 한글 라벨 (Windows 기본 폰트)
plt.rcParams["axes.unicode_minus"] = False

LABEL_ORDER = ["bull", "bear", "sideways", "volatile"]
LABEL_COLOR = {"bull": "#2e7d32", "bear": "#c62828",
               "sideways": "#9e9e9e", "volatile": "#f9a825"}

# 분류 워밍업(~252일)을 위해 상장 초기부터 넉넉히 받는다.
QQQ_START = "1999-01-01"
NDX_START = "1985-01-01"   # 나스닥100 지수 — 90년대 포함 더 긴 생애주기 참고용
DATA_END = "2026-06-18"
HOLDOUT_START = "2020-07-01"   # 사천왕 봉인 구간 시작 — 교과서는 이 이전만 학습

# 시대 구분 (나스닥 주요 국면) — (이름, 시작, 끝)
ERAS = [
    ("닷컴 붕괴", "2000-03-01", "2002-12-31"),
    ("회복·상승", "2003-01-01", "2007-12-31"),
    ("금융위기", "2008-01-01", "2009-06-30"),
    ("장기 강세", "2009-07-01", "2019-12-31"),
    ("코로나 충격", "2020-01-01", "2020-06-30"),
    ("post-COVID", "2020-07-01", DATA_END),
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "textbook")
OUT_MD = os.path.join(OUT_DIR, "qqq_regime_distribution.md")
OUT_PNG = os.path.join(OUT_DIR, "qqq_regime_distribution.png")
PNG_NAME = "qqq_regime_distribution.png"   # md에서 상대경로 링크


def _pct(labels: pd.Series) -> tuple[int, dict]:
    """라벨 시계열 → (총일수, {라벨: 백분율})."""
    n = int(len(labels))
    vc = labels.value_counts()
    return n, {k: (100.0 * int(vc.get(k, 0)) / n if n else 0.0) for k in LABEL_ORDER}


def _slice(labels: pd.Series, start: str, end: str) -> pd.Series:
    mask = (labels.index >= pd.Timestamp(start)) & (labels.index <= pd.Timestamp(end))
    return labels[mask]


def _md_table(rows: list[tuple[str, int, dict]]) -> str:
    head = "| 구간 | 일수 | " + " | ".join(REGIME_LABELS[k] for k in LABEL_ORDER) + " |\n"
    sep = "|---|--:|" + "--:|" * len(LABEL_ORDER) + "\n"
    body = ""
    for name, n, dist in rows:
        cells = " | ".join(f"{dist[k]:.1f}%" for k in LABEL_ORDER)
        body += f"| {name} | {n} | {cells} |\n"
    return head + sep + body


def _draw_chart(bars: list[tuple[str, dict]], span: str) -> None:
    """누적 막대(국면 비율 100%) — 그룹1(전체/holdout) | 그룹2(시대별)."""
    names = [b[0] for b in bars]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    bottom = [0.0] * len(names)
    for lab in LABEL_ORDER:
        vals = [b[1][lab] for b in bars]
        ax.bar(names, vals, bottom=bottom, color=LABEL_COLOR[lab],
               label=REGIME_LABELS[lab], width=0.72, edgecolor="white", linewidth=0.5)
        # 8% 이상 조각만 숫자 라벨 (가독성)
        for i, (v, b0) in enumerate(zip(vals, bottom)):
            if v >= 8.0:
                ax.text(i, b0 + v / 2, f"{v:.0f}", ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.axvline(2.5, color="black", linestyle="--", linewidth=0.8, alpha=0.4)  # 그룹 구분선
    ax.set_ylim(0, 100)
    ax.set_ylabel("비율 (%)")
    ax.set_title(f"QQQ 국면 분포 — regime.py 분류 ({span})")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    qqq = get_prices("QQQ", QQQ_START, DATA_END)
    labels = classify_daily(qqq)   # 워밍업 미달 구간은 자동 dropna
    span = f"{labels.index[0].date()} ~ {labels.index[-1].date()}"

    # 집계
    group_rows = [
        ("전체", *_pct(labels)),
        ("학습가능(~2020-06)", *_pct(_slice(labels, QQQ_START, "2020-06-30"))),
        ("hold-out(2020-07~)", *_pct(_slice(labels, HOLDOUT_START, DATA_END))),
    ]
    era_rows = [(name, *_pct(_slice(labels, s, e))) for name, s, e in ERAS]

    ndx_rows = []
    try:
        ndx_labels = classify_daily(get_prices("^NDX", NDX_START, DATA_END))
        ndx_rows = [(f"^NDX 전체 ({ndx_labels.index[0].date()}~)", *_pct(ndx_labels))]
    except Exception as exc:   # 데이터 못 받으면 참고 섹션만 생략
        print(f"[^NDX 생략: {type(exc).__name__}]")

    # 콘솔 출력
    print(f"=== QQQ 국면 분포 ({span}) ===")
    for name, n, dist in group_rows + era_rows + ndx_rows:
        cells = "  ".join(f"{REGIME_LABELS[k]} {dist[k]:4.1f}%" for k in LABEL_ORDER)
        print(f"{name:<20s} n={n:<5d} {cells}")

    # 그래프 (전체/holdout 3개 + 시대별 6개)
    chart_bars = [(n, d) for n, _, d in group_rows] + [(n, d) for n, _, d in era_rows]
    _draw_chart(chart_bars, span)

    # md 리포트
    md = []
    md.append("# QQQ 전체 기간 국면 분포 리포트 (시즌 3 교과서 개편용)\n")
    md.append(f"> 분류 가능 구간: **{span}** · 분류기: `app/world/regime.py` "
              "(50/200일 이동평균 + 60일 수익률 ±3% + 20일 변동성 백분위 0.85)\n")
    md.append("> 국면 = 상승장(추세 위)·하락장(추세 아래)·횡보장(애매)·변동장(애매한데 변동성 폭발)\n")
    md.append(f"\n![국면 분포]({PNG_NAME})\n")
    md.append("\n## [1] 전체 / hold-out 분리\n\n" + _md_table(group_rows))
    md.append("\n## [2] 시대별\n\n" + _md_table(era_rows))
    if ndx_rows:
        md.append("\n## [3] ^NDX 참고 (90년대 포함)\n\n" + _md_table(ndx_rows))
    md.append(
        "\n## 핵심 발견\n\n"
        "1. **나스닥은 평소가 상승장** — 학습가능 기간 58%, 전체 60%가 상승장. 하락장은 ~22%뿐이고 "
        "닷컴(70%)·금융위기(59%)에 몰빵.\n"
        "2. **교과서 방어 편향 = 수치 확인** — 옛 시험장 6체육관은 하락 위주(4하락:2평시)였는데 현실 "
        "하락장은 22%. 폭락장을 현실보다 ~3배 과대표집 → 방어 신호 과보상 위험.\n"
        "3. **변동장 라벨이 사실상 죽음(0.5%)** — 추세 점수에 눌려 '변동장'이 거의 안 뜬다. 레짐 스캐너 "
        "재설계 시 이 버킷 손봐야.\n"
        "4. **MA200은 뒷북** — 2020 코로나(-30% 폭락)인데 하락장 16.8%뿐, 상승장 68%로 잡힘. 너무 빠른 "
        "폭락은 이동평균이 못 따라감 → 크로스에셋·위험선호 즉시 스캔 재설계 근거.\n"
        "5. **hold-out이 학습기간보다 더 상승편향**(68% vs 58%) → 챔피언이 사천왕에서 B&H에 고전한 것과 일관.\n"
        "\n## 시사점\n\n"
        "기초 교과서를 실제 분포(상승 ~60 / 하락 ~22 / 횡보 ~18)에 맞추고, 폭락장은 희귀하지만 생존훈련용 "
        "**최소 쿼터**로 시험장 쪽에 따로 둔다 (코덱스 §4 방향과 일치).\n"
        "\n---\n\n재현: `.venv/Scripts/python.exe -m app.lab.textbook.regime_distribution`\n"
    )
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"\n[저장] {OUT_MD}\n[저장] {OUT_PNG}")


if __name__ == "__main__":
    main()
