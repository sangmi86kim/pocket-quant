"""v2 리그 상·하위 시그널 비중 + 아레나(국면)별 성과 분석 — 시즌 3 준비 (코덱스안 2번).

[왜] 시즌 3은 "최적화 더 돌리기"가 아니라 "v2 데이터를 분석해 더 좋은 신호를 잡는" 방향.
어떤 포켓퀀트(신호) 비중이 상위 트레이더를 만들었나, 그리고 그 신호가 어느 시험 조건
(졸업시험/빅토리 로드 (OOS)/평행세계/사천왕)에서 실제로 돈을 벌었나를 본다.

[데이터]
  - 리그 결과: app/league/results/season_v2_top30_league.json (후보별 4아레나 종료잔고)
  - 가중치 원본: app/academy/training/results/classroom_top30_20260615_v2.json (params=13신호 가중치)
  후보 120명(TPE/CMA-ES/GP/NSGA ×30)을 (그룹, trial)로 매칭.

[방법]
  1) 가중치를 후보별 합=1로 정규화 → '비중'.
  2) 종합 순위 = 4아레나 백분위의 평균. 상위 30 vs 하위 30의 평균 비중 비교(lift).
  3) 신호×아레나 스피어만 상관 = "이 신호 비중이 높을수록 그 아레나서 잘하나"(국면별 성과).

[주의] 이건 생존자/상관 분석 — 인과 아님. fANOVA(학습목적 중요도)와 과녁이 다를 수 있다
(train/test 미스매치 가설, worklog 2026-06-16).

[출력] 콘솔 + app/lab/reports/season/v2_signal_analysis.{md,png}(신호×아레나 상관 히트맵).
실행: .venv/Scripts/python.exe -m app.lab.season.v2_signal_analysis
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from app.pocket.signals import SIGNAL_NAMES

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

LEAGUE_JSON = "app/league/results/season_v2_top30_league.json"
WEIGHTS_JSON = "app/academy/training/results/classroom_top30_20260615_v2.json"

ARENAS = ["exam", "oos", "world", "holdout"]
ARENA_KR = {"exam": "졸업시험", "oos": "빅토리 로드 (OOS)", "world": "평행세계", "holdout": "사천왕"}

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "season")
OUT_MD = os.path.join(OUT_DIR, "v2_signal_analysis.md")
OUT_PNG = os.path.join(OUT_DIR, "v2_signal_analysis.png")
PNG_NAME = "v2_signal_analysis.png"


def _load() -> pd.DataFrame:
    """후보 120명 × (정규화 비중 13 + 아레나 잔고 4) 데이터프레임."""
    league = json.load(open(LEAGUE_JSON, encoding="utf-8"))
    wsrc = json.load(open(WEIGHTS_JSON, encoding="utf-8"))

    # (그룹, trial) -> 정규화 비중 dict
    weights = {}
    for cls in wsrc["classrooms"]:
        gname = cls["name"].replace("-III", "")   # 가중치파일 'NSGA-III' → 리그 그룹 'NSGA'
        for m in cls["topk"]:
            p = m["params"]
            tot = sum(p.values()) or 1.0
            norm = {name: p.get(f"w_{name}", 0.0) / tot for name in SIGNAL_NAMES}
            weights[(gname, m["trial"])] = norm

    recs = []
    for r in league["rows"]:
        if r.get("kind") != "candidate":
            continue
        key = (r["group"], r["trial"])
        w = weights.get(key)
        if w is None:
            continue
        rec = {"name": r["name"], "group": r["group"]}
        rec.update({a: float(r[a]) for a in ARENAS})
        rec.update({f"w_{name}": w[name] for name in SIGNAL_NAMES})
        recs.append(rec)
    return pd.DataFrame(recs)


def _baselines(league: dict) -> dict:
    """기준선(어플삭제단 중앙값·성실이) 아레나별 잔고 — 비교선."""
    out = {}
    for r in league["rows"]:
        if r.get("kind") == "candidate":
            continue
        out[r["name"]] = {a: float(r[a]) for a in ARENAS}
    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = _load()
    n = len(df)

    # 종합 순위 = 4아레나 백분위 평균 (스케일이 다른 아레나를 공정 결합)
    pct = pd.DataFrame({a: df[a].rank(pct=True) for a in ARENAS})
    df["overall_pct"] = pct.mean(axis=1)
    df = df.sort_values("overall_pct", ascending=False).reset_index(drop=True)

    k = 30   # 상/하위 30명 (각 25%)
    top, bot = df.head(k), df.tail(k)
    wcols = [f"w_{s}" for s in SIGNAL_NAMES]
    prof = pd.DataFrame({
        "상위30_비중%": 100 * top[wcols].mean().values,
        "하위30_비중%": 100 * bot[wcols].mean().values,
    }, index=SIGNAL_NAMES)
    prof["lift%p"] = prof["상위30_비중%"] - prof["하위30_비중%"]
    prof = prof.sort_values("lift%p", ascending=False)

    # 신호 × 아레나 스피어만 상관
    corr = pd.DataFrame(
        {a: [df[f"w_{s}"].corr(df[a], method="spearman") for s in SIGNAL_NAMES]
         for a in ARENAS},
        index=SIGNAL_NAMES,
    )

    # 그룹별 아레나 평균 잔고
    grp = df.groupby("group")[ARENAS].mean().round(0)

    league = json.load(open(LEAGUE_JSON, encoding="utf-8"))
    base = _baselines(league)

    # ── 콘솔 ──
    print(f"=== v2 리그 시그널 분석 (후보 {n}명, 종합순위=4아레나 백분위 평균) ===\n")
    print("[상·하위 30 평균 비중 / lift]")
    print(prof.round(1).to_string())
    print("\n[신호 × 아레나 스피어만 상관]")
    print(corr.round(2).rename(columns=ARENA_KR).to_string())
    print("\n[그룹별 아레나 평균 잔고(만원)]")
    print((grp / 10000).round(0).rename(columns=ARENA_KR).to_string())

    _draw_heatmap(corr)
    _write_md(n, prof, corr, grp, base)
    print(f"\n[저장] {OUT_MD}\n[저장] {OUT_PNG}")


def _draw_heatmap(corr: pd.DataFrame) -> None:
    """신호(행) × 아레나(열) 스피어만 상관 히트맵."""
    mat = corr.values
    fig, ax = plt.subplots(figsize=(6.4, 8.2))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(ARENAS)))
    ax.set_xticklabels([ARENA_KR[a] for a in ARENAS])
    ax.set_yticks(range(len(SIGNAL_NAMES)))
    ax.set_yticklabels(SIGNAL_NAMES)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.35 else "black", fontsize=8)
    ax.set_title("신호 비중 × 아레나 성과 스피어만 상관\n(+빨강=비중↑일수록 그 아레나서 잘함)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Spearman ρ")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _write_md(n, prof, corr, grp, base) -> None:
    md = ["# v2 리그 시그널 분석 — 상·하위 비중 + 아레나별 성과 (시즌 3 준비)\n",
          f"> 후보 **{n}명**(TPE/CMA-ES/GP/NSGA ×30) · 4아레나(졸업시험/빅토리 로드 (OOS)/평행세계/사천왕) · "
          "13신호 정규화 비중. 종합순위 = 4아레나 백분위 평균.\n",
          "> ⚠️ 생존자/상관 분석 = 인과 아님. fANOVA(학습목적)와 과녁이 다를 수 있음(train/test 미스매치).\n",
          f"\n![신호×아레나 상관]({PNG_NAME})\n",
          "\n## [1] 상·하위 30 평균 비중 (lift 내림차순)\n\n",
          "| 신호 | 상위30 비중% | 하위30 비중% | lift%p |\n|---|--:|--:|--:|\n"]
    for s, row in prof.iterrows():
        md.append(f"| {s} | {row['상위30_비중%']:.1f} | {row['하위30_비중%']:.1f} | {row['lift%p']:+.1f} |\n")
    md.append("\n## [2] 신호 × 아레나 스피어만 상관 (비중↑ ↔ 그 아레나 성과)\n\n")
    md.append("| 신호 | " + " | ".join(ARENA_KR[a] for a in ARENAS) + " |\n")
    md.append("|---|" + "--:|" * len(ARENAS) + "\n")
    for s in corr.index:
        md.append(f"| {s} | " + " | ".join(f"{corr.loc[s, a]:+.2f}" for a in ARENAS) + " |\n")
    md.append("\n## [3] 그룹별 아레나 평균 잔고 (만원)\n\n")
    md.append("| 그룹 | " + " | ".join(ARENA_KR[a] for a in ARENAS) + " |\n")
    md.append("|---|" + "--:|" * len(ARENAS) + "\n")
    for g, row in grp.iterrows():
        md.append(f"| {g} | " + " | ".join(f"{row[a]/10000:.0f}" for a in ARENAS) + " |\n")
    for bname, bvals in base.items():
        md.append(f"| _{bname}_ | " + " | ".join(f"{bvals[a]/10000:.0f}" for a in ARENAS) + " |\n")
    md.append(
        "\n## 읽는 법\n\n"
        "- **[1] lift** = 상위 후보가 하위보다 더 실은 신호. 양수 클수록 '상위권의 색깔'.\n"
        "- **[2] 상관** = 빨강(+)이면 그 신호 비중을 키울수록 그 아레나서 잔고가 높음, 파랑(−)이면 반대.\n"
        "  아레나마다 부호가 갈리면 그 신호는 국면 의존(어떤 시장에선 약).\n"
        "- **[3]** = 어느 학습법(샘플러)이 어느 아레나서 강한지. 기준선(어플삭제단·성실이)이 비교선.\n"
        "\n재현: `.venv/Scripts/python.exe -m app.lab.season.v2_signal_analysis`\n"
    )
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("".join(md))


if __name__ == "__main__":
    main()
