"""졸업 시험 — 학교 top30을 실QQQ 6체육관에 응시시켜 졸업 성적표(md + 박스플랏)를 낸다.

[위치] 졸업시험은 **진단 전용(아이큐 테스트)** 이다 — 선발/게이트가 아니다. 선발은 학교
합성장 median 점수로 끝났고(training), 이 산출물은 "그 졸업생들이 진짜 시장 6국면에서
어떻게 서는가"를 눈으로 보는 성적표일 뿐이다. 진짜 줄세움은 리그(OOS·평행세계)에서 한다.

[입력] training이 뱉은 top30 json (`classroom_top30_*_v2.json`). 거기 담긴 trial별
`params`를 디코딩해 실QQQ 6체육관에 다시 돌린다 — stale할 수 있는 점수 키엔 안 의존한다.

[잣대] 후보 1명의 졸업 점수 = 6체육관 종료잔고의 **중앙값**(median 목적과 정합).
그래프는 matplotlib PNG 딱 2장: ① 반별 종합 비교 ② 체육관별 비교(6칸 한 장). 성실이(DCA) 기준선 포함.

[실행] .venv/Scripts/python.exe -m app.academy.exam.graduate
"""
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")                                  # 헤드리스 — 파일로만 저장
import matplotlib.pyplot as plt                         # noqa: E402
from matplotlib.lines import Line2D                     # noqa: E402
from matplotlib.patches import Patch                    # noqa: E402

matplotlib.rcParams["font.family"] = "Malgun Gothic"   # 한글 라벨(Windows)
matplotlib.rcParams["axes.unicode_minus"] = False

from app.academy.exam import all_gyms, gym_key          # noqa: E402
from app.academy.exam.grade import evaluate_balances    # noqa: E402
from app.academy.training.candidate import decode_params  # noqa: E402
from app.pocket.battle import fight_dca, terminal_balance  # noqa: E402
from app.pocket.signals import SIGNAL_NAMES             # noqa: E402
from app.world.data_loader import load_gyms             # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
TRAIN_RESULTS = ROOT / "app" / "academy" / "training" / "results"
REPORTS_DIR = ROOT / "app" / "academy" / "reports"
SEED_KRW = 1_000_000

# 실행 옵션은 모듈 상수로 둔다(argparse 없이). None이면 가장 최근 top30 파일을 자동 선택.
TOP30_PATH: Path | None = None

# 체육관키 → 성적표 열에 쓸 짧은 이름(콘솔 인코딩 영향 없는 표시용 라벨).
GYM_LABEL = {
    "dotcom": "닷컴",
    "gfc": "리먼",
    "rebound": "회복",
    "crash_v": "코로나",
    "bull": "상승",
    "chop": "횡보",
}
# 교실 고유색 — Spotfire 계열 카테고리 팔레트. 1차/보충은 같은 교실색에 채움만 달리한다
# (1차=빗금·연하게 / 보충=솔리드). 그룹명이 "TPE-1차"여도 베이스명 "TPE"로 색을 찾는다.
GROUP_COLORS = {
    "TPE": "#2e6db4",      # blue
    "CMA-ES": "#3fa34d",   # green
    "GP": "#7e5ca8",       # purple
    "NSGA": "#e58a1f",     # amber
    "성실이": "#c0392b",    # red (기준선)
}


def _base_name(group: str) -> str:
    """'TPE-1차'·'TPE-보충' → 'TPE' (색 조회용 베이스명)."""
    return group.replace("-1차", "").replace("-보충", "")


def _box_style(group: str) -> dict:
    """교실 고유색 + 단계 구분. 1차=빗금·연하게, 보충/단일=솔리드."""
    color = GROUP_COLORS.get(_base_name(group), "#7a8290")
    if group.endswith("-1차"):
        return dict(facecolor=color, alpha=0.30, edgecolor=color, hatch="////")
    return dict(facecolor=color, alpha=0.72, edgecolor=color, hatch=None)


def _latest_top30() -> Path:
    if TOP30_PATH is not None:
        if not _top30_compatible(TOP30_PATH):
            raise ValueError(f"top30 파일 사용 불가: {TOP30_PATH}")
        return TOP30_PATH
    cands = sorted(TRAIN_RESULTS.glob("classroom_top30_*_v2.json"))
    if not cands:
        raise FileNotFoundError(f"top30 파일 없음: {TRAIN_RESULTS}/classroom_top30_*_v2.json")
    compatible = [p for p in cands if _top30_compatible(p)]
    if not compatible:
        need = ", ".join(f"w_{g}" for g in SIGNAL_NAMES)
        raise FileNotFoundError(
            "현재 시그널 풀과 호환되는 top30 파일 없음 "
            f"(필요 가중치: {need}) 또는 비용 모델 메타데이터 없음. "
            "슬리피지/No-trade band 반영 후 top30 재선발 필요."
        )
    return compatible[-1]


def _cost_model_ready(top30: dict) -> bool:
    cost_model = top30.get("cost_model")
    return isinstance(cost_model, dict) and cost_model.get("complete") is True


def _classroom_topk(classroom: dict) -> list[dict]:
    """학기 산출물의 단일 후보 스키마(topk)를 읽는다."""
    topk = classroom.get("topk")
    if topk is None:
        raise KeyError(f"{classroom.get('name', '(unknown)')} 반에 topk 없음")
    return topk


def _candidate_sets(classroom: dict) -> list[tuple[str, list[dict]]]:
    """한 교실에서 졸업시험에 태울 후보 묶음들.

    새 2단계 학교 산출물은 phase1.topk와 최종 topk를 모두 남긴다. 둘 다 있으면
    1차/보충을 나란히 비교하고, 옛 파일처럼 최종 topk만 있으면 기존처럼 한 줄만 낸다.
    """
    name = classroom["name"].replace("NSGA-III", "NSGA")
    phase1 = classroom.get("phase1", {}).get("topk")
    final = _classroom_topk(classroom)
    if phase1 is None:
        return [(name, final)]
    return [(f"{name}-1차", phase1), (f"{name}-보충", final)]


def _missing_weights(item: dict) -> list[str]:
    params = item.get("params", {})
    return [f"w_{g}" for g in SIGNAL_NAMES if f"w_{g}" not in params]


def _top30_compatible(path: Path) -> bool:
    try:
        top30 = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not _cost_model_ready(top30):
        return False
    try:
        for classroom in top30.get("classrooms", []):
            for _label, topk in _candidate_sets(classroom):
                for item in topk:
                    if _missing_weights(item):
                        return False
    except KeyError:
        return False
    return True


def _candidate_gym_balances(item: dict, gyms, dca) -> dict[str, float]:
    """후보 1명을 6체육관에 응시 → {체육관키: 전략 종료잔고}."""
    missing = _missing_weights(item)
    if missing:
        raise ValueError(
            f"현재 시그널 풀({len(SIGNAL_NAMES)}마리)과 후보 params 불일치: "
            f"누락 {missing}. 14신호 top30 재선발 필요."
        )
    weights, params = decode_params(item["params"])
    balances = evaluate_balances(weights, params, gyms, dca, seed_krw=SEED_KRW)
    return {gym_key(name): row["strat"] for name, row in balances.items()}


def _stats(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    return {
        "n": int(len(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def build_payload(top30: dict, gyms, dca) -> dict:
    """top30(파일이든 메모리든) + 응시할 체육관/성실이 → 졸업 성적표 payload.

    파일 I/O·차트 생성과 분리해 둔다 — 스모크가 방금 키운 졸업생을 메모리째
    넘겨 이 조립(디코드→6체육관 채점→median)만 태울 수 있게(점검 그물)."""
    gym_keys = [gym_key(lg.gym.name) for lg in gyms]   # all_gyms 순서 유지
    # 성실이(DCA)는 후보와 무관하게 체육관마다 한 값 — 한 번만 잰다.
    dca_by_gym = {gym_key(lg.gym.name): terminal_balance(dca[lg.gym.name], SEED_KRW)
                  for lg in gyms}

    classrooms = []
    for classroom in top30["classrooms"]:
        for group, topk in _candidate_sets(classroom):
            members = []
            for item in topk:
                per_gym = _candidate_gym_balances(item, gyms, dca)
                members.append({
                    "trial": item.get("trial"),
                    "per_gym": per_gym,
                    "score": float(np.median([per_gym[k] for k in gym_keys])),
                })
            classrooms.append({"group": group, "members": members})

    return {
        "stamp": top30.get("stamp", "latest"),
        "top30_source": top30.get("_source", "(in-memory)"),
        "seed_krw": SEED_KRW,
        "gym_keys": gym_keys,
        "dca_by_gym": dca_by_gym,
        "dca_score": float(np.median([dca_by_gym[k] for k in gym_keys])),
        "classrooms": classrooms,
    }


def run() -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    top30_path = _latest_top30()
    top30 = json.loads(top30_path.read_text(encoding="utf-8"))
    top30["_source"] = str(top30_path)

    gyms = load_gyms(all_gyms())
    dca = {lg.gym.name: fight_dca(lg) for lg in gyms}
    payload = build_payload(top30, gyms, dca)
    stamp = payload["stamp"]

    overall_png = _overall_png(payload, stamp)
    by_gym_png = _by_gym_png(payload, stamp)
    _write_markdown(payload, stamp, overall_png, by_gym_png)
    return payload


def _draw_box(ax, payload: dict, value_of, dca_value: float, title: str) -> None:
    """한 axes에 반별 박스플랏 + 성실이(DCA) 점선 기준선을 그린다.

    value_of로 종합(6체육관 median)이든 한 체육관 종료잔고든 같은 틀을 재사용."""
    groups = [c["group"] for c in payload["classrooms"]]
    data = [np.array([value_of(m) for m in c["members"]], dtype=float) / 10000.0
            for c in payload["classrooms"]]
    bp = ax.boxplot(data, widths=0.6, patch_artist=True,
                    medianprops=dict(color="#1f2933", linewidth=1.6))
    for patch, group in zip(bp["boxes"], groups):
        style = _box_style(group)
        patch.set_facecolor(style["facecolor"])
        patch.set_alpha(style["alpha"])
        patch.set_edgecolor(style["edgecolor"])
        if style["hatch"]:
            patch.set_hatch(style["hatch"])
    for ln in bp["whiskers"] + bp["caps"]:
        ln.set_color("#52606d")
    dca_v = dca_value / 10000.0
    ax.axhline(dca_v, color=GROUP_COLORS["성실이"], ls="--", lw=1.4, alpha=0.85)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([f"{g}\n(n={len(d)})" for g, d in zip(groups, data)], fontsize=9)
    ax.set_ylabel("종료잔고 (만원)", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", color="#eef2f7", lw=1)
    ax.set_axisbelow(True)


def _has_phases(payload: dict) -> bool:
    return any(c["group"].endswith(("-1차", "-보충")) for c in payload["classrooms"])


def _legend_handles(payload: dict) -> list[Any]:
    """범례 — 성실이 점선 + (2단계면) 1차=빗금 / 보충=솔리드 설명."""
    handles: list[Any] = [
        Line2D([0], [0], color=GROUP_COLORS["성실이"], ls="--", lw=1.4,
               label="성실이(DCA)")
    ]
    if _has_phases(payload):
        handles += [
            Patch(facecolor="#7a8290", alpha=0.30, edgecolor="#7a8290",
                  hatch="////", label="1차 (보충 전)"),
            Patch(facecolor="#7a8290", alpha=0.72, edgecolor="#7a8290",
                  label="보충 (1차+보충)"),
        ]
    return handles


def _overall_png(payload: dict, stamp: str) -> Path:
    """① 종합 비교 — 반별 6체육관 median 종료잔고 분포 한 장."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    _draw_box(ax, payload, lambda m: m["score"], payload["dca_score"],
              "졸업 종합 — 반별 6체육관 median 종료잔고 (진단 전용)")
    ax.legend(handles=_legend_handles(payload), loc="lower left", fontsize=8,
              framealpha=0.9)
    fig.tight_layout()
    path = REPORTS_DIR / f"graduation_{stamp}_overall.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _by_gym_png(payload: dict, stamp: str) -> Path:
    """② 체육관별 비교 — 6체육관을 2×3 한 장에 (각 칸=한 체육관 반별 분포)."""
    keys = payload["gym_keys"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, k in zip(axes.flat, keys):
        label = GYM_LABEL.get(k, k)
        _draw_box(ax, payload, lambda m, k=k: m["per_gym"][k],
                  payload["dca_by_gym"][k], f"{label} 체육관")
    for ax in axes.flat[len(keys):]:    # 체육관이 6 미만이면 남는 칸 숨김
        ax.axis("off")
    fig.suptitle("체육관별 비교 — 반별 졸업생 종료잔고 분포",
                 fontsize=14, fontweight="bold")
    fig.legend(handles=_legend_handles(payload), loc="upper right",
               ncol=3, fontsize=10, framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = REPORTS_DIR / f"graduation_{stamp}_by_gym.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _ranked_table(lines: list[str], classrooms: list[dict], value_of, dca_value: float) -> None:
    """반별 분포 순위표(median 내림차순) + 성실이 기준선 행을 lines에 덧붙인다."""
    lines.append("| 반 | n | median | p25 | p75 | min | max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    ranked = sorted(
        classrooms,
        key=lambda c: float(np.median([value_of(m) for m in c["members"]])),
        reverse=True,
    )
    for c in ranked:
        s = _stats([value_of(m) for m in c["members"]])
        lines.append(
            f"| {c['group']} | {s['n']} | {s['median']/10000:.0f} | {s['p25']/10000:.0f} | "
            f"{s['p75']/10000:.0f} | {s['min']/10000:.0f} | {s['max']/10000:.0f} |"
        )
    lines.append(f"| 성실이(DCA) | 1 | {dca_value/10000:.0f} | · | · | · | · |")


def _write_markdown(payload: dict, stamp: str, overall_png: Path,
                    by_gym_png: Path) -> None:
    classrooms = payload["classrooms"]
    lines = [
        "# 🎓 졸업 시험 성적표 — 실QQQ 6체육관",
        "",
        f"> **진단 전용** (선발 아님 — 선발은 학교 합성장 median 점수로 끝). "
        f"top30 출처: `{Path(payload['top30_source']).name}` · 시드 100만원 · 잣대=종료잔고 median",
        f"> stamp: {stamp}",
        "",
        "## 종합 (후보별 6체육관 median 종료잔고 분포, 만원)",
        "",
        f"![종합 비교]({overall_png.name})",
        "",
    ]
    _ranked_table(lines, classrooms, lambda m: m["score"], payload["dca_score"])

    lines += [
        "",
        "## 체육관별 분석 (반별 졸업생 종료잔고 분포, 만원)",
        "",
        f"![체육관별 비교]({by_gym_png.name})",
        "",
        "> 각 체육관에서 반별 졸업생 분포. 점선=그 체육관 성실이(DCA). "
        "반 median이 성실이보다 낮으면 그 체육관이 그 반의 약점 과목.",
        "",
    ]
    for k in payload["gym_keys"]:
        label = GYM_LABEL.get(k, k)
        lines += [f"### {label} 체육관", ""]
        _ranked_table(lines, classrooms, lambda m, k=k: m["per_gym"][k],
                      payload["dca_by_gym"][k])
        lines.append("")

    md_path = REPORTS_DIR / f"graduation_{stamp}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    payload = run()
    print(f"reports_dir={REPORTS_DIR}", flush=True)
    print(f"graduation_{payload['stamp']}.md", flush=True)
    ranked = sorted(
        payload["classrooms"],
        key=lambda c: float(np.median([m["score"] for m in c["members"]])),
        reverse=True,
    )
    print("\n[졸업 종합 median, 만원]", flush=True)
    for c in ranked:
        med = float(np.median([m["score"] for m in c["members"]])) / 10000
        print(f"  {c['group']:<8} n={len(c['members']):>3} median={med:>8.0f}", flush=True)
    print(f"  {'성실이':<8} n=  1 median={payload['dca_score']/10000:>8.0f}", flush=True)


if __name__ == "__main__":
    main()
