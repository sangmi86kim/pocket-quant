"""성적 좋은 트레이더들이 어떤 시그널에 무게를 줬나 — 피처 임포턴스.

다음 시즌 시드/탐색공간을 좁히기 전에, "장기 실데이터에서 잘한 놈들"의 가중치
공통점을 본다. 장기 잣대 = OOS 11년 + 사천왕 hold-out (체육관·평행세계는 합성·단기라
제외 — 우리 목표가 장기 투자라서).

피처 임포턴스를 두 각도로 본다:
  1) 상위권 평균 가중치 vs 전체 평균 (lift) — 이긴 놈들이 어디에 몰빵했나.
  2) 가중치 ↔ 장기점수 스피어만 상관 — 그 시그널을 키울수록 장기 성적이 오르나.
     (가중치는 Σw로 나눈 합=1 구성비라 서로 음의 상관이 끼는 한계 있음 — 방향만 참고.)

입력은 리그 결과(점수) + top30 소스(숫자 가중치). 실행:
  .venv/Scripts/python.exe -m app.league.operations.feature_importance
"""
import json
from pathlib import Path

import numpy as np

from app.academy.training.candidate import decode_params
from app.pocket.signals import SIGNAL_NAMES

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "app" / "league" / "results"
LEAGUE_JSON = RESULTS_DIR / "season_v2_top30_league.json"
TOP30_JSON = (ROOT / "app" / "academy" / "training" / "results"
              / "classroom_top30_20260615_v2.json")
MD_OUT = RESULTS_DIR / "season_v2_feature_importance.md"

LONGTERM_ARENAS = ("oos", "holdout")   # 장기 실데이터만
TOP_FRACTION = 0.25                    # 상위 25% = 성적 좋은 놈들


def _weight_map() -> dict[tuple[str, int], np.ndarray]:
    """(group, trial) → 정규화 가중치 벡터(합=1, SIGNAL_NAMES 순)."""
    top30 = json.loads(TOP30_JSON.read_text(encoding="utf-8"))
    out = {}
    for classroom in top30["classrooms"]:
        group = classroom["name"].replace("NSGA-III", "NSGA")
        for item in classroom["topk"]:
            weights, _params = decode_params(item["params"])
            arr = np.array(weights, dtype=float)
            total = arr.sum()
            out[(group, item["trial"])] = arr / total if total > 0 else arr
        # decode_params는 SIGNAL_NAMES 순서를 보장 — combine_positions와 동일 계약.
    return out


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """값 → [0,1] 백분위(동점은 평균 순위). 아레나 스케일 통일용."""
    order = values.argsort()
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    # 동점 평균 처리
    _, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum - 1) / 2.0
    ranks = avg[inv]
    return ranks / (len(values) - 1) if len(values) > 1 else ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """순위 상관 = 순위에 매긴 피어슨."""
    ra = _percentile_rank(a)
    rb = _percentile_rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def analyze() -> dict:
    league = json.loads(LEAGUE_JSON.read_text(encoding="utf-8"))
    wmap = _weight_map()

    rows = []
    for r in league["rows"]:
        if r["kind"] != "candidate":
            continue
        w = wmap.get((r["group"], r["trial"]))
        if w is None:
            continue
        rows.append({"group": r["group"], "trial": r["trial"], "w": w,
                     **{a: float(r[a]) for a in LONGTERM_ARENAS}})

    n = len(rows)
    W = np.array([r["w"] for r in rows])                      # (n, 13)
    # 장기점수 = 아레나별 백분위 평균 (OOS·hold-out 스케일 다르므로)
    arena_pct = {a: _percentile_rank(np.array([r[a] for r in rows]))
                 for a in LONGTERM_ARENAS}
    longterm = np.mean([arena_pct[a] for a in LONGTERM_ARENAS], axis=0)

    k = max(1, int(round(n * TOP_FRACTION)))
    top_idx = np.argsort(longterm)[::-1][:k]
    top_mask = np.zeros(n, dtype=bool)
    top_mask[top_idx] = True

    feats = []
    for j, name in enumerate(SIGNAL_NAMES):
        mean_all = float(W[:, j].mean())
        mean_top = float(W[top_mask, j].mean())
        feats.append({
            "signal": name,
            "mean_all": mean_all,
            "mean_top": mean_top,
            "lift": mean_top - mean_all,
            "spearman": _spearman(W[:, j], longterm),
        })
    feats.sort(key=lambda f: f["mean_top"], reverse=True)

    top_groups = {}
    for i in top_idx:
        g = rows[i]["group"]
        top_groups[g] = top_groups.get(g, 0) + 1

    return {"n": n, "k": k, "feats": feats, "top_groups": top_groups,
            "rows": rows, "longterm": longterm, "top_mask": top_mask}


def _write_md(res: dict) -> None:
    feats = res["feats"]
    lines = ["# Season v2 — 피처 임포턴스 (장기 성적 상위권)", ""]
    lines.append(f"- 후보 {res['n']}명 중 장기(OOS+hold-out) 백분위 평균 상위 "
                 f"{res['k']}명을 '성적 좋은 놈들'로 봄")
    grp = ", ".join(f"{g} {c}" for g, c in sorted(res["top_groups"].items(),
                                                   key=lambda kv: -kv[1]))
    lines.append(f"- 상위권 교실 구성: {grp}")
    lines.append("")
    lines.append("| 시그널 | 상위권 평균 | 전체 평균 | lift | 장기점수 상관 |")
    lines.append("|---|---:|---:|---:|---:|")
    for f in feats:
        lines.append(f"| {f['signal']} | {f['mean_top']*100:.1f}% | "
                     f"{f['mean_all']*100:.1f}% | {f['lift']*100:+.1f}%p | "
                     f"{f['spearman']:+.2f} |")
    lines.append("")
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    res = analyze()
    print(f"후보 {res['n']}명 · 상위권 {res['k']}명 (장기 OOS+hold-out 백분위 평균 기준)")
    grp = ", ".join(f"{g} {c}" for g, c in sorted(res["top_groups"].items(),
                                                  key=lambda kv: -kv[1]))
    print(f"상위권 교실 구성: {grp}\n")
    print(f"{'시그널':<10} {'상위권':>8} {'전체':>8} {'lift':>8} {'상관':>7}")
    for f in res["feats"]:
        print(f"{f['signal']:<10} {f['mean_top']*100:>7.1f}% {f['mean_all']*100:>7.1f}%"
              f" {f['lift']*100:>+7.1f}p {f['spearman']:>+7.2f}")
    _write_md(res)
    print(f"\nmd={MD_OUT}")


if __name__ == "__main__":
    main()
