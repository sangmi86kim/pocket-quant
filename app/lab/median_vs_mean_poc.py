"""mean vs median 줄세우기 비교 PoC.

질문(대표님): 학교 목적함수의 평균 잔고(mean_balance)를 중앙값(median)으로 바꾸면
후보 순위 — 특히 1등 챔피언 — 이 흔들리나?

방법: NSGA 재경기(최적화)는 하지 않는다. 확정된 v2 top30 후보들의 가중치를
실데이터 3개 무대에서 "한 번 더 채점"해 시장별 종료 잔고를 뽑고, 같은 후보 풀을
mean 으로 줄세운 순위 vs median 으로 줄세운 순위를 견준다.

여기서 mean/median 은 "한 후보가 여러 시장에서 받은 종료 잔고들"의 평균/중앙값이다
(그룹 분포의 중앙값이 아님). 평균의 약점 = 한 시장 대박이 평균을 통째로 끌어올림 →
median 은 그 한 방에 안 휘둘림. 그게 흔들림으로 잡히는지 본다.

max(최대 수익)는 목적이 아니라 참고지표로만 출력한다 — 운빨 한 방 보상이라 학습엔 안 씀.

무대(전부 실데이터, 시장별 잔고 분포):
  - oos:     victory_road OOS 11개 연도   (챔피언이 실제 선발된 관문)
  - exam:    공식 6체육관
  - holdout: 사천왕 hold-out 7개 라운드

실행: .venv/Scripts/python.exe -m app.lab.median_vs_mean_poc
"""
import json
from pathlib import Path

import numpy as np

from app.academy.exam import all_gyms
from app.league import elite_four as EF
from app.league import victory_road as VR
from app.league.v2.classroom_league import SEED_KRW, TOP30_PATH
from app.pocket.battle import _score_position, terminal_balance
from app.pocket.signals import SIGNAL_NAMES, combine_positions, positions_with_params
from app.world.data_loader import get_prices, load_gyms

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "median_vs_mean_poc"
CHAMPION = "NSGA-t5938"  # v2 확정 챔피언 — 추적용


def _decode_weights(params: dict) -> list[float]:
    """v2 저장 params(w_*) → 현행 SIGNAL_NAMES 순서 가중치.

    v2(2026-06-15)는 13신호라 시즌3에서 추가된 FEAR_NQ(w_FEAR_NQ)가 없다.
    빠진 신호는 0으로 채운다 — 그 후보는 원래 그 신호를 안 썼으니 기여 0이 충실한 복원."""
    return [float(params.get(f"w_{g}", 0.0)) for g in SIGNAL_NAMES]


def _load_candidates() -> list[dict]:
    """v2 top30 후보들의 (name, group, weights)."""
    top30 = json.loads(TOP30_PATH.read_text(encoding="utf-8"))
    cands = []
    for classroom in top30["classrooms"]:
        group = classroom["name"].replace("NSGA-III", "NSGA")
        for item in classroom["topk"]:
            weights = _decode_weights(item["params"])
            cands.append({
                "name": f"{group}-t{item['trial']}",
                "group": group,
                "weights": weights,
                "label": item.get("weights", ""),
            })
    return cands


def _arena_markets() -> dict[str, list[tuple]]:
    """무대별 (loaded, base_positions) 페어 리스트. 시그널 포지션은 1회만 계산해 재사용."""
    exam_gyms = load_gyms(all_gyms())
    prices = get_prices(VR.TICKER, "1999-03-10", "2026-06-09")
    oos_loaded = {year: VR._loaded_window(prices, year) for year in VR.OOS_YEARS}
    holdout_loaded = {name: EF._loaded_window(prices, start, end)
                      for name, start, end in EF.ROUNDS}
    return {
        "oos": [(lg, positions_with_params(lg.prices))
                for lg in oos_loaded.values()],
        "exam": [(lg, positions_with_params(lg.prices)) for lg in exam_gyms],
        "holdout": [(lg, positions_with_params(lg.prices))
                    for lg in holdout_loaded.values()],
    }


def _market_balances(weights: list[float], markets: list[tuple]) -> list[float]:
    """후보 하나가 한 무대의 시장마다 받는 종료 잔고 리스트."""
    out = []
    for loaded, base_pos in markets:
        pos = combine_positions(base_pos, weights)
        out.append(float(terminal_balance(_score_position(pos, loaded), SEED_KRW)))
    return out


def _rank_desc(values: list[float]) -> list[int]:
    """큰 값이 1등. 동률은 평균 순위."""
    arr = np.asarray(values, dtype=float)
    order = np.argsort(-arr, kind="stable")
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1)
    # 동률 평균 순위 보정
    _, inv, counts = np.unique(-arr, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return [float(avg[inv[i]]) for i in range(len(arr))]


def _spearman(a: list[float], b: list[float]) -> float:
    ra = _rank_desc(a)
    rb = _rank_desc(b)
    ra = np.asarray(ra) - np.mean(ra)
    rb = np.asarray(rb) - np.mean(rb)
    denom = np.sqrt(np.sum(ra**2) * np.sum(rb**2))
    return float(np.sum(ra * rb) / denom) if denom else float("nan")


def _analyze_arena(cands: list[dict], arena_key: str) -> dict:
    names = [c["name"] for c in cands]
    means = [c[arena_key]["mean"] for c in cands]
    medians = [c[arena_key]["median"] for c in cands]
    maxes = [c[arena_key]["max"] for c in cands]

    r_mean = _rank_desc(means)
    r_median = _rank_desc(medians)
    r_max = _rank_desc(maxes)

    idx_mean1 = int(np.argmin(r_mean))
    idx_median1 = int(np.argmin(r_median))
    idx_max1 = int(np.argmin(r_max))

    shifts = sorted(
        ({"name": names[i], "r_mean": r_mean[i], "r_median": r_median[i],
          "shift": r_median[i] - r_mean[i]} for i in range(len(cands))),
        key=lambda d: abs(d["shift"]), reverse=True,
    )

    champ_i = next((i for i, n in enumerate(names) if n == CHAMPION), None)
    champ = None
    if champ_i is not None:
        champ = {"rank_mean": r_mean[champ_i], "rank_median": r_median[champ_i],
                 "rank_max": r_max[champ_i]}

    return {
        "n_candidates": len(cands),
        "n_markets": len(cands[0][arena_key]["balances"]),
        "spearman_mean_vs_median": _spearman(means, medians),
        "top1_mean": names[idx_mean1],
        "top1_median": names[idx_median1],
        "top1_changed": names[idx_mean1] != names[idx_median1],
        "top1_max": names[idx_max1],
        "top1_max_eq_mean": names[idx_max1] == names[idx_mean1],
        "biggest_shifts": shifts[:5],
        "champion": champ,
    }


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cands = _load_candidates()
    arenas = _arena_markets()

    for c in cands:
        for key, markets in arenas.items():
            bals = _market_balances(c["weights"], markets)
            c[key] = {
                "balances": bals,
                "mean": float(np.mean(bals)),
                "median": float(np.median(bals)),
                "worst": float(np.min(bals)),
                "max": float(np.max(bals)),
            }

    report = {key: _analyze_arena(cands, key) for key in arenas}
    payload = {"champion": CHAMPION, "arenas": report,
               "candidates": [{k: c[k] for k in ("name", "group", "oos", "exam", "holdout")}
                              for c in cands]}
    (OUT_DIR / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _fmt(report: dict) -> str:
    lines = ["# mean vs median 줄세우기 비교 (재경기 없음, v2 top30 재채점)", ""]
    lines.append(f"챔피언 추적 대상: {CHAMPION}")
    lines.append("- Spearman ≈ 1.00 이면 두 줄세우기가 사실상 같음 → median 으로 바꿔도 순위 안 흔들림")
    lines.append("- max(최대수익)는 참고지표 (목적 아님)")
    lines.append("")
    for key in ("oos", "exam", "holdout"):
        a = report[key]
        lines.append(f"## {key}  (후보 {a['n_candidates']}명 / 시장 {a['n_markets']}개)")
        lines.append(f"- Spearman(mean vs median) = **{a['spearman_mean_vs_median']:.4f}**")
        lines.append(f"- 1등: mean → `{a['top1_mean']}`  /  median → `{a['top1_median']}`  "
                     f"→ **{'바뀜!' if a['top1_changed'] else '동일'}**")
        lines.append(f"- (참고) max 1등 → `{a['top1_max']}`  "
                     f"→ mean 1등과 {'같음' if a['top1_max_eq_mean'] else '다름'}")
        if a["champion"]:
            ch = a["champion"]
            lines.append(f"- 챔피언 {CHAMPION} 순위: mean {ch['rank_mean']:.0f}등 / "
                         f"median {ch['rank_median']:.0f}등 / max {ch['rank_max']:.0f}등")
        lines.append("- 순위 변동 큰 후보 top5 (median등 − mean등):")
        for s in a["biggest_shifts"]:
            lines.append(f"  - {s['name']}: mean {s['r_mean']:.0f}등 → median "
                         f"{s['r_median']:.0f}등 ({s['shift']:+.0f})")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    report = run()
    text = _fmt(report)
    (OUT_DIR / "report.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"\nsaved: {OUT_DIR / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
