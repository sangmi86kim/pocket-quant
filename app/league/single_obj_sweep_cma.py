"""CMA-ES 단일목적 5시드 분산 — `search.cma_es.run_study` 시드별 호출 + 안정성 보고.

`single_obj_sweep.py`(TPE 6차원판)의 CMA-ES·13차원판. 비교(`single_obj_compare.py`)에서
TPE↔CMA-ES가 ±4% 격차로 "수렴 안 됨" 판정이 나왔으니, CMA-ES 자체가 시드 노이즈인지
답이 결정적인지 5시드로 확인한다 (시드 간 폭으로 판정).

수렴 판정 기준 (sweep_seeds 패턴):
  ±0.5% 이내 = 수렴 양호
  ±2.5% 이내 = 수렴 보통
  그 이상     = 들쭉날쭉 (trials 부족 / sampler 노이즈 / multi-modal)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from app.academy.study import cma_es
from app.academy.study.nsga3 import evaluate_balances
from app.backend.genes.signals import ALL_GENES

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

SEEDS = [42, 7, 11, 19, 23]
TRIALS = 2000
OUT_MD = _ROOT / "reports" / "single_obj_sweep_cma.md"


def _norm_weights(weights: list[float]) -> dict[str, float]:
    total = sum(weights) or 1.0
    return {g: w / total for g, w in zip(ALL_GENES, weights)}


def _genes_str(norm: dict[str, float]) -> str:
    main = sorted([(g, p) for g, p in norm.items() if p > 0.1],
                  key=lambda x: x[1], reverse=True)
    return " · ".join(f"{g} {p*100:.0f}%" for g, p in main) or "분산"


def _verdict(spread_pct: float) -> str:
    if spread_pct < 0.5:
        return "수렴 양호 — CMA-ES가 결정적 답에 모음"
    if spread_pct < 2.5:
        return "수렴 보통 — 가중치는 안정적이라도 잔고 합에 미세 차"
    return "들쭉날쭉 — sampler 노이즈/multi-modal, 단일 답으로 채택 어려움"


def main() -> None:
    print("=== 단일목적 CMA-ES × 5 시드 — 잔고 합 max + 안정성 ===")
    print(f"시드 {SEEDS} · trials {TRIALS} · sampler CMA-ES · "
          f"가중치 {len(ALL_GENES)}차원\n")

    # 데이터는 시드 무관 — 한 번 준비해 5번 재사용 (yfinance/fight_dca 중복 제거)
    loaded_gyms, dca = cma_es.prepare_data()

    results = []
    for seed in SEEDS:
        print(f"▶ seed={seed} ...", flush=True)
        t0 = time.perf_counter()
        study, _, _ = cma_es.run_study(
            trials=TRIALS, seed=seed, loaded_gyms=loaded_gyms, dca=dca,
        )
        weights, bals, summary = cma_es.champion_balances(study, loaded_gyms, dca)
        elapsed = time.perf_counter() - t0
        norm = _norm_weights(weights)
        print(f"  1등 #{summary['trial']:<5} 잔고 합 {summary['balance_sum']/10000:6.1f}만 "
              f"({elapsed:.1f}s)  {_genes_str(norm)}")
        results.append({
            "seed": seed, "trial": summary["trial"],
            "balance_sum": summary["balance_sum"],
            "per_gym": {gym: b["strat"] for gym, b in bals.items()},
            "weights_norm": norm,
            "elapsed": elapsed,
        })

    # 기준점 — 현 챔피언 동일가중 VOL+REV_RSI+REV_BB
    champ_w = [1.0 if g in ("VOL", "REV_RSI", "REV_BB") else 0.0 for g in ALL_GENES]
    champ_bals = evaluate_balances(champ_w, {}, loaded_gyms, dca, seed_krw=cma_es.SEED_KRW)
    champ_sum = sum(b["strat"] for b in champ_bals.values())
    dca_sum = sum(b["dca"] for b in champ_bals.values())

    sums = [r["balance_sum"] for r in results]
    mean = sum(sums) / len(sums)
    spread = (max(sums) - min(sums)) / mean * 100
    verdict = _verdict(spread / 2)

    # 시드 간 가중치 변동 — 코사인 유사도의 평균(쌍별)
    import math

    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    vecs = [[r["weights_norm"][g] for g in ALL_GENES] for r in results]
    pairs = [cos(vecs[i], vecs[j])
             for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    cos_mean = sum(pairs) / len(pairs) if pairs else 0.0

    print(f"\n=== 안정성 ===")
    print(f"  잔고 합 평균 {mean/10000:.1f}만 · 시드 간 폭 ±{spread/2:.2f}%")
    print(f"  가중치 시드 간 코사인 평균 {cos_mean:.3f}")
    print(f"  → {verdict}")
    print(f"\n=== 기준점 ===")
    print(f"  현 챔피언 (VOL+REV_RSI+REV_BB 동일) {champ_sum/10000:.1f}만 · "
          f"성실이 {dca_sum/10000:.1f}만")
    print(f"  CMA-ES 1등 평균 - 현 챔피언 = {(mean-champ_sum)/10000:+.1f}만")

    print(f"\n=== 시드별 가중치 (정규화 %) ===")
    print("  seed   " + "  ".join(f"{g:>9}" for g in ALL_GENES) + "  잔고 합")
    for r in results:
        cells = "  ".join(f"{r['weights_norm'][g]*100:9.1f}" for g in ALL_GENES)
        print(f"  {r['seed']:>4}  {cells}  {r['balance_sum']/10000:6.1f}만")

    # MD 저장
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    gene_cols = " | ".join(ALL_GENES)
    gene_align = " | ".join(["---:"] * len(ALL_GENES))
    md = [
        "# 단일목적 CMA-ES × 5 시드 — 결과 요약 (v1.x 13마리)",
        "",
        f"- 엔진: `app/academy/study/cma_es.py` (CMA-ES 단일목적 — 잔고 합 max)",
        f"- 시드: {SEEDS} · trials/시드: {TRIALS}",
        f"- 기준점: 현 챔피언 {champ_sum/10000:.1f}만, 성실이 {dca_sum/10000:.1f}만",
        "",
        "## 시드별 1등",
        "",
        "| 시드 | trial | 잔고 합 | 주력 (≥10%) | 소요 |",
        "|---:|---:|---:|---|---:|",
    ]
    for r in results:
        md.append(
            f"| {r['seed']} | #{r['trial']} | {r['balance_sum']/10000:.1f}만 | "
            f"{_genes_str(r['weights_norm'])} | {r['elapsed']:.1f}s |"
        )
    md += [
        "",
        "## 시드별 가중치 (정규화 %)",
        "",
        f"| 시드 | {gene_cols} |",
        f"|---:| {gene_align} |",
    ]
    for r in results:
        cells = " | ".join(f"{r['weights_norm'][g]*100:.1f}" for g in ALL_GENES)
        md.append(f"| {r['seed']} | {cells} |")
    md += [
        "",
        "## 안정성",
        "",
        f"- 잔고 합 평균: {mean/10000:.1f}만 (현 챔피언 대비 {(mean-champ_sum)/10000:+.1f}만)",
        f"- 시드 간 폭: ±{spread/2:.2f}%",
        f"- 가중치 시드 간 코사인 평균: {cos_mean:.3f}",
        f"- 판정: {verdict}",
    ]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nsaved: {OUT_MD.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
