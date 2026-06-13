"""단일목적 5시드 분산 — `engine.tpe.run_study` 시드별 호출 + 안정성 보고.

엔진(`app/backend/engine/tpe.py`)이 탐색·1등 추출까지 책임 — 본 어댑터는 5번 호출,
시드별 1등 표 작성, 시드 간 폭(%)으로 수렴 판정 + reports/single_obj_sweep.md 저장.

수렴 판정 기준 (sweep_seeds 패턴):
  ±0.5% 이내 = 수렴 양호
  ±2.5% 이내 = 수렴 보통
  그 이상     = 들쭉날쭉 (trials 부족 / sampler 노이즈)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from app.backend.engine import tpe
from app.backend.engine.nsga3 import evaluate_balances
from app.backend.genes.signals import ALL_GENES

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")    # type: ignore[union-attr]
    except Exception:
        pass

SEEDS = [42, 7, 11, 19, 23]
TRIALS = 2000
OUT_MD = _ROOT / "reports" / "single_obj_sweep.md"


def _norm_weights(weights: list[float]) -> dict[str, float]:
    total = sum(weights) or 1.0
    return {g: w / total for g, w in zip(ALL_GENES, weights)}


def _genes_str(norm: dict[str, float]) -> str:
    main = sorted([(g, p) for g, p in norm.items() if p > 0.1],
                  key=lambda x: x[1], reverse=True)
    return " · ".join(f"{g} {p*100:.0f}%" for g, p in main) or "분산"


def _verdict(spread_pct: float) -> str:
    if spread_pct < 0.5:
        return "수렴 양호 — TPE가 진짜 답 근방에 모음"
    if spread_pct < 2.5:
        return "수렴 보통 — 가중치 분포는 안정적이라도 잔고 합엔 미세 차"
    return "들쭉날쭉 — TPE 노이즈 큼, 답이 안정적이지 않음"


def main() -> None:
    print("=== 단일목적 TPE × 5 시드 — 잔고 합 max + 안정성 ===")
    print(f"시드 {SEEDS} · trials {TRIALS} · sampler TPE · 가중치 6차원\n")

    # 데이터는 시드 무관 — 한 번 준비해 5번 재사용 (yfinance/fight_dca 중복 제거)
    loaded_gyms, dca = tpe.prepare_data()

    results = []
    for seed in SEEDS:
        print(f"▶ seed={seed} ...", flush=True)
        t0 = time.perf_counter()
        study, _, _ = tpe.run_study(
            trials=TRIALS, seed=seed, loaded_gyms=loaded_gyms, dca=dca,
        )
        weights, bals, summary = tpe.champion_balances(study, loaded_gyms, dca)
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

    # 기준점 (5시드 다 같은 loaded_gyms/dca 사용 — 마지막 시드 거 그대로)
    champ_w = [1.0 if g in ("VOL", "REV_RSI", "REV_BB") else 0.0 for g in ALL_GENES]
    champ_bals = evaluate_balances(champ_w, {}, loaded_gyms, dca, seed_krw=tpe.SEED_KRW)
    champ_sum = sum(b["strat"] for b in champ_bals.values())
    dca_sum = sum(b["dca"] for b in champ_bals.values())

    sums = [r["balance_sum"] for r in results]
    mean = sum(sums) / len(sums)
    spread = (max(sums) - min(sums)) / mean * 100
    verdict = _verdict(spread / 2)

    print(f"\n=== 안정성 ===")
    print(f"  잔고 합 평균 {mean/10000:.1f}만 · 시드 간 폭 ±{spread/2:.2f}%")
    print(f"  → {verdict}")
    print(f"\n=== 기준점 ===")
    print(f"  현 챔피언 (VOL+REV_RSI+REV_BB 동일) {champ_sum/10000:.1f}만 · "
          f"성실이 {dca_sum/10000:.1f}만")
    print(f"  단일목적 1등 평균 - 현 챔피언 = {(mean-champ_sum)/10000:+.1f}만")

    print(f"\n=== 시드별 가중치 (정규화 %) ===")
    print("  seed   " + "  ".join(f"{g:>6}" for g in ALL_GENES) + "  잔고 합")
    for r in results:
        cells = "  ".join(f"{r['weights_norm'][g]*100:6.1f}" for g in ALL_GENES)
        print(f"  {r['seed']:>4}  {cells}  {r['balance_sum']/10000:6.1f}만")

    # MD 저장
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    md = [
        "# 단일목적 TPE × 5 시드 — 결과 요약",
        "",
        f"- 엔진: `app/backend/engine/tpe.py` (TPE 단일목적 — 잔고 합 max)",
        f"- 시드: {SEEDS} · trials/시드: {TRIALS}",
        f"- 기준점: 현 챔피언 {champ_sum/10000:.1f}만, 성실이 {dca_sum/10000:.1f}만",
        "",
        "## 시드별 1등",
        "",
        "| 시드 | trial | 잔고 합 | 주력 | 소요 |",
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
        "| 시드 | DD | VOL | MA | MOM | REV_RSI | REV_BB |",
        "|---:|---:|---:|---:|---:|---:|---:|",
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
        f"- 판정: {verdict}",
    ]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nsaved: {OUT_MD.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
