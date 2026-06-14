"""단일목적 TPE ↔ CMA-ES 비교 — 같은 시드/같은 데이터로 답이 모이는지.

v1.x 시즌(야생 7마리 합류 = 13마리)에서 cma_es 주석이 약속한 "TPE vs CMA-ES
답이 모이는지 비교 후 채택"을 실측. prepare_data를 1번만 호출해 양쪽에 주입
(yfinance/fight_dca 중복 차단). 가중치 비율이 거의 같고 1등 잔고가 ±0.5%
이내면 = 단일목적 답이 둘 다 수렴 = 둘 중 아무거나 채택 가능.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.academy.study import tpe, cma_es
from app.academy.study.nsga3 import evaluate_balances
from app.backend.genes.signals import ALL_GENES

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

SEED = 42
TRIALS = 2000


def _norm(weights: list[float]) -> dict[str, float]:
    total = sum(weights) or 1.0
    return {g: w / total for g, w in zip(ALL_GENES, weights)}


def _genes_str(norm: dict[str, float], thr: float = 0.10) -> str:
    main = sorted([(g, p) for g, p in norm.items() if p > thr],
                  key=lambda x: x[1], reverse=True)
    return " · ".join(f"{g} {p*100:.0f}%" for g, p in main) or "분산"


def _run_one(engine, name: str, loaded_gyms, dca) -> dict:
    print(f"\n▶ {name} seed={SEED} trials={TRIALS} ...")
    t0 = time.perf_counter()
    study, _, _ = engine.run_study(
        trials=TRIALS, seed=SEED, loaded_gyms=loaded_gyms, dca=dca,
    )
    weights, bals, summary = engine.champion_balances(study, loaded_gyms, dca)
    elapsed = time.perf_counter() - t0
    norm = _norm(weights)
    print(f"  1등 #{summary['trial']} · 잔고 합 {summary['balance_sum']/10000:.1f}만 "
          f"({elapsed:.1f}s)")
    print(f"  주력 (≥10%): {_genes_str(norm)}")
    return {"name": name, "summary": summary, "norm": norm, "bals": bals,
            "elapsed": elapsed}


def main() -> None:
    print("=== 단일목적 TPE ↔ CMA-ES 비교 — 13마리 풀, 잔고 합 max ===")
    print(f"시드 {SEED} · trials {TRIALS} · 가중치 {len(ALL_GENES)}차원\n")

    loaded_gyms, dca = tpe.prepare_data()       # cma_es와 동일 — 한 번만

    results = [
        _run_one(tpe, "TPE", loaded_gyms, dca),
        _run_one(cma_es, "CMA-ES", loaded_gyms, dca),
    ]

    # 기준점 — 현 챔피언 동일가중 VOL+REV_RSI+REV_BB
    champ_w = [1.0 if g in ("VOL", "REV_RSI", "REV_BB") else 0.0 for g in ALL_GENES]
    champ_bals = evaluate_balances(champ_w, {}, loaded_gyms, dca, seed_krw=tpe.SEED_KRW)
    champ_sum = sum(b["strat"] for b in champ_bals.values())
    dca_sum = sum(b["dca"] for b in champ_bals.values())

    print("\n=== 비교 ===")
    print(f"  현 챔피언 (VOL+REV_RSI+REV_BB 동일) {champ_sum/10000:.1f}만 · "
          f"성실이 {dca_sum/10000:.1f}만\n")
    print("  엔진    | 잔고 합   | 챔피언 대비 | 주력")
    print("  --------|-----------|------------|------------------")
    for r in results:
        diff = r["summary"]["balance_sum"] - champ_sum
        sign = "+" if diff >= 0 else ""
        print(f"  {r['name']:<7} | {r['summary']['balance_sum']/10000:7.1f}만 | "
              f"{sign}{diff/10000:>6.1f}만   | {_genes_str(r['norm'])}")

    # 답 수렴도 = 잔고 합 격차 (%) + 가중치 코사인 유사도
    sums = [r["summary"]["balance_sum"] for r in results]
    spread_pct = (max(sums) - min(sums)) / (sum(sums) / 2) * 100

    import math
    a = [results[0]["norm"][g] for g in ALL_GENES]
    b = [results[1]["norm"][g] for g in ALL_GENES]
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    cos = dot / (na * nb) if na and nb else 0.0

    print(f"\n  잔고 합 격차: ±{spread_pct/2:.2f}%")
    print(f"  가중치 코사인 유사도: {cos:.3f}  (1.0=완전동일, 0=직교)")
    if spread_pct / 2 < 0.5 and cos > 0.95:
        verdict = "수렴 양호 — 둘 다 같은 답 근방, 아무거나 채택 가능"
    elif spread_pct / 2 < 2.5:
        verdict = "수렴 보통 — 잔고는 비슷하나 주력 시그널은 미세 차"
    else:
        verdict = "수렴 안 됨 — multi-modal 또는 trials 부족"
    print(f"  → {verdict}")

    print("\n=== 체육관별 (전략 잔고, 만원) ===")
    print("  체육관                | 성실이  | TPE     | CMA-ES")
    print("  ---------------------|---------|---------|--------")
    for gym_name in results[0]["bals"]:
        dca_v = results[0]["bals"][gym_name]["dca"] / 10000
        tpe_v = results[0]["bals"][gym_name]["strat"] / 10000
        cma_v = results[1]["bals"][gym_name]["strat"] / 10000
        print(f"  {gym_name:<20} | {dca_v:6.1f}  | {tpe_v:6.1f}  | {cma_v:6.1f}")


if __name__ == "__main__":
    main()
