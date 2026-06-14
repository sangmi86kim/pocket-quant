"""단일목적 TPE ↔ CMA-ES ↔ GP 3-way 비교 — saturation 얼리스토핑 기반.

trial 수를 cap으로 두고 plateau-based early stop으로 각 sampler가 자기 한계까지
가도록 한다 (사용자 지시: "trial 횟수 상관없이 saturation"). 같은 시드·같은 데이터.
각 sampler가 얼만큼 trial을 써서 어디까지 모이는지가 본 비교의 핵심.

[정지 기준]
  patience 트라이얼 동안 best_value 상대 개선 < min_delta_pct이면 study.stop().
  GP는 trial당 O(n³) 커널 비용 때문에 별도 max_trials cap을 더 작게 둔다.
"""

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import optuna

from app.academy.study import tpe, cma_es, gp
from app.academy.study.nsga3 import evaluate_balances
from app.backend.genes.signals import ALL_GENES

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")    # type: ignore[union-attr]
    except Exception:
        pass

SEED = 42
PATIENCE = 300                 # plateau 정지 — 개선 없는 트라이얼 한도
MIN_DELTA_PCT = 0.0005         # 0.05% (잔고 800만 기준 ~4천원 = 가중치 미세조정 노이즈)
CAP_FAST = 5000                # TPE/CMA-ES 안전 상한 — saturate 못 하면 여기서 끊김
CAP_GP = 1500                  # GP는 trial당 GP regression O(n³)이라 별도 cap


class PlateauStopCallback:
    """patience 트라이얼 동안 best_value 상대 개선 < min_delta_pct면 study.stop().

    direction=maximize 가정. 첫 best 갱신 전까지는 카운트 시작 안 함.
    """
    def __init__(self, patience: int, min_delta_pct: float):
        self.patience = patience
        self.min_delta_pct = min_delta_pct
        self.last_best: float | None = None
        self.last_improve_trial: int = 0
        self.stopped_at: int | None = None
        self.saturate_trial: int | None = None     # 마지막 의미있는 개선 시점

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        try:
            best = study.best_value
        except ValueError:           # best가 아직 없음 (모든 trial fail)
            return
        n = trial.number
        if self.last_best is None:
            self.last_best = best
            self.last_improve_trial = n
            self.saturate_trial = n
            return
        rel = (best - self.last_best) / max(abs(self.last_best), 1.0)
        if rel > self.min_delta_pct:
            self.last_best = best
            self.last_improve_trial = n
            self.saturate_trial = n
        elif n - self.last_improve_trial >= self.patience:
            self.stopped_at = n
            study.stop()


def _norm(weights: list[float]) -> dict[str, float]:
    total = sum(weights) or 1.0
    return {g: w / total for g, w in zip(ALL_GENES, weights)}


def _genes_str(norm: dict[str, float], thr: float = 0.10) -> str:
    main = sorted([(g, p) for g, p in norm.items() if p > thr],
                  key=lambda x: x[1], reverse=True)
    return " · ".join(f"{g} {p*100:.0f}%" for g, p in main) or "분산"


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _run_one(engine, name: str, cap: int, loaded_gyms, dca) -> dict:
    print(f"\n▶ {name} seed={SEED} cap={cap} patience={PATIENCE} "
          f"min_delta={MIN_DELTA_PCT*100:.3f}% ...")
    stop_cb = PlateauStopCallback(PATIENCE, MIN_DELTA_PCT)

    # 진행 출력은 500트라이얼마다 (saturate trial과 stop trial은 콜백이 따로 추적)
    def progress(done: int, total: int, best_value: float) -> None:
        if done % 500 == 0:
            print(f"    [{done:>5}/{total}] best {best_value/10000:6.1f}만 · "
                  f"plateau {done - stop_cb.last_improve_trial}/{PATIENCE}")

    t0 = time.perf_counter()
    study, _, _ = engine.run_study(
        trials=cap, seed=SEED, loaded_gyms=loaded_gyms, dca=dca,
        on_progress=progress, extra_callbacks=[stop_cb],
    )
    weights, bals, summary = engine.champion_balances(study, loaded_gyms, dca)
    elapsed = time.perf_counter() - t0
    n_trials = len(study.trials)
    sat = stop_cb.saturate_trial or 0
    stopped_label = f"plateau stop @ {stop_cb.stopped_at}" if stop_cb.stopped_at \
                    else f"cap hit @ {n_trials}"
    print(f"  1등 #{summary['trial']} · 잔고 합 {summary['balance_sum']/10000:.1f}만 · "
          f"trials {n_trials} · saturate @ {sat} · {stopped_label} ({elapsed:.1f}s)")
    print(f"  주력 (≥10%): {_genes_str(_norm(weights))}")
    return {"name": name, "summary": summary, "norm": _norm(weights),
            "bals": bals, "elapsed": elapsed, "n_trials": n_trials,
            "saturate_trial": sat,
            "stopped_at": stop_cb.stopped_at, "cap": cap}


def main() -> None:
    print("=== 단일목적 TPE ↔ CMA-ES ↔ GP — saturation 비교 ===")
    print(f"시드 {SEED} · 가중치 {len(ALL_GENES)}차원 · "
          f"plateau patience={PATIENCE} min_delta={MIN_DELTA_PCT*100:.3f}%")
    print(f"cap: TPE/CMA-ES={CAP_FAST}, GP={CAP_GP} (GP는 trial당 O(n³) 비용)\n")

    loaded_gyms, dca = tpe.prepare_data()

    results = [
        _run_one(tpe,    "TPE",    CAP_FAST, loaded_gyms, dca),
        _run_one(cma_es, "CMA-ES", CAP_FAST, loaded_gyms, dca),
        _run_one(gp,     "GP",     CAP_GP,   loaded_gyms, dca),
    ]

    # 기준점 — 현 챔피언 동일가중 VOL+REV_RSI+REV_BB
    champ_w = [1.0 if g in ("VOL", "REV_RSI", "REV_BB") else 0.0 for g in ALL_GENES]
    champ_bals = evaluate_balances(champ_w, {}, loaded_gyms, dca, seed_krw=tpe.SEED_KRW)
    champ_sum = sum(b["strat"] for b in champ_bals.values())
    dca_sum = sum(b["dca"] for b in champ_bals.values())

    print("\n=== 비교 ===")
    print(f"  현 챔피언 (VOL+REV_RSI+REV_BB 동일) {champ_sum/10000:.1f}만 · "
          f"성실이 {dca_sum/10000:.1f}만\n")
    print("  엔진    | 잔고 합   | 챔피언 대비 | trials | saturate | 소요")
    print("  --------|-----------|------------|--------|----------|-------")
    for r in results:
        diff = r["summary"]["balance_sum"] - champ_sum
        sign = "+" if diff >= 0 else ""
        sat = r["saturate_trial"]
        sat_pct = sat / r["n_trials"] * 100 if r["n_trials"] else 0
        print(f"  {r['name']:<7} | {r['summary']['balance_sum']/10000:7.1f}만 | "
              f"{sign}{diff/10000:>6.1f}만   | {r['n_trials']:>6} | "
              f"#{sat:<4} ({sat_pct:>3.0f}%) | {r['elapsed']:>5.1f}s")
    print("  ※ saturate = 마지막으로 의미있는 개선이 일어난 trial 번호")
    print("    → '#X (Y%)' = 전체의 Y% 시점에서 사실상 답이 굳어졌고 이후는 plateau")

    print("\n  주력 (≥10%):")
    for r in results:
        print(f"    {r['name']:<7} : {_genes_str(r['norm'])}")

    # 답 수렴도
    sums = [r["summary"]["balance_sum"] for r in results]
    spread_pct = (max(sums) - min(sums)) / (sum(sums) / len(sums)) * 100
    vecs = [[r["norm"][g] for g in ALL_GENES] for r in results]
    pairs = [(i, j, _cos(vecs[i], vecs[j]))
             for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    cos_mean = sum(c for _, _, c in pairs) / len(pairs) if pairs else 0.0

    print(f"\n  잔고 합 폭: ±{spread_pct/2:.2f}% (3-way max-min)")
    print(f"  가중치 코사인 페어 평균: {cos_mean:.3f}")
    print(f"    {results[0]['name']}↔{results[1]['name']}: {pairs[0][2]:.3f}  "
          f"{results[0]['name']}↔{results[2]['name']}: {pairs[1][2]:.3f}  "
          f"{results[1]['name']}↔{results[2]['name']}: {pairs[2][2]:.3f}")

    print("\n=== 체육관별 (전략 잔고, 만원) ===")
    gyms = list(results[0]["bals"].keys())
    head = "  " + f"{'체육관':<20}" + " | 성실이 | " + " | ".join(f"{r['name']:>7}" for r in results)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for gym_name in gyms:
        dca_v = results[0]["bals"][gym_name]["dca"] / 10000
        cells = " | ".join(f"{r['bals'][gym_name]['strat']/10000:7.1f}" for r in results)
        print(f"  {gym_name:<20} | {dca_v:6.1f} | {cells}")


if __name__ == "__main__":
    main()
