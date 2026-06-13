"""5 시드 분산 실행 — 새 챔피언 후보 + 시드 안정성 확인 (06-13).

사용자 안: trials 크게(2000), pop 100, 시드 5개 다양 + 적응형 mutation ON.
HV-MA(5) 얼리스탑이 정체 시 자동으로 끊으므로 trials=2000은 상한.

산출:
  - 각 시드별: front 크기, 통과 후보 수, HV 수렴 곡선, mutation 궤적
  - 시드 간 비교: best 후보 잔고 합 (성실이 대비) — 시드 안정성 가늠
  - 라벨 후보 잔고 (Defensive/Balanced/Aggressive/Low-turnover) 6체육관 매트릭스
  - 산출물: reports/league_v1/sweep_seeds.md + .json

실행: python tools/sweep_seeds.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import optuna

from app.academy.study import nsga3
from app.backend.engine.battle import terminal_balance

SEEDS = [42, 7, 11, 19, 23]
TRIALS = 2000
POPULATION = 100
EARLY_STOP_WINDOW = 5
SEED_KRW = 1_000_000

OUT_MD = Path("reports/league_v1/sweep_seeds.md")
OUT_JSON = OUT_MD.with_suffix(".json")


def run_one(seed: int) -> dict:
    """한 시드 실행 → 요약 dict.
    front 전체에서 잔고 합 1등을 직접 추출 (사용자 안 '다른 장에서 회수' 일관)."""
    t0 = time.time()
    print(f"\n=== seed {seed} 시작 (trials≤{TRIALS}, pop {POPULATION}) ===")
    study, lg, dca, hv_cb, mut_cb = nsga3.run_study(
        TRIALS, seed=seed, population_size=POPULATION,
        early_stop_window=EARLY_STOP_WINDOW, adaptive_mutation=True)
    summary = nsga3.summarize_front(study, loaded_gyms=lg, dca=dca)

    # ── front 전체 후보의 잔고 합 계산 ──
    # 게이트(tolerance/turnover) 무시 — 사용자 안: 한 체육관 깊게 깨져도 다른 장
    # 회수 양수면 OK. 챔피언 후보 = 6체육관 잔고 합 최대.
    front_balances = []
    for t in study.best_trials:
        w, sig = nsga3.decode_params(t.params)
        b = nsga3.evaluate_balances(w, sig, lg, dca, SEED_KRW)
        sumv = sum(v["strat"] for v in b.values())
        sumdca = sum(v["dca"] for v in b.values())
        front_balances.append({
            "trial_number": int(t.number),
            "per_gym": {k: v["strat"] for k, v in b.items()},
            "합": sumv, "성실이_합": sumdca, "차": sumv - sumdca,
            "values": list(t.values),                          # 5국면 score + turnover
            "params": dict(t.params),
        })
    front_balances.sort(key=lambda r: -r["합"])
    top5 = front_balances[:5]

    elapsed = time.time() - t0
    n_trials_done = len(study.trials)
    front_size = summary["front_size"]
    top = top5[0] if top5 else None
    if top:
        print(f"  seed {seed} 완료 — {n_trials_done} trials, front {front_size}, "
              f"1등 #{top['trial_number']} 합 {top['합']//10000:,}만원 "
              f"vs 성실이 {top['성실이_합']//10000:,}만원 "
              f"(+{top['차']//10000:,}), {elapsed:.0f}초")
    else:
        print(f"  seed {seed} 완료 — front 비어있음, {elapsed:.0f}초")

    return {
        "seed": seed,
        "n_trials": n_trials_done,
        "front_size": front_size,
        "hv_curve": list(hv_cb.hv) if hv_cb else [],
        "hv_stopped": getattr(hv_cb, "stopped", False),
        "mut_history": list(mut_cb.history) if mut_cb else [],
        "top5": top5,            # 잔고 합 상위 5
        "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    results = []
    for seed in SEEDS:
        results.append(run_one(seed))

    # ── 비교 표 ──
    md = ["# 5 시드 분산 실행 — 새 챔피언 + 안정성 가늠 (06-13)", ""]
    md.append(f"- 설정: trials≤{TRIALS} · pop {POPULATION} · "
              f"HV-MA({EARLY_STOP_WINDOW}) 얼리스탑 · 적응형 mutation ON")
    md.append(f"- 시드: {SEEDS}")
    md.append("")

    md.append("## 시드별 요약")
    md.append("")
    md.append("| 시드 | trials | front | 1등 trial | 1등 잔고 합 | 성실이 합 | 차 | HV 최종 | 소요 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        hv_last = r["hv_curve"][-1] if r["hv_curve"] else 0
        if r["top5"]:
            t = r["top5"][0]
            tn = f"#{t['trial_number']}"
            sumv = t["합"] // 10000
            sumd = t["성실이_합"] // 10000
            diff = t["차"] // 10000
            md.append(f"| {r['seed']} | {r['n_trials']} | {r['front_size']} | "
                      f"{tn} | {sumv:,} | {sumd:,} | +{diff:,} | "
                      f"{hv_last:.4f} | {r['elapsed_s']:.0f}초 |")
        else:
            md.append(f"| {r['seed']} | {r['n_trials']} | {r['front_size']} | "
                      f"- | - | - | - | {hv_last:.4f} | {r['elapsed_s']:.0f}초 |")
    md.append("")

    # ── 시드별 잔고 합 1등의 체육관별 잔고 ──
    md.append("## 시드별 1등 후보 — 체육관 6개 잔고 (100만원 시드, 단위 만원)")
    md.append("")
    gym_order = list(next(iter(results[0]["top5"][0]["per_gym"].keys()
                                 for r in results if r["top5"]), []))
    if not gym_order and results and results[0]["top5"]:
        gym_order = list(results[0]["top5"][0]["per_gym"].keys())
    # 짧은 별명
    nick = {g: next(t for t in ("닷컴","금융위기","회복","코로나","상승","횡보")
                      if t[:2] in g) for g in gym_order}
    md.append("| 시드 | trial | " + " | ".join(nick[g] for g in gym_order) + " | 합 | 성실이 |")
    md.append("|---|---|" + "---:|" * (len(gym_order) + 2))
    for r in results:
        if not r["top5"]:
            continue
        t = r["top5"][0]
        cells = " | ".join(f"{t['per_gym'][g] // 10000:,}" for g in gym_order)
        md.append(f"| {r['seed']} | #{t['trial_number']} | {cells} | "
                  f"{t['합'] // 10000:,} | {t['성실이_합'] // 10000:,} |")
    md.append("")

    md.append("> 사용자 안: \"한 장에서 깨져도 다른 장에서 회수\" — 합산 잔고가 성실이보다 크면 OK.")
    md.append("> 시드 간 잔고 합이 비슷하면 수렴, 들쭉날쭉하면 노이즈.")
    md.append("")

    # ── 새 챔피언 후보 ──
    all_cands = [t for r in results for t in r["top5"]]
    all_cands.sort(key=lambda t: -t["합"])
    if all_cands:
        # 시드/trial 매칭
        seed_of = {}
        for r in results:
            for t in r["top5"]:
                seed_of[(r["seed"], t["trial_number"])] = r["seed"]
        winner = all_cands[0]
        winner_seed = next(r["seed"] for r in results if winner in r["top5"])
        md.append("## 새 챔피언 후보 (5 시드 통합 — 잔고 합 1등)")
        md.append("")
        md.append(f"- 시드 {winner_seed} · trial #{winner['trial_number']}")
        md.append(f"- 잔고 합 **{winner['합'] // 10000:,}만원** vs 성실이 "
                  f"**{winner['성실이_합'] // 10000:,}만원**")
        md.append(f"- 차이 **+{winner['차'] // 10000:,}만원** (6체육관 합산)")
        md.append("")
        md.append("**체육관별 잔고:**")
        for g in gym_order:
            v = winner["per_gym"][g]
            md.append(f"- {nick[g]}: {v // 10000:,}만원")
        md.append("")
        md.append("**가중치 (w_DD/VOL/MA/MOM/REV_RSI/REV_BB):**")
        ws = [winner["params"][f"w_{g}"] for g in
              ("DD", "VOL", "MA", "MOM", "REV_RSI", "REV_BB")]
        md.append(f"- {[round(w, 3) for w in ws]}")
        md.append("")
        md.append("현 챔피언(동일가중 VOL+REV_RSI+REV_BB, 사천왕 통과)과의 비교는")
        md.append("챔피언로드 ①②③ 풀 흐름으로 확정. 이건 \"훈련장 1등\" 단계.")
        md.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2,
                                     default=str), encoding="utf-8")
    print(f"\nsaved: {OUT_MD} (+ .json)")


if __name__ == "__main__":
    main()
