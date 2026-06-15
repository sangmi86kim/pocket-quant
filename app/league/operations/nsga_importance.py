"""NSGA 스터디 자체의 피처 임포턴스 — Optuna fANOVA.

생존자 120명을 surrogate로 마이닝하면 가중치가 합=1 구성비라 상관이 오염된다
(US10Y가 무게 1등인데 동인은 아니었던 그 함정). 대신 **NSGA 탐색이 본 6180 trial
전체**에서 fANOVA로 "각 시그널 가중치가 목적값 분산을 얼마나 설명하나"를 뽑으면,
그게 곧 결정변수 임포턴스다. 이게 'NSGA 돌리면 나오는 피처 임포턴스'.

주의: 이 임포턴스는 **학습 목적(합성 아카데미 3목적)** 기준이다 — OOS·사천왕 장기
성적 기준이 아니라, "탐색이 뭘 보고 해를 갈랐나"를 말한다. 둘은 상보적이다.

목적: mean_balance(평균 누적자산↑) · worst_balance(최악 누적자산↑) · turnover(매매량↓).

실행: .venv/Scripts/python.exe -m app.league.operations.nsga_importance
"""
import json
from pathlib import Path

import optuna
from optuna.importance import FanovaImportanceEvaluator, get_param_importances

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / "app" / "academy" / "training" / "results" / "classroom_nsga3_20260615_010219.db"
MD_OUT = ROOT / "app" / "league" / "results" / "season_v2_nsga_importance.md"

OBJECTIVES = [(0, "mean_balance"), (1, "worst_balance"), (2, "turnover")]
FANOVA_SEED = 0   # 재현성 — fANOVA 내부 랜덤포레스트 시드 고정


def analyze() -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{DB.as_posix()}"
    name = optuna.study.get_all_study_names(storage)[0]
    study = optuna.load_study(study_name=name, storage=storage)

    evaluator = FanovaImportanceEvaluator(seed=FANOVA_SEED)
    per_obj = {}
    for idx, label in OBJECTIVES:
        imp = get_param_importances(
            study, evaluator=evaluator, target=lambda t, i=idx: t.values[i])
        # w_SIG → SIG
        per_obj[label] = {k.removeprefix("w_"): float(v) for k, v in imp.items()}

    signals = sorted(per_obj["mean_balance"], key=per_obj["mean_balance"].get,
                     reverse=True)
    rows = []
    for sig in per_obj["mean_balance"]:   # 모든 시그널
        scores = {label: per_obj[label].get(sig, 0.0) for _i, label in OBJECTIVES}
        scores["avg"] = sum(scores.values()) / len(OBJECTIVES)
        rows.append({"signal": sig, **scores})
    rows.sort(key=lambda r: r["avg"], reverse=True)
    return {"study": name, "n_trials": len(study.trials), "rows": rows}


def _write_md(res: dict) -> None:
    lines = [f"# Season v2 — NSGA fANOVA 피처 임포턴스", ""]
    lines.append(f"- study `{res['study']}` · {res['n_trials']} trials · fANOVA(seed={FANOVA_SEED})")
    lines.append("- 학습 목적(합성 아카데미 3목적) 분산 설명력 기준. 값=상대 중요도(합≈1).")
    lines.append("")
    lines.append("| 순위 | 시그널 | 평균자산 | 최악자산 | 턴오버 | 3목적 평균 |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for i, r in enumerate(res["rows"], 1):
        lines.append(f"| {i} | {r['signal']} | {r['mean_balance']*100:.1f}% | "
                     f"{r['worst_balance']*100:.1f}% | {r['turnover']*100:.1f}% | "
                     f"{r['avg']*100:.1f}% |")
    lines.append("")
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    res = analyze()
    print(f"study={res['study']} · {res['n_trials']} trials · fANOVA\n")
    print(f"{'순위':>3} {'시그널':<10} {'평균자산':>8} {'최악자산':>8} {'턴오버':>8} {'평균':>8}")
    for i, r in enumerate(res["rows"], 1):
        print(f"{i:>3} {r['signal']:<10} {r['mean_balance']*100:>7.1f}% "
              f"{r['worst_balance']*100:>7.1f}% {r['turnover']*100:>7.1f}% "
              f"{r['avg']*100:>7.1f}%")
    _write_md(res)
    print(f"\nmd={MD_OUT}")


if __name__ == "__main__":
    main()
