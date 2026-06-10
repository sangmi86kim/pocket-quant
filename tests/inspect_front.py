"""
inspect_front.py - 저장된 NSGA-III 스터디(sqlite)의 Pareto front 검사 도구

본 스터디는 sqlite에 저장되므로(중단/재개 가능), 실행이 끝난 뒤 언제든
이 스크립트로 다시 열어 분석할 수 있다:
  - tolerance 스윕: 하드 필터를 얼마나 풀어야 후보가 몇 개 나오나
  - 라벨 후보(Defensive/Balanced/Aggressive/Low-turnover) 상세
  - 전체 front를 reports/nsga3_front.csv 로 내보내기 (엑셀 검사용)

실행: 프로젝트 루트에서  python tests/inspect_front.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optuna

from app.backend.engine import nsga3
from app.service import _format_candidate_params, _format_objective_vector

# 경로는 전부 프로젝트 루트 기준 절대 경로 — 어디서 실행해도(IDE 포함) 같은 DB를 연다.
# (상대 경로면 PyCharm처럼 작업 디렉토리가 다른 실행에서 빈 DB를 새로 만들어 버린다.)
_ROOT = Path(__file__).resolve().parent.parent
STORAGE = f"sqlite:///{(_ROOT / 'optuna_pocketquant.db').as_posix()}"
STUDY = "nsga3_v2_weights"      # v1(가중치+파라미터, 관문① 전멸)도 DB에 남아 있음
CSV_OUT = _ROOT / "reports" / "nsga3_front.csv"


def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=STUDY, storage=STORAGE)
    done = len(study.trials)
    front = study.best_trials
    print(f"=== 스터디 {STUDY}: 트라이얼 {done}개 · Pareto front {len(front)}개 ===")

    # 기준점 (현 단일목적 챔피언)
    from app.backend.market.data import load_gyms
    from app.backend.market.gym import all_gyms
    from app.backend.engine.battle import fight_dca
    loaded = load_gyms(all_gyms())
    dca = {lg.gym.name: fight_dca(lg) for lg in loaded}
    ref = nsga3.reference_vector(loaded, dca)
    print("\n[기준점] 현 챔피언 VOL+REV_RSI+REV_BB (동일가중·기본 파라미터)")
    print("  " + _format_objective_vector(
        [min(ref["dotcom"], ref["gfc"]), ref["rebound"], ref["crash_v"],
         ref["bull"], ref["chop"], ref["turnover"]]))

    # tolerance 스윕 — 필터를 얼마나 풀면 후보가 몇 개인가
    print("\n=== 하드 필터 스윕 (턴오버 ≤ 0.10 고정) ===")
    for tol in (0.0, 0.02, 0.05, 0.10):
        s = nsga3.summarize_front(study, tolerance=tol)
        print(f"  전 국면 ≥ {-tol * 100:+.0f} : {len(s['passed'])}개 통과")

    # 기본 필터의 라벨 후보 상세
    summary = nsga3.summarize_front(study)
    print(f"\n=== 라벨 후보 (전 국면 ≥ -5, 턴오버 ≤ 0.10) ===")
    for label, row in summary["labels"].items():
        print(f"\n[{label}]  trial #{row['number']}  (5국면 평균 {row['mean5'] * 100:+.1f})")
        print("  " + _format_objective_vector(row["values"]))
        print("  " + _format_candidate_params(row["params"]))

    # CSV 내보내기 (front 전체)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(CSV_OUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["trial"] + nsga3.OBJECTIVE_NAMES + ["mean5", "min5"]
                        + sorted(front[0].params.keys()))
        for t in front:
            mean5 = sum(t.values[:5]) / 5
            min5 = min(t.values[:5])
            writer.writerow([t.number] + [f"{v:.6f}" for v in t.values]
                            + [f"{mean5:.6f}", f"{min5:.6f}"]
                            + [t.params[k] for k in sorted(t.params.keys())])
    print(f"\nfront 전체 {len(front)}개 → {CSV_OUT} 저장")


if __name__ == "__main__":
    main()
