"""Run classroom studies."""
import json
import secrets
import traceback
from datetime import datetime
from pathlib import Path

from app.academy.curriculum import prepare_academy_data
from app.academy.training.classroom import nsga3, gp, cma_es, tpe


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "app" / "academy" / "training" / "results"

SINGLE_TRIALS = {
    "TPE": 5000,
    "CMA-ES": 5000,
    "GP": 1500,
}
NSGA_TRIALS = 10000
NSGA_GYMS = 20
POPULATION = 30          # 3목적 reference-point 정합(≈28점) + trial당 수렴 빠름 (pop 30 vs 50 실측)
EARLY_STOP_WINDOW = 5    # HV MA(5)가 5세대 연속 변화 없으면 조기 종료


def roll_seed() -> int:
    return secrets.randbelow(1_000_000_000)


def prepare_school_data(n_gyms: int = NSGA_GYMS,
                        seed: int | None = None):
    """한 학기 합성장 + 성실이 기준선. (gyms, dca) 생성은 curriculum에 위임."""
    return prepare_academy_data(n_gyms=n_gyms, seed=seed)


def single_trials(study) -> list[dict]:
    out = []
    for trial in study.trials:
        if trial.value is None:
            continue
        out.append({
            "trial": trial.number,
            "value": trial.value,
            "params": dict(trial.params),
        })
    return out


def run_single_classroom(name: str, engine, loaded_gyms, dca,
                         trials: int | None = None) -> dict:
    seed = roll_seed()
    study, _, _ = engine.run_study(
        trials=trials or SINGLE_TRIALS[name],
        seed=seed,
        loaded_gyms=loaded_gyms,
        dca=dca,
    )
    return {
        "name": name,
        "seed": seed,
        "kind": "single",
        "trials": len(study.trials),
        "trial_target": trials or SINGLE_TRIALS[name],
        "early_stop": study.user_attrs.get("early_stop"),
        "items": single_trials(study),
    }


def run_nsga_classroom(stamp: str, loaded_gyms, dca,
                       academy_seed: int,
                       trials: int = NSGA_TRIALS) -> dict:
    seed = roll_seed()
    storage_path = RESULTS_DIR / f"classroom_nsga3_{stamp}.db"
    storage = f"sqlite:///{storage_path.as_posix()}"
    study_name = f"classroom_nsga3_{stamp}"
    study, gyms, dca, hv_cb, mut_cb = nsga3.run_study(
        n_trials=trials,
        seed=seed,
        academy_seed=academy_seed,
        loaded_gyms=loaded_gyms,
        dca=dca,
        n_gyms=NSGA_GYMS,
        population_size=POPULATION,
        storage=storage,
        study_name=study_name,
        early_stop_window=EARLY_STOP_WINDOW,
        adaptive_mutation=True,
    )
    summary = nsga3.summarize_front(study, gyms, dca)
    rows = []
    for row in summary["front"]:
        rows.append({
            "trial": row["number"],
            "values": row["values"],
            "academy": row["academy"],
            "graduated": row["graduated"],
            "params": dict(row["params"]),
        })
    return {
        "name": "NSGA-III",
        "kind": "multi",
        "seed": seed,
        "academy_seed": academy_seed,
        "trial_target": trials,
        "storage": storage,
        "study_name": study_name,
        "trials": len(study.trials),
        "front_size": summary["front_size"],
        "passed": len(summary["passed"]),
        "baselines": summary["baselines"],
        "hv_points": len(hv_cb.hv) if hv_cb else None,
        "hv_stopped": hv_cb.stopped if hv_cb else None,
        "mut_points": len(mut_cb.history) if mut_cb else None,
        "items": rows,
    }


def run_all() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    academy_seed = roll_seed()
    loaded_gyms, dca = prepare_school_data(seed=academy_seed)
    json_path = RESULTS_DIR / f"classroom_studies_{stamp}.json"
    results: list[dict] = []
    failed: list[str] = []

    def flush() -> None:
        # 한 반이 끝날 때마다 즉시 저장 — 뒤 반이 죽어도 앞 반 결과는 남는다.
        payload = {
            "stamp": stamp,
            "source": "app.academy.training.study",
            "academy_seed": academy_seed,
            "academy_gyms": len(loaded_gyms),
            "failed": failed,
            "classrooms": results,
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def study_one(name: str, run) -> None:
        # 한 반의 실패가 학기 전체를 죽이지 않게 격리한다 (예: GP는 torch 필요).
        print(f"Running {name} classroom...", flush=True)
        try:
            result = run()
        except Exception as exc:  # noqa: BLE001 — 반별 고립, 나머지 보존이 목적
            failed.append(name)
            print(f"  [FAIL] {name} classroom: {exc!r}", flush=True)
            traceback.print_exc()
            flush()
            return
        result["academy_seed"] = academy_seed
        results.append(result)
        flush()
        print(f"Done {name}: trials={result['trials']}", flush=True)

    for name, engine in (("TPE", tpe), ("CMA-ES", cma_es), ("GP", gp)):
        study_one(name, lambda e=engine, n=name:
                  run_single_classroom(n, e, loaded_gyms, dca))
    study_one("NSGA-III",
              lambda: run_nsga_classroom(stamp, loaded_gyms, dca, academy_seed))

    if failed:
        print(f"[WARN] 실패한 반: {failed} — 완료된 {len(results)}개는 저장됨",
              flush=True)
    return json_path


def main() -> None:
    json_path = run_all()
    print(f"study_results={json_path}", flush=True)


if __name__ == "__main__":
    main()
