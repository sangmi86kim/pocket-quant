"""Run classroom studies."""
import json
import secrets
import traceback
from datetime import datetime
from pathlib import Path

from app.academy.curriculum import prepare_academy_data
from app.academy.training import remedial
from app.academy.training.multi_objective import nsga3
from app.academy.training.single_objective import cma_es, gp, tpe


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "app" / "academy" / "training" / "results"

SINGLE_TRIALS = {
    "TPE": 5000,
    "CMA-ES": 5000,
    "GP": 1500,
}
GP_SEED_LEAGUE = 5       # GP는 단일 study top-k가 한 전략으로 도배되므로 독립 seed별 대표 1명씩 뽑는다
SINGLE_TOPK = 30         # TPE/CMA-ES 단일목적은 점수(median) 순 top-k 선발 (GP는 seedleague 예외)
NSGA_TRIALS = 10000
NSGA_GYMS = 20
POPULATION = 30          # 3목적 reference-point 정합(≈28점) + trial당 수렴 빠름 (pop 30 vs 50 실측)
EARLY_STOP_WINDOW = 5    # HV MA(5)가 5세대 연속 변화 없으면 조기 종료
TEXTBOOK = "rs"


def roll_seed() -> int:
    return secrets.randbelow(1_000_000_000)


def prepare_school_data(n_gyms: int = NSGA_GYMS,
                        seed: int = 42):
    """한 학기 합성장 + 성실이 기준선. (gyms, dca) 생성은 curriculum에 위임."""
    return prepare_academy_data(n_gyms=n_gyms, seed=seed, textbook=TEXTBOOK,
                                name_prefix="RS")


def single_trials(study, k: int = SINGLE_TOPK) -> list[dict]:
    """완료 trial을 점수(value=median 잔고) 내림차순 상위 k명.

    단일목적은 목적값 자체가 순위라, NSGA의 복합점수(median+worst)와 달리 그냥
    점수순 top-k가 곧 선발이다(심플). GP는 seedleague(run_gp_seedleague)라 예외."""
    done = [{"trial": t.number, "value": t.value, "params": dict(t.params)}
            for t in study.trials if t.value is not None]
    done.sort(key=lambda r: r["value"], reverse=True)
    return done[:k]


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
        "topk": single_trials(study),
    }


def run_single_classroom_2phase(name: str, engine, phase1_gyms, phase1_dca,
                                diagnostic: dict, trials: int | None = None) -> dict:
    """단일목적 교실 2단계 학습 — RS 1차 → 약점 보충 2차."""
    target = trials or SINGLE_TRIALS[name]
    seed1 = roll_seed()
    study1, _, _ = engine.run_study(
        trials=target,
        seed=seed1,
        loaded_gyms=phase1_gyms,
        dca=phase1_dca,
    )
    phase1_topk = single_trials(study1)
    weak, regime_score = remedial.diagnose_weak_regime(phase1_topk, diagnostic)
    phase2_academy_seed = roll_seed()
    phase2_gyms, phase2_dca, phase2_meta = remedial.make_phase2_gyms(
        phase1_gyms, phase2_academy_seed, weak, len(phase1_gyms))
    seed2 = roll_seed()
    study2, _, _ = engine.run_study(
        trials=target,
        seed=seed2,
        loaded_gyms=phase2_gyms,
        dca=phase2_dca,
        warmstart=remedial.warmstart_params(phase1_topk),
    )
    return {
        "name": name,
        "kind": "single",
        "trials": len(study1.trials) + len(study2.trials),
        "trial_target": target,
        "weak_regime": weak,
        "regime_score": regime_score,
        "phase1": {
            "seed": seed1,
            "trials": len(study1.trials),
            "early_stop": study1.user_attrs.get("early_stop"),
            "topk": phase1_topk,
        },
        "phase2": {
            "seed": seed2,
            "academy_seed": phase2_academy_seed,
            "trials": len(study2.trials),
            "early_stop": study2.user_attrs.get("early_stop"),
            **phase2_meta,
        },
        "topk": single_trials(study2),
    }


def run_gp_seedleague(loaded_gyms, dca, trials: int | None = None,
                      n_seeds: int = GP_SEED_LEAGUE) -> dict:
    """GP를 랜덤 seed n개의 독립 study로 돌려 seed별 best 1명씩 선발한다.

    [왜] GP는 한 study 안에서 좋은 점을 찾으면 그 주변을 exploit해 top-k가 사실상
    같은 전략으로 도배된다. 성적순 top-k 대신 독립 seed별 대표 1명으로 다양성을 확보한다.
    같은 합성장(loaded_gyms·dca)을 공유하고 GP 탐색 seed만 다르게 가, 같은 문제를
    여러 출발점에서 독립적으로 푼다.
    """
    target = trials or SINGLE_TRIALS["GP"]
    seeds = [roll_seed() for _ in range(n_seeds)]
    topk = []
    total_trials = 0
    for seed in seeds:
        study, _, _ = gp.run_study(
            trials=target, seed=seed,
            loaded_gyms=loaded_gyms, dca=dca, early_stop=True,
        )
        best = study.best_trial
        total_trials += len(study.trials)
        topk.append({
            "seed": seed,
            "trial": best.number,
            "value": best.value,
            "params": dict(best.params),
            "study_trials": len(study.trials),
            "early_stop": study.user_attrs.get("early_stop"),
        })
    return {
        "name": "GP",
        "kind": "single",
        "seedleague": True,
        "seeds": seeds,
        "trial_target": target,
        "trials": total_trials,
        "topk": topk,
    }


def run_gp_seedleague_2phase(phase1_gyms, phase1_dca, diagnostic: dict,
                             trials: int | None = None,
                             n_seeds: int = GP_SEED_LEAGUE) -> dict:
    """GP seedleague 2단계 학습 — seed별 대표를 유지한다."""
    target = trials or SINGLE_TRIALS["GP"]
    seeds1 = [roll_seed() for _ in range(n_seeds)]
    studies1 = []
    phase1_topk = []
    total_trials = 0
    for seed in seeds1:
        study, _, _ = gp.run_study(
            trials=target, seed=seed,
            loaded_gyms=phase1_gyms, dca=phase1_dca, early_stop=True,
        )
        studies1.append(study)
        total_trials += len(study.trials)
        best = study.best_trial
        phase1_topk.append({
            "seed": seed,
            "trial": best.number,
            "value": best.value,
            "params": dict(best.params),
            "study_trials": len(study.trials),
            "early_stop": study.user_attrs.get("early_stop"),
        })

    weak, regime_score = remedial.diagnose_weak_regime(phase1_topk, diagnostic)
    phase2_academy_seed = roll_seed()
    phase2_gyms, phase2_dca, phase2_meta = remedial.make_phase2_gyms(
        phase1_gyms, phase2_academy_seed, weak, len(phase1_gyms))
    seeds2 = [roll_seed() for _ in range(n_seeds)]
    final_topk = []
    for seed, study1 in zip(seeds2, studies1):
        study2, _, _ = gp.run_study(
            trials=target, seed=seed,
            loaded_gyms=phase2_gyms, dca=phase2_dca, early_stop=True,
            warmstart=remedial.warmstart_params(
                [{"params": p} for p in _topk_params(study1, remedial.WARMSTART_K)],
                remedial.WARMSTART_K),
        )
        total_trials += len(study2.trials)
        best = study2.best_trial
        final_topk.append({
            "seed": seed,
            "trial": best.number,
            "value": best.value,
            "params": dict(best.params),
            "study_trials": len(study2.trials),
            "early_stop": study2.user_attrs.get("early_stop"),
        })

    return {
        "name": "GP",
        "kind": "single",
        "seedleague": True,
        "trial_target": target,
        "trials": total_trials,
        "weak_regime": weak,
        "regime_score": regime_score,
        "phase1": {
            "seeds": seeds1,
            "trials": sum(t["study_trials"] for t in phase1_topk),
            "topk": phase1_topk,
        },
        "phase2": {
            "seeds": seeds2,
            "trials": sum(t["study_trials"] for t in final_topk),
            "academy_seed": phase2_academy_seed,
            **phase2_meta,
        },
        "topk": final_topk,
    }


def _topk_params(study, k: int) -> list[dict]:
    done = [t for t in study.trials if t.value is not None]
    done.sort(key=lambda t: t.value, reverse=True)
    return [dict(t.params) for t in done[:k]]


def nsga_topk(summary: dict) -> list[dict]:
    """summarize_front 결과 → 학기 저장용 topk(top30 selected, select_score 포함).

    payload 조립의 핵심 소비부를 함수로 빼 둔다 — smoke가 이걸 직접 태워, summary
    키를 소비하다 깨지는 류(제거된 키 참조·selected/select_score 누락)를 1차 그물에서
    잡게 하기 위함. run_nsga_classroom과 smoke가 같은 조립 코드를 공유한다.
    """
    return [{
        "trial": row["number"],
        "values": row["values"],
        "academy": row["academy"],
        "graduated": row["graduated"],
        "select_score": row["select_score"],
        "params": dict(row["params"]),
    } for row in summary["selected"]]


def run_nsga_classroom(stamp: str, loaded_gyms, dca,
                       academy_seed: int,
                       trials: int = NSGA_TRIALS) -> dict:
    seed = roll_seed()
    storage_path = RESULTS_DIR / f"classroom_nsga3_{stamp}.db"
    storage = f"sqlite:///{storage_path.as_posix()}"
    study_name = f"classroom_nsga3_{stamp}"
    study, _gyms, _dca, hv_cb, mut_cb = nsga3.run_study(
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
    summary = nsga3.summarize_front(study)
    rows = nsga_topk(summary)
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
        "hv_points": len(hv_cb.hv) if hv_cb else None,
        "hv_stopped": hv_cb.stopped if hv_cb else None,
        "mut_points": len(mut_cb.history) if mut_cb else None,
        "topk": rows,
    }


def run_nsga_classroom_2phase(stamp: str, phase1_gyms, phase1_dca,
                              diagnostic: dict, academy_seed: int,
                              trials: int = NSGA_TRIALS,
                              use_storage: bool = True) -> dict:
    """NSGA-III 2단계 학습 — 1차 selected로 진단·웜스타트한다."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed1 = roll_seed()
    storage_path1 = RESULTS_DIR / f"classroom_nsga3_{stamp}_phase1.db"
    storage1 = f"sqlite:///{storage_path1.as_posix()}" if use_storage else None
    study_name1 = f"classroom_nsga3_{stamp}_phase1"
    study1, _gyms, _dca, hv1, mut1 = nsga3.run_study(
        n_trials=trials,
        seed=seed1,
        academy_seed=academy_seed,
        loaded_gyms=phase1_gyms,
        dca=phase1_dca,
        n_gyms=NSGA_GYMS,
        population_size=POPULATION,
        storage=storage1,
        study_name=study_name1,
        early_stop_window=EARLY_STOP_WINDOW,
        adaptive_mutation=True,
    )
    summary1 = nsga3.summarize_front(study1)
    phase1_topk = nsga_topk(summary1)
    weak, regime_score = remedial.diagnose_weak_regime(phase1_topk, diagnostic)

    phase2_academy_seed = roll_seed()
    phase2_gyms, phase2_dca, phase2_meta = remedial.make_phase2_gyms(
        phase1_gyms, phase2_academy_seed, weak, len(phase1_gyms))
    seed2 = roll_seed()
    storage_path2 = RESULTS_DIR / f"classroom_nsga3_{stamp}_phase2.db"
    storage2 = f"sqlite:///{storage_path2.as_posix()}" if use_storage else None
    study_name2 = f"classroom_nsga3_{stamp}_phase2"
    study2, _gyms, _dca, hv2, mut2 = nsga3.run_study(
        n_trials=trials,
        seed=seed2,
        academy_seed=phase2_academy_seed,
        loaded_gyms=phase2_gyms,
        dca=phase2_dca,
        n_gyms=NSGA_GYMS,
        population_size=POPULATION,
        storage=storage2,
        study_name=study_name2,
        early_stop_window=EARLY_STOP_WINDOW,
        adaptive_mutation=True,
        warmstart=remedial.warmstart_params(phase1_topk),
    )
    summary2 = nsga3.summarize_front(study2)
    return {
        "name": "NSGA-III",
        "kind": "multi",
        "trials": len(study1.trials) + len(study2.trials),
        "trial_target": trials,
        "weak_regime": weak,
        "regime_score": regime_score,
        "phase1": {
            "seed": seed1,
            "academy_seed": academy_seed,
            "storage": storage1,
            "study_name": study_name1,
            "trials": len(study1.trials),
            "front_size": summary1["front_size"],
            "passed": len(summary1["passed"]),
            "hv_points": len(hv1.hv) if hv1 else None,
            "hv_stopped": hv1.stopped if hv1 else None,
            "mut_points": len(mut1.history) if mut1 else None,
            "topk": phase1_topk,
        },
        "phase2": {
            "seed": seed2,
            "academy_seed": phase2_academy_seed,
            "storage": storage2,
            "study_name": study_name2,
            "trials": len(study2.trials),
            "front_size": summary2["front_size"],
            "passed": len(summary2["passed"]),
            "hv_points": len(hv2.hv) if hv2 else None,
            "hv_stopped": hv2.stopped if hv2 else None,
            "mut_points": len(mut2.history) if mut2 else None,
            **phase2_meta,
        },
        "topk": nsga_topk(summary2),
    }


def run_all() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    academy_seed = roll_seed()
    loaded_gyms, dca = prepare_school_data(seed=academy_seed)
    diagnostic = remedial.make_diagnostic_gyms(seed=academy_seed + 50_000)
    json_path = RESULTS_DIR / f"classroom_studies_{stamp}.json"
    # 소비자(graduate·리그·signal analysis)는 classroom_top30_*_v2.json을 글롭한다.
    # 2단계 산출물의 topk가 곧 선발이라, 같은 payload를 그 이름으로도 남겨 생산자↔소비자를 잇는다.
    top30_path = RESULTS_DIR / f"classroom_top30_{stamp}_v2.json"
    results: list[dict] = []
    failed: list[str] = []

    def flush() -> None:
        # 한 반이 끝날 때마다 즉시 저장 — 뒤 반이 죽어도 앞 반 결과는 남는다.
        payload = {
            "stamp": stamp,
            "source": "app.academy.training.study",
            "academy_seed": academy_seed,
            "academy_textbook": "rs_2phase",
            "academy_gyms": len(loaded_gyms),
            "failed": failed,
            "classrooms": results,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        json_path.write_text(text, encoding="utf-8")
        top30_path.write_text(text, encoding="utf-8")

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

    for name, engine in (("TPE", tpe), ("CMA-ES", cma_es)):
        study_one(name, lambda e=engine, n=name:
                  run_single_classroom_2phase(n, e, loaded_gyms, dca, diagnostic))
    study_one("GP", lambda: run_gp_seedleague_2phase(loaded_gyms, dca, diagnostic))
    study_one("NSGA-III",
              lambda: run_nsga_classroom_2phase(
                  stamp, loaded_gyms, dca, diagnostic, academy_seed))

    if failed:
        print(f"[WARN] 실패한 반: {failed} — 완료된 {len(results)}개는 저장됨",
              flush=True)
    print(f"top30_results={top30_path}", flush=True)
    return json_path


def main() -> None:
    json_path = run_all()
    print(f"study_results={json_path}", flush=True)


if __name__ == "__main__":
    main()
