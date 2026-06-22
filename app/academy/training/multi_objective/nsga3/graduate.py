"""학교 졸업 판정 — front 후보를 turnover 스펙으로 거르고 라벨 부여.

[책임]
  - 졸업 필터(summarize_front): front 후보 중 turnover cap 통과자 추림 + 대표 라벨

turnover는 목적함수가 아니라 운용 스펙이다. 합성장(가짜)에서 벤치마크(성실이 DCA·
어플삭제맨 buy&hold)를 이겼는지는 보지 않는다 — 진짜 벤치마크 승부는 졸업시험
(실데이터)에서 가린다. 최악 시장 방어(worst)도 졸업 게이트가 아니라 top-k 선발
점수(정규화 median+worst)로 옮겼다.

이건 sampler 진행이 아니라 "채점/선발" 성격의 일이다 — objectives(목적함수)·
callbacks(튜닝)와 분리해 두면 나중에 채점 규칙을 손볼 때 엔진을 안 건드린다.
"""
import numpy as np

from app.academy.training.multi_objective.nsga3.objectives import SEED_KRW, academy_metrics

TURNOVER_CAP_SWEEP = [0.02, 0.03, 0.04, 0.05, 0.075, 0.10]


def _percentile_rank(values: list[float]) -> list[float]:
    """각 값의 백분위 순위(0=최저~1=최고). 잔고 스케일·이상치에 강건한 정규화.

    raw 가중합은 잔고(원 단위) 때문에 worst가 묻힌다 — 순위로 바꿔 같은 0~1 축에 둔다."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n <= 1:
        return [1.0] * n
    order = arr.argsort()
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    return list(ranks / (n - 1))


def select_topk(passed: list[dict], k: int = 30,
                w_median: float = 0.65, w_worst: float = 0.35) -> list[dict]:
    """졸업자를 '정규화 median + 정규화 worst' 점수로 줄세워 상위 k명.

    파레토 프론트는 본래 순위가 없다(서로 비지배) — 최종 선발엔 점수화가 필요하다.
    median(전형적 성과) 단독이면 졸업선 바로 위 고위험 후보가 올라오니, worst(생존력)를
    섞는다. 잔고 스케일이 커서 졸업자 내부 백분위로 정규화한 뒤 가중합한다.
    가중치(median 0.65 / worst 0.35)는 GPT 고문 협의 기본값 — 운영하며 조정 대상.
    """
    if not passed:
        return []
    med_rank = _percentile_rank([r["academy"]["median_balance"] for r in passed])
    wor_rank = _percentile_rank([r["academy"]["worst_balance"] for r in passed])
    scored = [{**r, "select_score": w_median * mr + w_worst * wr}
              for r, mr, wr in zip(passed, med_rank, wor_rank)]
    scored.sort(key=lambda r: r["select_score"], reverse=True)
    return scored[:k]


def _trial_academy_metrics(trial, seed_krw: int) -> dict:
    """새 study는 user_attrs에 turnover까지 저장한다. 옛 study는 values만 복구한다."""
    metrics = trial.user_attrs.get("academy_metrics")
    if isinstance(metrics, dict):
        return dict(metrics)
    values = list(trial.values)
    metrics = academy_metrics(values, seed_krw)
    if len(values) >= 3:
        metrics["turnover"] = values[2]
    return metrics


def summarize_front(study, turnover_cap: float = 0.05,
                    seed_krw: int = SEED_KRW) -> dict:
    """학교 front 졸업 후보 요약.

    졸업 필터 = turnover cap 하나. 벤치마크(성실이/어플삭제맨) 상대비교는 제거됐다
    — 합성장은 가짜라, 진짜 비교는 졸업시험(실데이터)에서 한다. worst 방어는 졸업
    게이트가 아니라 top-k 선발 점수로 옮겼다.

    라벨(Rich/Sturdy/Low-turnover)은 졸업자 중 대표를 뽑는 표시용이다.
    """
    front = []
    for t in study.best_trials:
        row = {"number": t.number, "values": list(t.values),
               "params": dict(t.params)}
        row["academy"] = _trial_academy_metrics(t, seed_krw)
        row["graduated"] = row["academy"]["turnover"] <= turnover_cap
        front.append(row)

    passed = [r for r in front if r["graduated"]]

    labels = {}
    if passed:
        labels["Rich"] = max(passed, key=lambda r: r["academy"]["median_balance"])
        labels["Sturdy"] = max(passed, key=lambda r: r["academy"]["worst_balance"])
        labels["Low-turnover"] = min(passed, key=lambda r: r["academy"]["turnover"])

    cap_sweep = {
        f"{cap:.3f}": sum(r["academy"]["turnover"] <= cap for r in front)
        for cap in TURNOVER_CAP_SWEEP
    }

    return {
        "front_size": len(front),
        "front": front,
        "passed": passed,
        "selected": select_topk(passed),  # top30 — 정규화 median+worst 점수
        "labels": labels,
        "turnover_cap": turnover_cap,
        "turnover_cap_sweep": cap_sweep,
    }
