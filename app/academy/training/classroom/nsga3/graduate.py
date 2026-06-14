"""학교 졸업 판정 — front 후보를 기준선과 견줘 졸업 필터·라벨 부여.

[책임]
  - 기준선 계산(_baseline_summary): 성실이(DCA)·어플삭제맨(buy&hold) 평균/최악
  - 졸업 필터(summarize_front): front 후보 중 4개 조건 통과자 추림 + 라벨

이건 sampler 진행이 아니라 "채점/선발" 성격의 일이다 — objectives(목적함수)·
callbacks(튜닝)와 분리해 두면 나중에 채점 규칙을 손볼 때 엔진을 안 건드린다.
"""
from app.academy.training.classroom.nsga3.objectives import (
    SEED_KRW,
    academy_metrics,
    prepare_data,
)
from app.pocket.battle import terminal_balance
from app.world.data_loader import LoadedGym


def _buy_hold_balance(loaded: LoadedGym, seed_krw: int = SEED_KRW) -> int:
    prices = loaded.prices.loc[loaded.gym.start:loaded.gym.end]
    rets = prices.pct_change().dropna()
    if len(rets) == 0:
        return seed_krw
    return int(seed_krw * float((1 + rets).cumprod().iloc[-1]))


def _baseline_summary(loaded_gyms: list[LoadedGym], dca: dict,
                      seed_krw: int = SEED_KRW) -> dict:
    dca_balances = [terminal_balance(dca[lg.gym.name], seed_krw)
                    for lg in loaded_gyms]
    bh_balances = [_buy_hold_balance(lg, seed_krw) for lg in loaded_gyms]
    return {
        "dca_mean": sum(dca_balances) / len(dca_balances),
        "dca_worst": min(dca_balances),
        "bh_mean": sum(bh_balances) / len(bh_balances),
        "bh_worst": min(bh_balances),
    }


def summarize_front(study, loaded_gyms: list[LoadedGym] | None = None,
                    dca: dict | None = None,
                    turnover_cap: float = 0.10,
                    bh_mean_floor: float = 0.90,
                    seed_krw: int = SEED_KRW) -> dict:
    """학교 front 졸업 후보 요약.

    졸업 필터 (전부 기준선 상대 — "성실이/어플삭제맨을 이겼나"):
      ① 평균 잔고가 성실이 평균보다 큼
      ② 최악 잔고가 성실이 최악보다 큼 (최악 평행세계에서도 성실이보다 덜 잃음)
      ③ 평균 잔고가 어플삭제맨 평균의 bh_mean_floor 이상
      ④ turnover cap 이하

    ②는 옛 "최악 > 시드(절대 흑자)"에서 바뀜 — 성실이도 -21% 잃는 학살 평행세계에서
    흑자를 요구하던 불가능 게이트라 front 전원 탈락. 기준을 성실이 최악으로 맞춰 ①③과 일관.
    """
    if loaded_gyms is None or dca is None:
        loaded_gyms, dca = prepare_data(
            seed=study.user_attrs.get("academy_seed"))
    baselines = _baseline_summary(loaded_gyms, dca, seed_krw)
    front = []
    for t in study.best_trials:
        row = {"number": t.number, "values": list(t.values),
               "params": dict(t.params)}
        row["academy"] = academy_metrics(row["values"], seed_krw)
        row["graduated"] = (
            row["academy"]["mean_balance"] > baselines["dca_mean"]
            and row["academy"]["worst_balance"] > baselines["dca_worst"]
            and row["academy"]["mean_balance"] >= baselines["bh_mean"] * bh_mean_floor
            and row["academy"]["turnover"] <= turnover_cap
        )
        front.append(row)

    passed = [r for r in front if r["graduated"]]

    labels = {}
    if passed:
        labels["Rich"] = max(passed, key=lambda r: r["academy"]["mean_balance"])
        labels["Sturdy"] = max(passed, key=lambda r: r["academy"]["worst_balance"])
        labels["Low-turnover"] = min(passed, key=lambda r: r["academy"]["turnover"])

    return {
        "front_size": len(front),
        "front": front,
        "passed": passed,
        "labels": labels,
        "baselines": baselines,
        "turnover_cap": turnover_cap,
        "bh_mean_floor": bh_mean_floor,
    }
