"""2단계 보충학습 전이검증 PoC (lab) — 약점 진단 기반 적대적 보충.

[질문] RS 교과서만 공부한 학생 vs RS 1차 + '자기 약점 국면' 맞춤 보충 2차 한 학생 —
어느 쪽이 실전(OOS)에서 더 잘 버티나?

[핵심 순서 — 적대적의 본질]
  1. 1차 교육 : RS 교과서(실제 비율)로 GP 5명 키움.
  2. 약점 진단: 그 5명을 국면별 시험장(상승만/하락만/횡보만/변동만)에 넣어
                **집단이 제일 못 본 국면**을 측정(보충자료는 남이 아닌 *학생*이 정함).
  3. 맞춤 보충: 그 약점 국면을 집중 출제한 교과서(베이스 섞음=배운 것 까먹기 방지).
  4. 2차 보충: 1차 우등생을 웜스타트로 이어받아 보충 학습 → 5명.
  5. 채점    : 안 가르친 실제 QQQ(2020-07~, 봉인 전) score_vs_dca(성실이 대비).

  사전등록 합격: 2집단 중앙값 ≥ 1집단 중앙값 AND 2집단 최악 ≥ 1집단 최악.

[주의] OOS는 약-오염(과거 본 적 있음)이나 두 집단에 똑같이 묻어 상대비교는 강건.
약점은 집단(5명) 평균으로 진단(개별·trial 단위 아님 — 메모리 2단계 설계).

실행: .venv/Scripts/python.exe -m app.lab.rs_transfer_poc
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from app.academy.training.single_objective.engine import PlateauStopCallback, _objective
from app.academy.training.candidate import decode_params
from app.lab.rs_hybrid_textbook import STATES, make_world_rs
from app.pocket.battle import (
    SLIPPAGE_COST, TRADE_COST, _dca_position, _score_position,
    apply_no_trade_band, fight_dca, score_vs_dca,
)
from app.pocket.signals import combine_positions, positions_with_params
from app.pocket.models import Gym
from app.world.data_loader import LoadedGym, get_prices
from app.world.regime import REGIME_LABELS, classify_daily

N_SEEDS = 5
N_GYMS = 12
PHASE1_TRIALS = 300
PHASE2_TRIALS = 200
PATIENCE = 70                 # 평탄 70트라이얼이면 조기 종료(PoC 속도)
PHASE2_SKEW = 3.0            # 측정된 약점 국면을 3배 보충 출제(베이스 섞어 forgetting 방지)
DIAG_GYMS = 3               # 약점 진단용 국면별 시험장 수
DIAG_SKEW = 50.0            # 진단 시험장은 해당 국면에 거의 몰빵(near-pure)
WARMSTART_K = 5             # 1차 우등생 top-K를 2차 시작점으로 넣음
OOS_WARMUP_START = "2019-01-01"
OOS_EVAL_START = "2020-07-01"
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "rs_transfer_poc")


def _named_gyms(n: int, textbook_seed: int, prefix: str, skew=None):
    """make_world_rs로 N권 생성 + 고유 이름(dca 키 충돌 방지) + 성실이 페어."""
    gyms = []
    for i in range(n):
        lg = make_world_rs(seed=textbook_seed + i, skew=skew)
        gyms.append(LoadedGym(
            gym=Gym(f"{prefix}#{i + 1:02d}", difficulty=0, volatility=0,
                    ticker="SYNTH", start=lg.gym.start, end=lg.gym.end),
            prices=lg.prices))
    dca = {g.gym.name: fight_dca(g) for g in gyms}
    return gyms, dca


def _oos_gym() -> tuple[LoadedGym, dict]:
    """실제 QQQ 2020-07~(봉인 이후 loader가 자동 절단) OOS 시험장 + 성실이."""
    prices = get_prices("QQQ", OOS_WARMUP_START, "2026-12-31")
    end = prices.index[-1].strftime("%Y-%m-%d")
    gym = Gym("OOS#post2020", difficulty=0, volatility=0, ticker="QQQ",
              start=OOS_EVAL_START, end=end)
    lg = LoadedGym(gym=gym, prices=prices)
    return lg, {gym.name: fight_dca(lg)}


def _train(gyms, dca, trials: int, seed: int, warmstart=None) -> optuna.Study:
    """GP 단일목적 학습. warmstart=params 리스트면 2차 시작점으로 enqueue(웜스타트)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.GPSampler(seed=seed, deterministic_objective=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    for params in (warmstart or []):
        study.enqueue_trial(params)
    stop = PlateauStopCallback(PATIENCE, 0.0005)
    study.optimize(lambda t: _objective(t, gyms, dca), n_trials=trials, callbacks=[stop])
    return study


def _topk_params(study: optuna.Study, k: int) -> list[dict]:
    done = [t for t in study.trials if t.value is not None]
    done.sort(key=lambda t: t.value if t.value is not None else float("-inf"),
              reverse=True)
    return [dict(t.params) for t in done[:k]]


def _score_on(params: dict, lg: LoadedGym, dca) -> float:
    weights, sig_params = decode_params(params)
    positions = positions_with_params(lg.prices, sig_params)
    position = combine_positions(positions, weights)
    return score_vs_dca(_score_position(position, lg), dca[lg.gym.name])


def _regime_edges(params: dict, oos_lg: LoadedGym) -> dict:
    """OOS를 국면별로 쪼개 성실이 대비 일별 초과수익(연율) — 흩어진 국면 날도 일별이라 OK.

    score_vs_dca(연속 곡선 필요)를 못 쓰는 자리라, 같은 실행 모델(하루 lag·턴오버 과금)로
    전략/성실이 일별 수익을 재현한 뒤 그날의 국면 라벨로 묶어 평균낸다.
    """
    weights, sig_params = decode_params(params)
    positions = positions_with_params(oos_lg.prices, sig_params)
    position = combine_positions(positions, weights)
    prices = oos_lg.prices
    mret = prices.pct_change()
    eff = apply_no_trade_band(position).shift(1)
    strat_ret = eff * mret - eff.diff().abs() * (TRADE_COST + SLIPPAGE_COST)
    deff = _dca_position(oos_lg).shift(1)
    dca_ret = deff * mret - deff.diff().abs() * SLIPPAGE_COST
    excess = strat_ret - dca_ret
    labels = classify_daily(prices)
    start, end = pd.Timestamp(oos_lg.gym.start), pd.Timestamp(oos_lg.gym.end)
    out = {}
    for R in STATES:
        days = labels.index[labels == R]
        days = days[(days >= start) & (days <= end)]
        e = excess.reindex(days).dropna()
        out[R] = float(e.mean() * 252) if len(e) else float("nan")   # 연율 환산
    return out


def _oos_regime_days(oos_lg: LoadedGym) -> dict:
    labels = classify_daily(oos_lg.prices)
    start, end = pd.Timestamp(oos_lg.gym.start), pd.Timestamp(oos_lg.gym.end)
    sub = labels[(labels.index >= start) & (labels.index <= end)]
    return {R: int((sub == R).sum()) for R in STATES}


def _diagnose_weak(champs: list[dict], diag: dict) -> tuple[str, dict]:
    """집단(champs)을 국면별 시험장에서 채점 → 평균 점수 최저 국면 = 약점."""
    regime_score = {}
    for s in STATES:
        gyms, dca = diag[s]
        regime_score[s] = float(np.mean([_score_on(c, lg, dca)
                                         for c in champs for lg in gyms]))
    weak = min(regime_score, key=lambda s: regime_score[s])
    return weak, regime_score


def run(n_seeds=N_SEEDS, n_gyms=N_GYMS, p1=PHASE1_TRIALS, p2=PHASE2_TRIALS):
    print(f"1차 공통 교과서(RS) {n_gyms}권 + 진단 시험장 준비 ...", flush=True)
    rs_gyms, rs_dca = _named_gyms(n_gyms, 1000, "RS")
    diag = {s: _named_gyms(DIAG_GYMS, 7000 + i * 100, f"DIAG-{s}", skew={s: DIAG_SKEW})
            for i, s in enumerate(STATES)}
    oos_lg, oos_dca = _oos_gym()

    # 1차: 5명 키움
    print("① 1차 교육(RS) ...", flush=True)
    studies1, champ1 = [], []
    for seed in range(n_seeds):
        s1 = _train(rs_gyms, rs_dca, p1, seed)
        studies1.append(s1)
        champ1.append(s1.best_trial.params)
        print(f"   seed {seed} 1차 완료 (trials {len(s1.trials)})", flush=True)

    # 2차 약점 진단(집단)
    weak, regime_score = _diagnose_weak(champ1, diag)
    print("② 약점 진단 - 국면별 집단 평균 score_vs_dca:")
    for s in STATES:
        mark = "  ← 약점(보충 대상)" if s == weak else ""
        print(f"   {REGIME_LABELS[s]}: {regime_score[s]:+.3f}{mark}")

    # 맞춤 보충 교과서 + 2차 웜스타트 보충
    print(f"③ 맞춤 보충 교과서({REGIME_LABELS[weak]} ×{PHASE2_SKEW}) + 2차 보충 ...", flush=True)
    adv_gyms, adv_dca = _named_gyms(n_gyms, 5000, "BOCHUNG", skew={weak: PHASE2_SKEW})
    champ2 = []
    for seed in range(n_seeds):
        s2 = _train(adv_gyms, adv_dca, p2, seed,
                    warmstart=_topk_params(studies1[seed], WARMSTART_K))
        champ2.append(s2.best_trial.params)
        print(f"   seed {seed} 2차 완료 (trials {len(s2.trials)})", flush=True)

    # OOS 채점
    rs_scores = np.array([_score_on(c, oos_lg, oos_dca) for c in champ1])
    adv_scores = np.array([_score_on(c, oos_lg, oos_dca) for c in champ2])
    verdict = (np.median(adv_scores) >= np.median(rs_scores)
               and np.min(adv_scores) >= np.min(rs_scores))

    print("\nOOS score_vs_dca (성실이 대비, 양수=이김)")
    for seed in range(n_seeds):
        print(f"   seed {seed}: RS={rs_scores[seed]:+.3f}  RS→보충={adv_scores[seed]:+.3f}")
    print(f"  RS만   중앙 {np.median(rs_scores):+.3f}  최악 {np.min(rs_scores):+.3f}")
    print(f"  RS→보충 중앙 {np.median(adv_scores):+.3f}  최악 {np.min(adv_scores):+.3f}")
    print(f"  판정: {'합격(보충 효과 있음)' if verdict else '불합격(효과 없음/악화)'}")

    # ★ OOS를 국면별로 쪼개 두 집단 비교 (성실이 대비 연율 초과수익)
    days = _oos_regime_days(oos_lg)
    rs_edges = [_regime_edges(c, oos_lg) for c in champ1]
    adv_edges = [_regime_edges(c, oos_lg) for c in champ2]
    rs_reg = {R: float(np.nanmean([e[R] for e in rs_edges])) for R in STATES}
    adv_reg = {R: float(np.nanmean([e[R] for e in adv_edges])) for R in STATES}
    print("\n국면별 OOS 초과수익 (성실이 대비, 연율%):")
    print("  국면        일수   RS만    RS→보충")
    for R in STATES:
        print(f"  {REGIME_LABELS[R]:6s} {days[R]:5d}  "
              f"{rs_reg[R]*100:+7.1f}  {adv_reg[R]*100:+7.1f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    _fig(rs_scores, adv_scores, rs_reg, adv_reg, days)
    _write_md(rs_scores, adv_scores, regime_score, weak, verdict,
              rs_reg, adv_reg, days, n_seeds, n_gyms, p1, p2)
    print(f"[저장] {OUT_DIR}")


def _fig(rs_scores, adv_scores, rs_reg, adv_reg, days):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                  gridspec_kw={"width_ratios": [1.5, 1.0]})

    # ★ 국면별 초과수익 — RS만 vs 보충 (실제 OOS를 국면으로 쪼갬)
    xs = np.arange(len(STATES))
    w = 0.38
    ax.bar(xs - w / 2, [rs_reg[s] * 100 for s in STATES], w,
           label="RS만", color="#90a4ae")
    ax.bar(xs + w / 2, [adv_reg[s] * 100 for s in STATES], w,
           label="RS→보충", color="#ef6c00")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{REGIME_LABELS[s]}\n({days[s]}일)" for s in STATES])
    ax.set_ylabel("성실이 대비 초과수익 (연율 %)")
    ax.set_title("국면별 OOS 성적 — RS만 vs 보충 (실제 2020-07~)", fontsize=11)
    ax.legend(fontsize=9)

    data = [rs_scores, adv_scores]
    ax2.boxplot(data, widths=0.5, medianprops=dict(color="#c62828", lw=2))
    jit = np.random.default_rng(0).normal(0, 0.03, len(rs_scores))
    for i, arr in enumerate(data, start=1):
        ax2.scatter(np.full(len(arr), i) + jit, arr, color="#1565c0", alpha=0.8, zorder=3)
    ax2.axhline(0, color="gray", ls="--", lw=1, alpha=0.7)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["RS만", "RS→보충"])
    ax2.set_ylabel("전체 OOS score_vs_dca")
    ax2.set_title("전체 종합 — 챔피언 5명씩", fontsize=11)

    fig.suptitle("2단계 보충학습 — 국면별 실전 성적 비교", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "rs_transfer_poc.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def _write_md(rs_scores, adv_scores, regime_score, weak, verdict,
              rs_reg, adv_reg, days, n_seeds, n_gyms, p1, p2):
    L = [
        "# 2단계 보충학습 전이검증 PoC — RS만 vs RS→약점 보충\n",
        f"> seed {n_seeds} · 교과서 {n_gyms}권 · 1차 {p1}/2차 {p2} 트라이얼 · "
        f"보충 skew ×{PHASE2_SKEW} · OOS {OOS_EVAL_START}~(봉인 전).\n",
        "\n![전이검증](rs_transfer_poc.png)\n",
        "\n## ① 약점 진단 (1차 집단 5명의 국면별 평균 score_vs_dca)\n\n",
        "| 국면 | 점수 |\n|---|--:|\n",
    ]
    for s in STATES:
        mark = " **← 약점(보충 대상)**" if s == weak else ""
        L.append(f"| {REGIME_LABELS[s]} | {regime_score[s]:+.3f}{mark} |\n")
    L.append(
        f"\n→ 보충 교과서는 **{REGIME_LABELS[weak]}**를 ×{PHASE2_SKEW} 집중 출제(베이스 섞음).\n"
        "\n## ② OOS 성적 (score_vs_dca, 양수=성실이 이김)\n\n"
        "| 집단 | 중앙값 | 최악 | 5명 |\n|---|--:|--:|---|\n"
        f"| RS만 | {np.median(rs_scores):+.3f} | {np.min(rs_scores):+.3f} | "
        + ", ".join(f"{x:+.2f}" for x in rs_scores) + " |\n"
        f"| RS→약점 보충 | {np.median(adv_scores):+.3f} | {np.min(adv_scores):+.3f} | "
        + ", ".join(f"{x:+.2f}" for x in adv_scores) + " |\n"
        "\n## ③ 국면별 OOS 성적 (실제 OOS를 국면으로 쪼갬, 성실이 대비 연율%)\n\n"
        "| 국면 | OOS 일수 | RS만 | RS→보충 | 보충 효과 |\n|---|--:|--:|--:|:--:|\n"
    )
    for R in STATES:
        better = "↑" if adv_reg[R] > rs_reg[R] else "↓"
        L.append(f"| {REGIME_LABELS[R]} | {days[R]} | {rs_reg[R]*100:+.1f} | "
                 f"{adv_reg[R]*100:+.1f} | {better} |\n")
    L.append(
        f"\n## 판정 (사전등록)\n\n"
        f"2집단 중앙·최악 **모두** ≥ 1집단 → "
        f"**{'합격: 약점 보충이 OOS robust 개선' if verdict else '불합격: 보충 효과 없음/악화'}**\n"
        "단, 종합 점수는 상승장(68%)에 눌리므로 ③ 국면별 표에서 하락·횡보 효과를 함께 본다.\n"
        "\n재현: `.venv/Scripts/python.exe -m app.lab.rs_transfer_poc`\n"
    )
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("".join(L))


if __name__ == "__main__":
    run()
