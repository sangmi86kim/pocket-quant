"""2단계 보충학습 helper — 약점 진단 + 보충 교과서 구성."""
import numpy as np

from app.academy.curriculum.course import prepare_academy_data
from app.academy.curriculum.textbook import REGIME_STATES
from app.academy.training.candidate import decode_params
from app.pocket.battle import _score_position, fight_dca, score_vs_dca
from app.pocket.signals import combine_positions, positions_with_params
from app.world.data_loader import LoadedGym

BASE_RATIO = 0.70
ADVERSARIAL_RATIO = 0.30
DIAG_GYMS = 3
DIAG_SKEW = 50.0
PHASE2_SKEW = 3.0
WARMSTART_K = 5


def score_candidate(params: dict, lg: LoadedGym, dca: dict) -> float:
    """후보 params를 한 체육관에서 성실이 대비 score로 채점한다."""
    weights, sig_params = decode_params(params)
    positions = positions_with_params(lg.prices, sig_params)
    position = combine_positions(positions, weights)
    return score_vs_dca(_score_position(position, lg), dca[lg.gym.name])


def make_diagnostic_gyms(seed: int, n_per_regime: int = DIAG_GYMS,
                         skew: float = DIAG_SKEW) -> dict:
    """국면별 near-pure 진단장 묶음."""
    out = {}
    for i, regime in enumerate(REGIME_STATES):
        out[regime] = prepare_academy_data(
            n_gyms=n_per_regime,
            seed=seed + i * 100,
            textbook="rs",
            skew={regime: skew},
            name_prefix=f"DIAG-{regime}",
        )
    return out


def diagnose_weak_regime(topk: list[dict], diagnostic: dict) -> tuple[str, dict]:
    """교실 1차 topk 집단을 국면별 진단장에 태워 최저 점수 국면을 찾는다."""
    if not topk:
        raise ValueError("약점 진단할 topk가 비어 있음")
    regime_score = {}
    for regime in REGIME_STATES:
        gyms, dca = diagnostic[regime]
        scores = [
            score_candidate(item["params"], lg, dca)
            for item in topk
            for lg in gyms
        ]
        regime_score[regime] = float(np.mean(scores))
    weak = min(regime_score, key=lambda r: regime_score[r])
    return weak, regime_score


def make_phase2_gyms(base_gyms: list[LoadedGym], seed: int, weak_regime: str,
                     total_gyms: int, base_ratio: float = BASE_RATIO,
                     skew: float = PHASE2_SKEW) -> tuple[list[LoadedGym], dict, dict]:
    """1차 RS 교과서 일부 + 약점 적대 교과서 일부를 섞어 보충장을 만든다."""
    base_n = max(1, int(round(total_gyms * base_ratio)))
    base_n = min(base_n, len(base_gyms), total_gyms)
    adv_n = max(1, total_gyms - base_n)
    base_pick = base_gyms[:base_n]
    adv_gyms, _ = prepare_academy_data(
        n_gyms=adv_n,
        seed=seed,
        textbook="rs",
        skew={weak_regime: skew},
        name_prefix=f"BOCHUNG-{weak_regime}",
    )
    gyms = base_pick + adv_gyms
    dca = {lg.gym.name: fight_dca(lg) for lg in gyms}
    meta = {
        "base_ratio": base_ratio,
        "adversarial_ratio": 1.0 - base_ratio,
        "base_gyms": len(base_pick),
        "adversarial_gyms": len(adv_gyms),
        "skew": {weak_regime: skew},
    }
    return gyms, dca, meta


def warmstart_params(topk: list[dict], k: int = WARMSTART_K) -> list[dict]:
    """topk 후보에서 Optuna enqueue_trial용 params만 뽑는다."""
    return [dict(item["params"]) for item in topk[:k]]
