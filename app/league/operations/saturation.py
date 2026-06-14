"""Saturation 정지 콜백 — Optuna study가 일정 기간 의미있는 개선 없으면 self stop.

[배경]
v1.x 라인업에서 단일목적 sampler 5시드 sweep을 돌릴 때, "포화(saturate)" 시점
(이후 개선이 노이즈 수준)을 정확히 잡아야 시드별 비교가 공정하다. cap만 박으면
빨리 saturate한 시드도 끝까지 도는 낭비 + 늦게 saturate한 시드는 cap에 잘림.

[원본]
2026-06-14 이전 `app/league/single_obj_compare_gp.py`에 같이 있던 단일 클래스 +
두 상수. 시즌 무관 재사용 유틸이라 operations/로 옮겨 둠.

[설계]
direction=maximize 가정. 첫 best 갱신 전까지는 카운트 시작 안 함.
patience trial 동안 best의 상대 개선 < min_delta_pct면 study.stop().
saturate_trial = 마지막 의미있는 개선 시점 — 리포트용 노출.
"""
import optuna


PATIENCE = 300                 # plateau 정지 — 개선 없는 트라이얼 한도
MIN_DELTA_PCT = 0.0005         # 0.05% (잔고 800만 기준 ~4천원 = 가중치 미세조정 노이즈)


class PlateauStopCallback:
    """patience 트라이얼 동안 best_value 상대 개선 < min_delta_pct면 study.stop()."""

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
