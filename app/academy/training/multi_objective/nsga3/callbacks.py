"""NSGA-III 튜닝 콜백 — HV 트렌드 트래커 + 그 추세를 받아 행동하는 리스너 둘.

[구조] HV 계산은 한 곳, 행동은 리스너가 한다.
  - HVTrendTracker   : 세대별 hypervolume을 계산·평활해 추세를 낸다 (단일 소스, optuna 콜백).
  - HVEarlyStopper   : 추세(이동평균)가 정체하면 study.stop().
  - AdaptiveMutation : 추세가 개선되면 변이율↓(미세조정), 정체면 변이율↑(탐색 강화).

옛 구조는 조기종료 콜백이 HV 계산까지 들고 있고 적응변이가 그 내부 리스트를 또
들여다봤다 — 계산/행동을 갈라, 트래커 하나가 세대마다 추세를 내보내고 리스너가
on_generation으로 받아 움직인다. 모두 sampler 진행 상태만 보는 일반 튜닝 기계장치라
학교 목적/졸업 판정(objectives/graduate)과 분리돼 있다.
"""
import numpy as np
from optuna._hypervolume import compute_hypervolume  # optuna 내장 HV (private — 인터페이스 변동 가능)

from app.academy.training.multi_objective.nsga3.objectives import DIRECTIONS


class HVTrendTracker:
    """세대별 hypervolume 추세를 계산·평활하는 단일 소스(트렌드 트래커).

    HV는 "front가 더 커지나"만 보는 단조 증가 추세 지수다 — 1.0 캡은 없다.
      - 첫 scale_warmup 세대: 목적값만 모은다(스케일 박스 확보), HV 미계산.
      - 이후: scale_warmup 세대 누적 min/max로 축 단위 통일(잔고 ~백만 vs turnover ~0.04)
        → optuna 내장 HV. 좋은 쪽은 무제한, ref보다 나쁜 쪽만 HV 유효성을 위해 컷한다.
      - 세대 경계(= population_size trial)에서만 측정한다 (현업 관행: 세대 단위).

    세대마다 raw HV를 append하고 listener.on_generation(self, study)로 알린다. 리스너
    (조기종료·적응변이)는 moving_avg(window)로 필요한 만큼만 평활해 같은 추세를 읽는다.
    """

    def __init__(self, population_size: int, scale_warmup: int = 6,
                 trend_window: int = 5, listeners=None):
        self.population_size = population_size
        self.scale_warmup = scale_warmup
        self.trend_window = trend_window
        self.listeners = list(listeners or [])
        self._sign = np.array([-1.0 if d == "maximize" else 1.0 for d in DIRECTIONS])
        self._ref = np.ones(len(DIRECTIONS))   # 스케일 nadir(누적 worst 코너) = HV 기준점
        self._lo: np.ndarray | None = None
        self._hi: np.ndarray | None = None
        self._n = 0
        self.hv: list[float] = []           # 세대별 raw HV (단조 추세 지수, 캡 없음)
        self.trend: list[list[float]] = []  # [trial_number, raw, ma] — DB 영속용
        self.stopped = False   # 조기종료 리스너가 멈췄으면 True (run 요약용)

    def moving_avg(self, window: int):
        """최근 window 세대 raw HV 평균(덜 차면 있는 만큼). 리스너 공용 입력."""
        if not self.hv:
            return None
        return float(np.mean(self.hv[-window:]))

    def __call__(self, study, trial) -> None:
        self._n += 1
        n = self._n
        if self._lo is None:
            # 워밍업: scale_warmup 세대(= population_size×scale_warmup trial) 누적 min/max로 스케일 고정
            if n < self.scale_warmup * self.population_size:
                return
            allv = np.array([t.values for t in study.trials
                             if t.values is not None]) * self._sign
            if len(allv) == 0:
                return
            self._lo, hi = allv.min(0), allv.max(0)
            self._hi = np.where(hi > self._lo, hi, self._lo + 1e-9)
            return
        if n % self.population_size:          # 세대 경계에서만 측정·기록
            return
        front = np.array([t.values for t in study.best_trials
                          if t.values is not None])
        if len(front) == 0:
            return
        scaled = (front * self._sign - self._lo) / (self._hi - self._lo)  # 좋은 쪽 무제한(no clip)
        scaled = np.minimum(scaled, self._ref)            # 나쁜 쪽만 ref로 컷(HV 유효성)
        hv = float(compute_hypervolume(scaled, self._ref))  # 1.0 캡 없음 — front 좋아질수록 증가
        self.hv.append(hv)
        # 기록 트렌드는 trend_window 평활값을 함께 남긴다(세대 노이즈를 눌러 수렴이 눈에 보이게).
        ma = self.moving_avg(self.trend_window)
        if ma is None:
            return
        self.trend.append([trial.number, round(hv, 6), round(ma, 6)])
        study.set_user_attr("hv_trend", self.trend)       # DB 영속: 세대별 [trial, raw, ma]
        for listener in self.listeners:
            listener.on_generation(self, study)


class HVEarlyStopper:
    """트래커 이동평균이 patience 세대 연속 정체하면 study.stop().

    트래커가 세대마다 on_generation으로 부른다 — HV는 트래커가 계산하고,
    여기선 '멈출 때인가'만 판단한다.
    """

    def __init__(self, window: int = 5, patience: int | None = None,
                 min_rel_improve: float = 1e-3):
        self.window = window
        self.patience = patience or window
        self.min_rel_improve = min_rel_improve
        self.best_ma = -1.0
        self.stale = 0
        self.stopped = False

    def on_generation(self, tracker, study) -> None:
        if len(tracker.hv) < self.window:
            return
        ma = tracker.moving_avg(self.window)
        if ma is None:
            return
        if ma > self.best_ma * (1 + self.min_rel_improve):   # 평활 HV가 의미있게 오르면 정체 카운터 리셋
            self.best_ma = ma
            self.stale = 0
            return
        self.stale += 1                                       # 변화 없는 세대 누적
        if self.stale >= self.patience:                       # patience 세대 연속 변화 없으면 stop
            print(f"  [early-stop] HV MA({self.window}) {self.patience}세대 연속 변화 없음 "
                  f"→ stop at gen {len(tracker.hv)} (HV_ma {ma:.4f})")
            self.stopped = True
            tracker.stopped = True
            study.stop()


class AdaptiveMutation:
    """트래커 추세가 개선되면 변이율↓(미세조정), 정체면 변이율↑(탐색 강화).

    트래커가 세대마다 on_generation으로 부른다. sampler 내부 경로가 바뀌면 스스로
    비활성화하고 history를 비워 둔다(엔진은 history 길이만 본다).
    """

    def __init__(self, sampler, n_params: int, window: int = 3,
                 up: float = 1.5, down: float = 0.85, hi: float = 0.5):
        self.sampler = sampler
        self.window = window
        self.up, self.down, self.hi = up, down, hi
        self.lo = 1.0 / n_params
        self.current = self.lo * 2
        self.best_ma = -1.0
        self.history: list[dict] = []
        self.enabled = self._set_prob(self.current)
        if not self.enabled:
            print("  [adaptive-mut] sampler path changed; disabled")

    def _set_prob(self, value: float) -> bool:
        try:
            self.sampler._child_generation_strategy._mutation_prob = value
            return True
        except AttributeError:
            return False

    def on_generation(self, tracker, study) -> None:
        if not self.enabled or len(tracker.hv) < self.window:
            return
        ma = tracker.moving_avg(self.window)
        if ma is None:
            return
        improved = ma > self.best_ma + 1e-9
        factor = self.down if improved else self.up      # 개선 중이면 좁히고, 정체면 넓힌다
        self.current = max(self.lo, min(self.hi, self.current * factor))
        if improved:
            self.best_ma = ma
        self._set_prob(self.current)
        self.history.append({"gen": len(tracker.hv), "hv": round(tracker.hv[-1], 4),
                             "ma": round(ma, 4), "improved": improved,
                             "mut_prob": round(self.current, 4)})
