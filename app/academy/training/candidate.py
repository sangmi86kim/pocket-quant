"""후보 코덱 — Optuna 트라이얼 파라미터 ↔ 후보(가중치 + 시그널 파라미터).

[왜 따로 빼냐]
이전엔 `exam/grade.py`에 `decode_params`가 얹혀 있었지만, 정작 exam은 이걸 안 쓰고
training/ops/tools만 끌어다 썼다 — 채점(scoring)이 아니라 "최적화기의 출력을 읽는
번역기"라 training 쪽 관심사다. 채점층(exam)과 분리해 여기 둔다.

[구성]
- `decode_params` : trial.params → (가중치, 시그널 파라미터). suggest의 역함수.
- `suggest_weights` : 가중치 전용(v2 기본) 후보를 제시하고 곧장 디코드.
  TPE/CMA-ES/GP/NSGA-III가 공통으로 쓰던 `w_*` 제시 루프를 한 곳에 모은다.
"""
import optuna

from app.pocket.signals import SIGNAL_NAMES


def decode_params(params: dict) -> tuple[list[float], dict]:
    """Optuna trial.params → (가중치, 시그널 파라미터). suggest_weights의 역함수.

    가중치 전용 리그(v2) 트라이얼엔 w_* 만 있다 → 시그널 파라미터는 기본값({})."""
    weights = [params[f"w_{g}"] for g in SIGNAL_NAMES]
    if "VOL_CALM" not in params:
        return weights, {}
    sig = {k: params[k] for k in
           ("DD_LIMIT", "MA_WINDOW", "MOM_LOOKBACK", "RSI_OVERSOLD", "BB_K", "VOL_CALM")}
    sig["VOL_STRESSED"] = params["VOL_CALM"] + params["VOL_SPREAD"]
    return weights, sig


def suggest_weights(trial: optuna.Trial) -> tuple[list[float], dict]:
    """가중치 전용 후보 제시 — SIGNAL_NAMES 순서대로 w_*를 뽑고 곧장 디코드한다."""
    params = {f"w_{g}": trial.suggest_float(f"w_{g}", 0.0, 1.0)
              for g in SIGNAL_NAMES}
    return decode_params(params)
