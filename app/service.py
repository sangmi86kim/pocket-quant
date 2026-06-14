"""
service.py - NSGA-III 실행 흐름 조립

호출 측이 옵션을 인자로 직접 넘기는 라이브러리식 진입점이다:
  - tools/e2e.py (NSGA-III smoke 스모크 단계)
  - 그 외 어댑터/REPL에서 `run_nsga3(...)` 직접 호출

[2026-06-13 정리] main.py + config.json은 제거됐다 — NSGA-III가 시즌의 본선이
아니라 백엔드 한 갈래로 남았고, 시즌별 탐색 진입점은 `app/league/<name>.py`
어댑터 직접 실행 패턴(예: `python app/league/single_obj_search.py`)으로 통일됐다.
"""
import json
import random
import sys
from pathlib import Path

# Windows 콘솔 기본 cp949에선 이모지(⚠️/👑/🎫 등)가 인코딩 에러로 크래시.
# 모든 진입점에서 stdout/stderr를 utf-8로 재설정 (3.7+ 지원, 실패해도 무시).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")    # type: ignore[union-attr]
    except Exception:
        pass

from app.backend.genes.signals import ALL_GENES


def _apply_seed(seed: int | None) -> None:
    """시드 고정 시 NSGA-III 초기 인구·교배·돌연변이가 매번 같게 재현된다."""
    if seed is not None:
        random.seed(seed)


def _format_objective_vector(values: list[float]) -> str:
    """목적 벡터를 한 줄로 (점수 ×100 표기, 턴오버는 원값)."""
    names = ["bear", "rebound", "crash_v", "bull", "chop"]
    scores = " ".join(f"{n} {v * 100:+6.1f}" for n, v in zip(names, values[:5]))
    return f"{scores}  | turnover {values[5]:.4f}/일"


def _format_candidate_params(params: dict) -> str:
    """트레이더의 X를 사람 읽는 형태로 — 가중치는 비율(%)로 정규화."""
    w = [params[f"w_{g}"] for g in ALL_GENES]
    total = sum(w) or 1.0
    weights = " ".join(f"{g} {x / total * 100:.0f}%" for g, x in zip(ALL_GENES, w))
    if "DD_LIMIT" not in params:                 # 가중치 전용 리그 (v2, A안)
        return f"가중치: {weights}\n  파라미터: 기본값 고정"
    tunables = (f"DD {params['DD_LIMIT']:.2f} · MA {params['MA_WINDOW']} · "
                f"MOM {params['MOM_LOOKBACK']} · RSI<{params['RSI_OVERSOLD']} · "
                f"BB k{params['BB_K']:.2f} · VOL {params['VOL_CALM']:.3f}"
                f"~{params['VOL_CALM'] + params['VOL_SPREAD']:.3f}")
    return f"가중치: {weights}\n  파라미터: {tunables}"


# ── Regime Scanner 입력원: 각 관문(훈련장·시험장·평행세계·사천왕)에서 ──
# ── 잔고 1등 후보를 누적 저장. 섹션별로 부분 갱신 (다른 섹션 보존).      ──
_REGIME_PICKS_PATH = Path("reports/regime_picks.json")


def _update_regime_picks(section: str, data) -> Path:
    """reports/regime_picks.json의 한 섹션만 갱신 (다른 섹션 보존)."""
    _REGIME_PICKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if _REGIME_PICKS_PATH.exists():
        payload = json.loads(_REGIME_PICKS_PATH.read_text(encoding="utf-8"))
    payload[section] = data
    _REGIME_PICKS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return _REGIME_PICKS_PATH


def run_nsga3(trials: int, seed: int | None = None,
              storage: str | None = None, study_name: str = "nsga3_v2_weights",
              tune_params: bool = False, population_size: int = 50,
              early_stop_window: int | None = None,
              adaptive_mutation: bool = False) -> None:
    """Optuna NSGA-III — 국면별 라이벌(DCA)전 5목적 + 턴오버.
    결과는 챔피언 1명이 아니라 Pareto front(트레이더 라인업)다.
    기본 = 가중치 전용 탐색(A안). tune_params=True는 고도화 단계용.
    early_stop_window: HV MA(window) 정체 시 self stop (None=끔).
    adaptive_mutation: True면 HV 정체/개선 신호로 mutation_prob 자동 조정."""
    _apply_seed(seed)
    from app.academy.study import nsga3   # optuna는 이 모드에서만 필요 — 지연 import

    n_weights = len(ALL_GENES)
    space = (f"가중치 {n_weights} + 파라미터 7"
             if tune_params else f"가중치 {n_weights} (파라미터 기본값 고정)")
    notes = []
    if early_stop_window:
        notes.append(f"HV-MA({early_stop_window}) 얼리스탑")
    if adaptive_mutation:
        notes.append("적응형 mutation")
    note_str = (" · " + " · ".join(notes)) if notes else ""
    print("=== PocketQuant NSGA-III 다목적 최적화 ===")
    print(f"트라이얼 {trials} · 목적 {nsga3.OBJECTIVE_NAMES} · X = {space}"
          f" · 인구 {population_size} · 시드 {seed}{note_str}\n")

    def on_progress(done, total, front_size):
        print(f"  [{done:>5}/{total}] Pareto front {front_size}개")

    study, loaded_gyms, dca, hv_cb, mut_cb = nsga3.run_study(
        trials, seed=seed, storage=storage, study_name=study_name,
        tune_params=tune_params, on_progress=on_progress,
        population_size=population_size, early_stop_window=early_stop_window,
        adaptive_mutation=adaptive_mutation)

    if hv_cb and hv_cb.hv:
        curve = " → ".join(f"{v:.4f}" for v in hv_cb.hv)
        tail = " (정체로 중단)" if getattr(hv_cb, "stopped", False) else ""
        print(f"\nHV 곡선 ({len(hv_cb.hv)}세대){tail}: {curve}")
    if mut_cb and mut_cb.history:
        print(f"\n적응형 mutation 궤적 ({len(mut_cb.history)}세대):")
        for h in mut_cb.history:
            arrow = "↓" if h["improved"] else "↑"
            print(f"  g{h['gen']:>2} HV {h['hv']:.4f} MA {h['ma']:.4f} "
                  f"{arrow} mut_prob {h['mut_prob']:.4f}")

    # 비교 기준: 현 단일목적 챔피언 (동일가중 VOL+REV_RSI+REV_BB)
    ref = nsga3.reference_vector(loaded_gyms, dca)
    print("\n=== 기준점: 현 챔피언 VOL+REV_RSI+REV_BB (동일가중·기본 파라미터) ===")
    print("  " + _format_objective_vector(
        [min(ref["dotcom"], ref["gfc"]), ref["rebound"], ref["crash_v"],
         ref["bull"], ref["chop"], ref["turnover"]]))

    summary = nsga3.summarize_front(study, loaded_gyms=loaded_gyms, dca=dca)
    print(f"\n=== Pareto front {summary['front_size']}개 → 하드 필터 통과 "
          f"{len(summary['passed'])}개 (전 국면 ≥ -{summary['tolerance'] * 100:.0f}, "
          f"턴오버 ≤ {summary['turnover_cap']}) ===")

    for label, row in summary["labels"].items():
        print(f"\n[{label}]  trial #{row['number']}")
        print("  " + _format_objective_vector(row["values"]))
        print("  " + _format_candidate_params(row["params"]))

    if not summary["labels"]:
        print("\n  ⚠️ 필터 통과 후보 없음 — tolerance/turnover_cap을 조정해 다시 보세요.")
        return

    # ── 표시·판정용 잔고 (100만원 시드) — 사람이 보는 층, 옵티마이저 아님 ──
    bals = {}
    for r in summary["passed"]:
        w, sig = nsga3.decode_params(r["params"])
        bals[r["number"]] = nsga3.evaluate_balances(w, sig, loaded_gyms, dca)

    # 체육관별 짧은 별명 (표 칼럼용) — GYM_KEYS의 한글 토큰("닷컴", "금융위기"...)
    nick = {lg.gym.name: next(t for t in nsga3.GYM_KEYS if t in lg.gym.name)
            for lg in loaded_gyms}
    gym_order = [lg.gym.name for lg in loaded_gyms]

    print("\n=== 라벨 후보 잔고 (100만원 시드 → 종료 잔고, 단위 만원) ===")
    head = "  " + f"{'후보':<10}{'라벨':<14}" + "".join(f"{nick[g]:>9}" for g in gym_order)
    print(head)
    for label, row in summary["labels"].items():
        b = bals[row["number"]]
        cells = "".join(f"{b[g]['strat']/10000:>9.1f}" for g in gym_order)
        print(f"  #{row['number']:<9}{label:<14}{cells}")
    sample = next(iter(bals.values()))
    dca_cells = "".join(f"{sample[g]['dca']/10000:>9.1f}" for g in gym_order)
    print(f"  {'성실이':<10}{'(DCA)':<14}{dca_cells}")

    # ── gate0_training: 6체육관 각각 잔고 1등 (통과 후보 중) → Regime Scanner 입력 ──
    gate0 = []
    for g in gym_order:
        win_num, win_b = max(bals.items(), key=lambda kv: kv[1][g]["strat"])
        gate0.append({"gym": g, "nick": nick[g], "winner": f"#{win_num}",
                      "잔고": win_b[g]["strat"], "성실이": win_b[g]["dca"],
                      "차": win_b[g]["strat"] - win_b[g]["dca"]})

    print("\n=== 훈련장 1등 (체육관별, 통과 후보 중 잔고 최고) ===")
    for e in gate0:
        sign = "+" if e["차"] >= 0 else ""
        print(f"  {e['nick']:<8} {e['winner']:<8} {e['잔고']:>10,}원  "
              f"(성실이 {e['성실이']:>10,}원, {sign}{e['차']:,})")

    out = _update_regime_picks("gate0_training", gate0)
    print(f"\nsaved: {out}")
