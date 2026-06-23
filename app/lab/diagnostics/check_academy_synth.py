"""
check_academy_synth.py - 아카데미 합성세계(textbook + curriculum) 산출물 검증

[검증 범위]
`app/academy/curriculum/textbook.py`(평행세계 1권)와 `curriculum/course.py`
(N권 학기 코스 + train/validation split) 계약을 고정한다.

  ① LoadedGym 확장 없이 prices.attrs만 사용
  ② QQQ, QQQ_volume, 외부 6개 스트림이 모두 존재
  ③ 모든 스트림은 합성 prices와 같은 인덱스를 공유
  ④ QQQ 상대강도 leg는 합성 prices와 동일
  ⑤ UUP처럼 늦게 생긴 원천은 없는 구간을 NaN으로 보존
  ⑥ signals._fetch_external은 attrs를 우선 읽고, 없으면 synthetic NaN 기권
  ⑦ VOL_SPIKE는 QQQ_volume attrs를 읽는다
  ⑧ academy.bootstrap_gyms도 textbook 산출물을 사용한다
  ⑨ train/validation synthetic split은 서로 다른 세계를 만든다
  ⑩ raw balance sum 목적식 계측은 그대로 가능하다
  ⑪ hold-out(2020-07~) 재료를 쓰지 않는다
  ⑫ QQQ_SPY/QQQ_DIA도 attrs 경로로 합성 외부 스트림을 읽는다
  ⑬ 단일목적 objective는 아직 raw balance sum 그대로다

실행: 프로젝트 루트에서  python -m app.lab.diagnostics.check_academy_synth
"""
import numpy as np
import optuna
import pandas as pd

from app.academy.curriculum.course import bootstrap_gyms, prepare_academy_split
from app.academy.curriculum.textbook import DATA_END, make_world, make_world_rs
from app.academy.exam.grade import evaluate_balances
from app.academy.training.single_objective.engine import _objective
from app.pocket.signals import (
    SIGNAL_NAMES,
    _fetch_external,
    signal_QQQ_DIA,
    signal_QQQ_SPY,
    signal_VOL_SPIKE,
)

EXPECTED_STREAMS = ["DIA", "QQQ", "QQQ_volume", "SPY", "TLT", "UUP", "^TNX", "^VIX", "^VXN"]


def run_check() -> bool:
    failures: list[str] = []

    def check(label: str, ok: bool):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            failures.append(label)

    print("=== 아카데미 세계공장 검증 ===")
    world = make_world(seed=1, n_days=300)
    prices = world.prices
    streams = prices.attrs.get("external_streams", {})
    rs_world = make_world_rs(seed=2, n_days=300)
    rs_prices = rs_world.prices
    rs_streams = rs_prices.attrs.get("external_streams", {})

    check("합성 세계 표식: prices.attrs['synthetic'] == True",
          prices.attrs.get("synthetic") is True)
    check("LoadedGym 필드 확장 없음: external_streams는 prices.attrs에만 있음",
          not hasattr(world, "external_streams"))
    check("야생 입력 스트림 키 고정",
          sorted(streams) == EXPECTED_STREAMS)
    check("RS 교과서도 야생 입력 스트림 키 고정",
          sorted(rs_streams) == EXPECTED_STREAMS)
    check("QQQ leg는 합성 prices와 동일",
          "QQQ" in streams and streams["QQQ"].equals(prices))
    check("모든 스트림 인덱스가 합성 prices와 동일",
          all(s.index.equals(prices.index) for s in streams.values()))
    check("QQQ 거래량 스트림 존재 및 전 구간 유효",
          "QQQ_volume" in streams and streams["QQQ_volume"].notna().all())
    check("UUP 상장 전 부재 구간은 NaN으로 보존",
          "UUP" in streams and streams["UUP"].isna().sum() > 0)
    check("합성 재료는 hold-out 시작 전까지만 사용",
          DATA_END < "2020-07-01")
    default_world = bootstrap_gyms(n=1, seed=8)[0]
    check("academy.bootstrap_gyms 기본 교과서는 RS",
          default_world.gym.name.startswith("아카데미#")
          and default_world.prices.attrs.get("synthetic") is True)
    check("signals._fetch_external은 attrs 외부 스트림을 우선 사용",
          _fetch_external("^VIX", prices).equals(streams["^VIX"]))
    bare_synth = pd.Series(prices.to_numpy(), index=prices.index, name=prices.name)
    bare_synth.attrs["synthetic"] = True
    check("attrs 없는 synthetic 외부 시그널은 NaN 기권",
          _fetch_external("^VIX", bare_synth).isna().all())

    idx = pd.bdate_range("2001-01-01", periods=25)
    spike_prices = pd.Series([100.0] * 24 + [99.0], index=idx, name="QQQ")
    spike_volume = pd.Series([100.0] * 24 + [1000.0], index=idx)
    spike_prices.attrs["synthetic"] = True
    spike_prices.attrs["external_streams"] = {"QQQ_volume": spike_volume}
    check("VOL_SPIKE는 synthetic QQQ_volume attrs를 읽음",
          signal_VOL_SPIKE(spike_prices).iloc[-1] == 1.0)
    ratio_idx = pd.bdate_range("2001-01-01", periods=80)
    ratio_prices = pd.Series([100.0] * 80, index=ratio_idx, name="QQQ")
    ratio_prices.attrs["synthetic"] = True
    ratio_prices.attrs["external_streams"] = {
        "QQQ": pd.Series([100.0 + i for i in range(80)], index=ratio_idx),
        "SPY": pd.Series([100.0] * 80, index=ratio_idx),
        "DIA": pd.Series([100.0] * 80, index=ratio_idx),
    }
    check("QQQ_SPY는 synthetic QQQ/SPY attrs를 읽음",
          signal_QQQ_SPY(ratio_prices).iloc[-1] == 1.0)
    check("QQQ_DIA는 synthetic QQQ/DIA attrs를 읽음",
          signal_QQQ_DIA(ratio_prices).iloc[-1] == 1.0)
    gyms_a = bootstrap_gyms(n=2, seed=7)
    gyms_b = bootstrap_gyms(n=2, seed=7)
    check("academy.bootstrap_gyms는 합성 외부 스트림을 보존",
          all(sorted(g.prices.attrs.get("external_streams", {})) == EXPECTED_STREAMS
              for g in gyms_a))
    check("academy.bootstrap_gyms는 같은 seed에서 재현 가능",
          all(a.prices.equals(b.prices)
              for a, b in zip(gyms_a, gyms_b)))
    check("academy.bootstrap_gyms는 체육관마다 다른 시험지를 생성",
          not gyms_a[0].prices.equals(gyms_a[1].prices))
    (train_gyms, train_dca), (val_gyms, val_dca) = prepare_academy_split(
        n_train=2, n_validation=2, train_seed=11, validation_seed=111)
    check("train/validation split은 둘 다 외부 스트림을 보존",
          all(sorted(g.prices.attrs.get("external_streams", {})) == EXPECTED_STREAMS
              for g in train_gyms + val_gyms))
    check("train/validation split은 서로 다른 합성 세계",
          not train_gyms[0].prices.equals(val_gyms[0].prices))
    try:
        prepare_academy_split(n_train=1, n_validation=1, train_seed=5, validation_seed=5)
        split_seed_guard = False
    except ValueError:
        split_seed_guard = True
    check("train/validation seed 충돌은 즉시 차단",
          split_seed_guard)
    equal_weights = [1.0] * len(SIGNAL_NAMES)
    train_bals = evaluate_balances(equal_weights, {}, train_gyms, train_dca)
    val_bals = evaluate_balances(equal_weights, {}, val_gyms, val_dca)
    check("raw balance sum 계측은 train/validation 모두 가능",
          sum(b["strat"] for b in train_bals.values()) > 0
          and sum(b["strat"] for b in val_bals.values()) > 0)
    fixed_params = {f"w_{g}": 1.0 for g in SIGNAL_NAMES}
    trial = optuna.trial.FixedTrial(fixed_params)
    # 목적값 = 체육관 잔고 중앙값(median). 2026-06-20 balance_sum→median 전환과 정합.
    expected = float(np.median([b["strat"] for b in train_bals.values()]))
    check("단일목적 objective는 raw balance median 유지",
          _objective(trial, train_gyms, train_dca) == expected)  # type: ignore[arg-type]

    print(f"\n=== 판정: {'PASS' if not failures else 'FAIL ' + str(failures)} ===")
    return not failures


def test_academy_synth_contract():
    assert run_check(), "아카데미 세계공장 산출물 계약 위반"


if __name__ == "__main__":
    raise SystemExit(0 if run_check() else 1)
