# ruff: noqa: E402
"""Validate market/data boundaries between academy workflows.

This is a leakage guard:
- training classrooms must study synthetic academy gyms, not official exam gyms
- exam owns the official QQQ gym list
- league gates may look at OOS/frontier/hold-out markets, but training may not
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from app.academy.exam import all_gyms
from app.academy.training import study
from app.academy.training.multi_objective import nsga3
from app.academy.training.single_objective import tpe
from app.league import battle_frontier, elite_four, victory_road
from app.world.data_loader import FUTURE_SEAL_DATE, get_prices

TRAINING_DIR = ROOT / "app" / "academy" / "training"

OFFICIAL_EXAM_YEARS = {
    2000, 2001, 2002,
    2008, 2009, 2010,
    2015, 2016, 2017, 2020,
}


def _gym_rows(label: str, loaded_or_gyms) -> list[dict]:
    rows = []
    for item in loaded_or_gyms:
        gym = getattr(item, "gym", item)
        rows.append({
            "workflow": label,
            "name": gym.name,
            "ticker": gym.ticker,
            "start": gym.start,
            "end": gym.end,
        })
    return rows


def _assert_all_synth(label: str, rows: list[dict]) -> None:
    bad = [r for r in rows if r["ticker"] != "SYNTH"]
    if bad:
        raise AssertionError(f"{label} saw non-SYNTH gyms: {bad[:3]}")


def _assert_all_qqq(label: str, rows: list[dict]) -> None:
    bad = [r for r in rows if r["ticker"] != "QQQ"]
    if bad:
        raise AssertionError(f"{label} saw non-QQQ gyms: {bad[:3]}")


def _scan_training_for_exam_leaks() -> list[tuple[Path, str]]:
    forbidden = [
        "from app.academy.exam import all_gyms",
        "load_gyms(all_gyms",
        "all_gyms()",
    ]
    hits = []
    for path in TRAINING_DIR.rglob("*.py"):
        rel = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            if pattern in text:
                hits.append((rel, pattern))
    return hits


def _print_rows(rows: list[dict]) -> None:
    print(f"{'workflow':<28} {'ticker':<7} {'start':<10} {'end':<10} name")
    for row in rows:
        print(
            f"{row['workflow']:<28} {row['ticker']:<7} "
            f"{row['start']:<10} {row['end']:<10} {row['name']}"
        )


def main() -> int:
    rows = []

    school_gyms, _school_dca = study.prepare_school_data(n_gyms=2, seed=101)
    school_rows = _gym_rows("training.study", school_gyms)
    _assert_all_synth("training.study", school_rows)
    rows.extend(school_rows)

    single_gyms, _single_dca = tpe.prepare_data(n_gyms=2, seed=202)
    single_rows = _gym_rows("single default", single_gyms)
    _assert_all_synth("single default", single_rows)
    rows.extend(single_rows)

    nsga_gyms, _nsga_dca = nsga3.prepare_data(n_gyms=2, seed=303)
    nsga_rows = _gym_rows("nsga default", nsga_gyms)
    _assert_all_synth("nsga default", nsga_rows)
    rows.extend(nsga_rows)

    exam_rows = _gym_rows("exam.official", all_gyms())
    _assert_all_qqq("exam.official", exam_rows)
    rows.extend(exam_rows)

    print("=== Workflow Market Map ===")
    _print_rows(rows)

    print("\n=== League Gate Constants ===")
    print(f"victory_road: ticker={victory_road.TICKER} "
          f"oos_years={victory_road.OOS_YEARS}")
    print(f"battle_frontier: data={battle_frontier.DATA_START}"
          f"~{battle_frontier.DATA_END} ticker=QQQ bootstrap")
    print(f"elite_four: holdout={elite_four.HOLDOUT_START}"
          f"~{elite_four.DATA_END} ticker={elite_four.TICKER}")

    if set(victory_road.OOS_YEARS) & OFFICIAL_EXAM_YEARS:
        overlap = sorted(set(victory_road.OOS_YEARS) & OFFICIAL_EXAM_YEARS)
        raise AssertionError(f"victory_road overlaps official exam years: {overlap}")
    if battle_frontier.DATA_END > "2020-06-30":
        raise AssertionError("battle_frontier reaches into hold-out")
    if elite_four.HOLDOUT_START < "2020-07-01":
        raise AssertionError("elite_four starts before hold-out boundary")

    # 미래 봉인(DLC hold-out) — 챔피언 결정전용으로 적립 중인 미래 데이터를 학습/검증이 못 보게
    # get_prices가 기본으로 SEAL 이후를 잘라내는지 잠근다. 지금은 실데이터가 SEAL에 못 미쳐
    # 자명히 통과하지만, 달이 쌓여 SEAL 이후 데이터가 생겨도 이 불변식이 봉인을 강제한다.
    print("\n=== Future Seal (DLC hold-out) ===")
    print(f"FUTURE_SEAL_DATE = {FUTURE_SEAL_DATE}")
    if FUTURE_SEAL_DATE <= "2020-07-01":
        raise AssertionError("FUTURE_SEAL_DATE가 사천왕 hold-out 경계보다 이르다")
    seal_ts = pd.Timestamp(FUTURE_SEAL_DATE)
    sealed = get_prices("QQQ", "1999-03-10", "2099-12-31")          # 봉인 너머까지 요청
    if sealed.index.max() > seal_ts:
        raise AssertionError(
            f"봉인 누수: 기본 get_prices가 {sealed.index.max().date()} "
            f"(> SEAL {FUTURE_SEAL_DATE})를 반환")
    print(f"  기본(allow_future=False) 마지막 {sealed.index.max().date()} ≤ SEAL ✓")
    opened = get_prices("QQQ", "1999-03-10", "2099-12-31", allow_future=True)  # DLC 개봉 경로
    if opened.empty:
        raise AssertionError("allow_future=True(DLC 개봉) 경로가 빈 시계열 반환")
    print(f"  DLC 개봉(allow_future=True) 호출 가능 · 마지막 {opened.index.max().date()}")

    hits = _scan_training_for_exam_leaks()
    if hits:
        for rel, pattern in hits:
            print(f"[LEAK] {rel}: {pattern}")
        raise AssertionError("training contains official exam data path")

    print("\n=== Verdict: PASS ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
