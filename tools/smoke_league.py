# ruff: noqa: E402
"""smoke_league.py — 리그(league) 워크플로우 스모크.

"졸업생 리그 태우기"가 끊김 없이 도는지만 본다(연결 확인용, 성능 아님).
시즌3 리그는 두 관문만 본다: ① 빅토리 로드 (OOS, victory_road) → ② 사천왕(elite_four).

게이트 본실행(run_gate1/run_gate3)은 풀 백테스트라 무겁다 — 스모크는 그 밑단
빌딩블록(연도/라운드 1~2개 채점)만 가볍게 태워 경로가 살아있는지 확인한다.

순서:
  ① 빅토리 로드 (OOS): 졸업생 1명을 victory_road 앞 2개 연도에 응시 → 종료잔고 > 0
  ② 사천왕 : 같은 졸업생을 elite_four 첫 라운드에 응시 → 종료잔고 > 0
  ③ v3 계약: season3_league 130명 runner 확인

실행: 프로젝트 루트에서  python tools/smoke_league.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")    # type: ignore[union-attr]
    except Exception:
        pass

from app.league import elite_four as EF
from app.league import victory_road as VR
from app.league.v3 import season3_league as v3
from app.pocket.battle import _score_position, terminal_balance
from app.pocket.signals import SIGNAL_NAMES, combine_positions, positions_with_params
from app.world.data_loader import get_prices

SEED_KRW = 1_000_000
OOS_SMOKE_YEARS = 2     # 빅토리 로드 (OOS) 앞 몇 개 연도만 (스모크라 소수)


def _graduate_balance(loaded, weights) -> int:
    """졸업생(가중치) 1명을 한 구간에 응시시켜 종료잔고를 낸다(기본 시그널 파라미터)."""
    pos = combine_positions(positions_with_params(loaded.prices, {}), weights)
    return terminal_balance(_score_position(pos, loaded), SEED_KRW)


def _check_oos(prices, weights) -> float:
    print("\n=== ① 빅토리 로드 (OOS) smoke (victory_road) ===")
    t0 = time.perf_counter()
    for year in VR.OOS_YEARS[:OOS_SMOKE_YEARS]:
        loaded = VR._loaded_window(prices, year)
        bal = _graduate_balance(loaded, weights)
        if bal <= 0:
            raise RuntimeError(f"빅토리 로드 (OOS) {year} 종료잔고 0 — 채점 경로 깨짐")
        print(f"  [PASS] 빅토리 로드 (OOS) {year} 종료잔고 {bal/10000:8.1f}만")
    return time.perf_counter() - t0


def _check_holdout(prices, weights) -> float:
    print("\n=== ② 사천왕 관문 smoke (elite_four) ===")
    t0 = time.perf_counter()
    name, start, end = EF.ROUNDS[0]
    loaded = EF._loaded_window(prices, start, end)
    bal = _graduate_balance(loaded, weights)
    if bal <= 0:
        raise RuntimeError(f"사천왕 '{name}' 종료잔고 0 — 채점 경로 깨짐")
    print(f"  [PASS] 사천왕 {name:<8} 종료잔고 {bal/10000:8.1f}만")
    return time.perf_counter() - t0


def _check_v3_contract() -> float:
    print("\n=== ③ 리그 v3 130명 본경기 계약 ===")
    t0 = time.perf_counter()
    stages = [s[0] for s in v3.STAGES]
    if stages != ["oos", "holdout"]:
        raise RuntimeError(f"v3 STAGES 순서 어긋남: {stages}")
    payload = v3.run()
    if payload.get("candidate_count") != 130:
        raise RuntimeError(f"v3 출전 인원 어긋남: {payload.get('candidate_count')}")
    if not {"overall", "oos", "holdout"} <= set(payload.get("summary", {}).get("GP-보충", {})):
        raise RuntimeError("v3 GP summary 스키마 어긋남")
    print(f"  [PASS] STAGES {stages} · candidates {payload['candidate_count']}명")
    return time.perf_counter() - t0


def main() -> int:
    print("=== PocketQuant 리그(league) smoke ===")
    print("관문: 빅토리 로드 (OOS, victory_road) → 사천왕(elite_four) · v3 130명 본경기\n")
    weights = [1.0] * len(SIGNAL_NAMES)   # 졸업생 대역 — 동일가중 1명
    rows = []
    try:
        prices = get_prices(VR.TICKER, "1999-03-10", "2026-06-09")
        rows.append(("빅토리 로드 (OOS)", _check_oos(prices, weights)))
        rows.append(("사천왕", _check_holdout(prices, weights)))
        rows.append(("v3 계약", _check_v3_contract()))
    except Exception as exc:
        print(f"\n=== 판정: FAIL ===\n{exc}")
        return 1

    print("\n=== 결과 ===")
    for name, elapsed in rows:
        print(f"  {name:<8} PASS  {elapsed:5.1f}s")
    print(f"\n=== 판정: PASS · 총 {sum(e for _, e in rows):.1f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
