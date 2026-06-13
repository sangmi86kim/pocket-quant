"""e2e.py - 전 파이프라인 스모크 검증

폴더 재구성·data.py 이전·골든 갱신 같은 큰 변경 후 한 번 돌려 본다.
각 단계는 subprocess로 격리 호출 — 한 단계가 깨져도 다음 단계는 진행.

검증 단계 (소요 시간 짧은 순):
  ① compileall          — 모든 .py 컴파일 가능?
  ② 도감 출력           — app.lab.dex 모듈 import + print_pokedex 동작?
  ③ 단위 게이트 4종     — 가중결합·룩어헤드·골든·돼지저금통
  ④ 진단 2종           — check_signals · check_dca
  ⑤ walk_forward       — 선발 과정 OOS 게이트 (22폴드, 수십 초)
  ⑥ NSGA-III smoke     — service.run_nsga3 작은 설정으로 실행 (trials 30, pop 10)

총 소요 시간: 2~5분 예상. 마지막에 종합 표.

실행: python tools/e2e.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# (이름, 명령, 타임아웃 초)
STAGES = [
    ("compileall",
     [sys.executable, "-m", "compileall", "-q", "main.py", "app", "tools"], 60),
    ("도감 출력 (app.lab.dex)",
     [sys.executable, "-m", "app.lab.dex"], 30),
    ("게이트: 가중결합 불변식",
     [sys.executable, "tools/test_weighted_combine.py"], 60),
    ("게이트: 룩어헤드 검사",
     [sys.executable, "tools/test_no_lookahead.py"], 60),
    ("게이트: 골든 넘버 (엔진 불변)",
     [sys.executable, "tools/test_engine_regression.py"], 60),
    # 풀 13마리 + 외부 데이터 fetch — 첫 캐시 미스 시 시간 더 듦. timeout 여유 늘림.
    ("게이트: 돼지저금통 퇴화 감시",
     [sys.executable, "tools/test_baselines.py"], 600),
    ("진단: 시그널 노출/발동률",
     [sys.executable, "tools/check_signals.py"], 120),
    ("진단: DCA 기준선 + score_vs_dca",
     [sys.executable, "tools/check_dca.py"], 900),
    # walk_forward는 legacy 단일목적 GA(클램프 스탯) 기반 — 풀 13마리에선 2^13 조합
    # 평가 폭증 + AGENTS.md §6 "클램프 스탯 금지"와도 안 맞아 e2e에서 제외.
    # 별도 도구로 유지 — 자산/기간 민감도 실험 시 직접 호출.
    # NSGA-III는 storage=None + trials 작게로 종속 없이 스모크.
    # config.json을 안 건드리려고 service.run_nsga3 직접 호출.
    ("NSGA-III smoke (trials 30, pop 10)",
     [sys.executable, "-c",
      "from app.service import run_nsga3; "
      "run_nsga3(trials=30, seed=42, storage=None, "
      "study_name='e2e_smoke', tune_params=False, "
      "population_size=10, early_stop_window=None, "
      "adaptive_mutation=False)"], 180),
]


def run_stage(name: str, cmd: list[str], timeout: int) -> tuple[bool, float, str]:
    """한 단계 실행. (성공 여부, 소요 시간, 짧은 메시지)."""
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, time.perf_counter() - t0, f"TIMEOUT ({timeout}s)"
    elapsed = time.perf_counter() - t0
    ok = result.returncode == 0
    # 실패하면 마지막 몇 줄을 보여줘서 원인을 빨리 짚을 수 있게.
    if not ok:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-5:]
        return False, elapsed, "rc=" + str(result.returncode) + " | " + " / ".join(tail)
    return True, elapsed, ""


def main() -> int:
    print("=== PocketQuant e2e 스모크 ===")
    print(f"루트: {ROOT}\n")
    rows = []
    for name, cmd, timeout in STAGES:
        print(f"▶ {name} ...", flush=True)
        ok, elapsed, note = run_stage(name, cmd, timeout)
        flag = "PASS" if ok else "FAIL"
        print(f"  {flag}  ({elapsed:5.1f}s){'  ' + note if note else ''}")
        rows.append((name, ok, elapsed, note))

    # 종합 표
    print("\n=== 결과 ===")
    print(f"  {'단계':<40}{'결과':<8}{'시간':>8}")
    for name, ok, elapsed, note in rows:
        flag = "PASS" if ok else "FAIL"
        print(f"  {name:<40}{flag:<8}{elapsed:>7.1f}s")
    total = sum(e for _, _, e, _ in rows)
    fails = [name for name, ok, _, _ in rows if not ok]
    print(f"\n  합계 {total:.1f}s · {'전 단계 PASS' if not fails else f'FAIL {len(fails)}건: ' + ', '.join(fails)}")
    # yfinance 메타(cookies.db/tkr-tz.db) 청소는 data.py의 atexit이 알아서 한다.
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
