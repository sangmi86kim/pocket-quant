"""Project-wide validation smoke.

Runs compile/import gates, regression validators, diagnostics, and a short
NSGA-III smoke. For the shorter academy -> gym -> league workflow check, run
`python tools/smoke_workflow.py`.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# (stage name, command, timeout seconds)
STAGES = [
    ("compileall",
     [sys.executable, "-m", "compileall", "-q", "app", "tools"], 60),
    ("dex import",
     [sys.executable, "-m", "app.pocket.dex"], 30),
    ("gate: academy synth",
     [sys.executable, "-m", "app.lab.check_academy_synth"], 60),
    ("gate: weighted combine",
     [sys.executable, "tools/test_weighted_combine.py"], 60),
    ("gate: no lookahead",
     [sys.executable, "tools/test_no_lookahead.py"], 60),
    ("gate: engine regression",
     [sys.executable, "tools/test_engine_regression.py"], 60),
    ("gate: baselines",
     [sys.executable, "tools/test_baselines.py"], 600),
    ("diagnostic: signals",
     [sys.executable, "-m", "app.lab.check_signals"], 120),
    ("diagnostic: dca",
     [sys.executable, "-m", "app.lab.check_dca"], 900),
    # Keep this dependency-free: storage=None, fixed seed, and tiny trials.
    ("NSGA-III smoke (trials 30, pop 10)",
     [sys.executable, "-c",
      "import app.academy.training.multi_objective.nsga3 as nsga3; "
      "nsga3.run_study(30, seed=42, storage=None, "
      "study_name='e2e_smoke', tune_params=False, population_size=10)"], 180),
]


def run_stage(name: str, cmd: list[str], timeout: int) -> tuple[bool, float, str]:
    """Run one isolated validation stage."""
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
    if not ok:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-5:]
        return False, elapsed, "rc=" + str(result.returncode) + " | " + " / ".join(tail)
    return True, elapsed, ""


def main() -> int:
    print("=== PocketQuant e2e smoke ===")
    print(f"root: {ROOT}\n")
    rows = []
    for name, cmd, timeout in STAGES:
        print(f"- {name} ...", flush=True)
        ok, elapsed, note = run_stage(name, cmd, timeout)
        flag = "PASS" if ok else "FAIL"
        print(f"  {flag}  ({elapsed:5.1f}s){'  ' + note if note else ''}")
        rows.append((name, ok, elapsed, note))

    print("\n=== Result ===")
    print(f"  {'stage':<40}{'result':<8}{'time':>8}")
    for name, ok, elapsed, note in rows:
        flag = "PASS" if ok else "FAIL"
        print(f"  {name:<40}{flag:<8}{elapsed:>7.1f}s")
    total = sum(e for _, _, e, _ in rows)
    fails = [name for name, ok, _, _ in rows if not ok]
    summary = "all stages PASS" if not fails else f"FAIL {len(fails)}: " + ", ".join(fails)
    print(f"\n  total {total:.1f}s - {summary}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
