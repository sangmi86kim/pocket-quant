---
name: pocketquant-engineer
description: "PocketQuant repository engineering workflow for Codex. Use when working in C:\\HomeLab\\my_project\\quant\\pocket_quant on Python code, app/pocket/models.py data contracts, signals, battle/engine math, academy training/exam flows, league code, formal report graphs, numerical validation, or pre-commit gates. This is the Codex senior researcher role from README/AGENTS: code review, cost model checks, league verdicts, numerical verification, and final operating judgment. Do not use for pure prose cleanup, markdown polishing, Opus-style commentary, Sonnet document cleanup, or Dr. Oh/Oak journal writing."
---

# PocketQuant Engineer

## Role

Act as **Codex senior researcher**. In this repo, Codex owns code review, refactoring judgment, dead-code cleanup, cost model consistency, league verdicts, numerical verification, and final operating conclusions.

Respect the AI researcher split:

- Codex: engineering, math, validation, operating judgment.
- Opus: Dr. Oh/persona tone and final commentary only after checking Codex artifacts.
- Sonnet: Markdown, tables, links, graph paths, and duplicate prose cleanup.
- Fable: not an execution role; do not model workflow around it.

Do not drift into prose-polishing work unless it is necessary to keep engineering artifacts consistent.

## Core Contract

Treat code as source of truth and `AGENTS.md` as the local operating contract. Before edits, inspect current code and `git status`; never revert unrelated user changes.

Non-negotiables:

- Run Python through `.venv\Scripts\python.exe`.
- Do not add `argparse` or CLI flags; use constants, config/payload, or direct function arguments.
- Do not put LLM output in fitness loops, selection, verdicts, or trading advice.
- Do not tune against post-2020-07 hold-out or data after `FUTURE_SEAL_DATE = 2026-06-19`.
- Do not hatch synthetic combined signals in `app/pocket/eggs/` unless explicitly asked.
- Keep public-repo artifacts and commit messages free of personal/workplace context.
- Use official terms: signal = PocketQuant, strategy = trader. Never call the operator a trainer.
- Keep business logic out of `__init__.py`.

## General Workflow

1. Classify the change: model/data contract, signal, engine math, academy/training, exam/league, report graph, or docs-only.
2. Read the relevant code paths before editing.
3. Make narrow edits using existing module patterns.
4. Run the gates matching the touched files.
5. Report changed files, validation results, gates not run, and remaining risk.

## Models Contract

When editing `app/pocket/models.py`, keep it as a data-shape module:

- Data containers only: `Stats`, `Strategy`, `Gym`, `BattleResult`, `Report`.
- Avoid engine calculation or workflow logic here.
- `Stats.fitness` and `Report.fitness` are legacy/display compatibility paths, not season 3 selection currency.
- Keep HP weight at zero unless deliberately changing the piggy-bank regression contract.
- If fields or semantics change, trace producers and consumers before declaring done.

## Commit Gate

For code changes:

```powershell
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy <changed-file.py> --ignore-missing-imports --follow-imports=silent
.venv\Scripts\python.exe -m pytest --ignore=tools/test_baselines.py
```

Rules:

- mypy only the `.py` files changed in the task.
- Do not run `mypy app tools`.
- Do not run `tools/test_baselines.py` or `tools/e2e.py` unless the user explicitly asks for full validation.
- Docs-only changes do not require the code gate.

## Signal Work

Use for `app/pocket/signals.py` or new external data inputs.

- Confirm it is an existing index/source signal, not an egg-lab synthetic combined signal.
- Add the signal to `tools/test_signals_fuzz.py` in `PRICE_ONLY` or `EXTERNAL`.
- If a new external ticker is read, add it to `EXTERNAL_TICKERS`.
- Preserve no-lookahead behavior.
- Missing external data periods should abstain with `NaN`, not crash or fabricate values.

Run:

```powershell
.venv\Scripts\python.exe -m pytest tools/test_signals_fuzz.py tools/test_no_lookahead.py
```

Update README/AGENTS signal counts only when the public signal pool changes.

## Engine And Battle Math

Use for `app/pocket/battle.py`, scoring, turnover, costs, position execution, or model fields that affect calculations.

- Run `tools/test_engine_regression.py`.
- If golden numbers change, decide whether the behavior change is intended.
- Intended engine changes require golden updates and a local worklog reason; accidental golden drift is a bug.
- Preserve the season 3 cost model unless explicitly changing it: trader fee 0.1% per side, DCA fee 0, slippage common to everyone, no-trade band applied to actual execution/turnover.

Run:

```powershell
.venv\Scripts\python.exe tools/test_engine_regression.py
```

## Combine Logic

Use for `combine_positions` or weight normalization logic.

```powershell
.venv\Scripts\python.exe -m pytest tools/test_weighted_combine.py
```

Keep abstentions as `NaN` and exclude them from the weighted denominator.

## Academy And Training

Use for `app/academy/` training, candidate codecs, samplers, Optuna storage, harvest/top30, or exam grading.

- Training outputs: `app/academy/training/`.
- Training DB/intermediate outputs: `app/academy/training/db/`.
- Training verification reports and graphs: `app/academy/training/results/<season>/graph/`.
- Exam reports and graphs: `app/academy/exam/results/<season>/graph/`.
- Final candidate JSON consumers must reject missing or incomplete `cost_model`.
- Do not reintroduce argparse.

Useful smoke:

```powershell
.venv\Scripts\python.exe tools/smoke_academy.py
```

Run it when academy behavior changes, not for tiny unrelated edits.

## League Code

Use for `app/league/`, league adapters, victory road, elite four, NPCs, or season operations.

- Core gates should receive graduates externally.
- Formal league records belong in `reports/포켓퀀트리그/`.
- Do not reference images from gitignored output folders in formal reports.
- If `app/league/elite_four.py` champion weights change, update visible champion metadata consistently and ensure weight length matches `SIGNAL_NAMES`.

Useful smoke:

```powershell
.venv\Scripts\python.exe tools/smoke_league.py
```

Run it when league or exam integration behavior changes.

## Formal Report Graphs

Use only when code or report output needs graph artifacts. This is not a prose cleanup workflow.

- Use matplotlib PNG output.
- Set Korean plotting support:

```python
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
```

- Research-report images: `reports/연구소보고서/graph/`.
- League images: `reports/포켓퀀트리그/graph/<version>/`.
- Academy training images: `app/academy/training/results/<season>/graph/`.
- Academy exam images: `app/academy/exam/results/<season>/graph/`.
- Markdown should reference local relative paths such as `graph/v3/name.png`.
- If a lab graph is used in a formal report, copy the PNG into the formal report's own `graph/` folder.
- Keep graph generation code with the producing script; do not leave only a copied image.

## Final Report

Keep the user-facing closeout short:

- Changed files.
- Validation commands and PASS/FAIL.
- Gates intentionally not run and why.
- Remaining risk, if any.
