"""SPY 리그 — 다목적 Top5 + 단일목적 Top5 vs NPC 4명 (자산-횡단 검증).

[가설] QQQ에서 만든 챔피언 가중치가 SPY에서도 통할까?
       다목적·단일목적 어느 쪽이 자산 바꿔도 더 robust한가?

[설계]
- 체육관 6개 = QQQ 기간 정의 그대로 (닷컴~횡보장), 자산만 SPY로 교체.
- 도전자 10명: 다목적 Top5 (QQQ 평행세계 ② 1~5위) + 단일목적 TPE 5시드.
- NPC 4명: 어플삭제맨(B&H) · 저축왕 · 성실이 · 돼지저금통.
- 시드 100만원 × 6체육관 = 600만원 판돈, 종료 잔고 합 매트릭스.

[실행] python app/league/spy_robustness.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd

from app.backend.data_io.data import LoadedGym, WARMUP_DAYS, get_prices
from app.backend.engine.battle import (_score_position, fight_dca, fight_savings,
                                       terminal_balance)
from app.backend.genes.signals import ALL_GENES, combine_positions, positions_with_params
from app.backend.market.gym import all_gyms

# 자산: QQQ가 아니라 SPY로 같은 체육관 기간 재현
ASSET = "SPY"
SEED_KRW = 1_000_000
OUT_MD = _ROOT / "reports" / "spy_robustness.md"

# 다목적 Top5 (QQQ 평행세계 ② 토탈 1~5위) — hall_of_fame 가중치
MULTI_TOP5 = [
    ("TOP06", [0.04, 0.08, 0.00, 0.01, 0.57, 0.30]),   # ② 1위 (QQQ)
    ("TOP01", [0.01, 0.09, 0.00, 0.02, 0.40, 0.49]),   # ② 2위
    ("TOP04", [0.01, 0.14, 0.02, 0.00, 0.53, 0.30]),   # ② 3위
    ("TOP05", [0.01, 0.14, 0.02, 0.00, 0.53, 0.30]),   # ② 4위
    ("TOP03", [0.00, 0.14, 0.02, 0.00, 0.51, 0.32]),   # ② 5위
]
# 단일목적 TPE 5시드 (single_obj_sweep 결과)
TPE_FIVE = [
    ("TPE-s11", [0.000, 0.002, 0.000, 0.000, 0.546, 0.452]),
    ("TPE-s42", [0.000, 0.012, 0.000, 0.000, 0.511, 0.476]),
    ("TPE-s07", [0.000, 0.031, 0.000, 0.000, 0.507, 0.461]),
    ("TPE-s19", [0.000, 0.057, 0.000, 0.000, 0.424, 0.519]),
    ("TPE-s23", [0.000, 0.088, 0.000, 0.000, 0.419, 0.493]),
]


def _load_spy_gyms() -> list[LoadedGym]:
    """QQQ 체육관 기간 정의 그대로, 자산만 SPY로 가격을 받아 LoadedGym 생성."""
    loaded = []
    for gym in all_gyms():
        fetch_start = (pd.Timestamp(gym.start) - pd.Timedelta(days=WARMUP_DAYS)
                       ).strftime("%Y-%m-%d")
        prices = get_prices(ASSET, fetch_start, gym.end)
        loaded.append(LoadedGym(gym=gym, prices=prices))
    return loaded


def eval_weights(weights: list[float], lw: LoadedGym) -> int:
    pos = combine_positions(positions_with_params(lw.prices), weights)
    return terminal_balance(_score_position(pos, lw), SEED_KRW)


def eval_buy_hold(lw: LoadedGym) -> int:
    pos = pd.Series(1.0, index=lw.prices.index)
    return terminal_balance(_score_position(pos, lw), SEED_KRW)


def eval_piggy(lw: LoadedGym) -> int:
    return SEED_KRW


def eval_savings(lw: LoadedGym) -> int:
    return terminal_balance(fight_savings(lw), SEED_KRW)


def eval_dca(lw: LoadedGym) -> int:
    return terminal_balance(fight_dca(lw), SEED_KRW)


def _short(name: str) -> str:
    """체육관 이름 → 표 칼럼용 짧은 별명."""
    for tok in ("닷컴", "금융위기", "회복", "코로나", "상승", "횡보"):
        if tok in name:
            return tok
    return name[:6]


def main() -> None:
    t0 = time.time()
    print(f"=== SPY 리그 — 자산-횡단 검증 (체육관 기간 그대로, 자산 SPY) ===\n")
    print(f"도전자 10명 (다목적 Top5 + 단일목적 TPE 5시드) · NPC 4명 (B&H/저축왕/성실이/돼지저금통)")
    print(f"시드 100만원 × 6체육관 = 600만원 판돈\n")

    loaded_gyms = _load_spy_gyms()
    gym_order = [lg.gym.name for lg in loaded_gyms]

    strategies = MULTI_TOP5 + TPE_FIVE
    label_of = {n: "다목적" for n, _ in MULTI_TOP5} | {n: "단일목적" for n, _ in TPE_FIVE}
    baselines = [("어플삭제맨", eval_buy_hold),
                 ("저축왕", eval_savings),
                 ("성실이", eval_dca),
                 ("돼지저금통", eval_piggy)]
    for n, _ in baselines:
        label_of[n] = "NPC"

    # 매트릭스: {name: {체육관: 잔고원}}
    balances: dict[str, dict[str, int]] = {}
    for name, w in strategies:
        balances[name] = {lg.gym.name: eval_weights(w, lg) for lg in loaded_gyms}
    for name, fn in baselines:
        balances[name] = {lg.gym.name: fn(lg) for lg in loaded_gyms}

    # 토탈 잔고 합 + 순위
    totals = {n: sum(b.values()) for n, b in balances.items()}
    rank = sorted(totals.keys(), key=lambda n: -totals[n])

    print(f"=== 종합 순위 — 6체육관 잔고 합 (총 판돈 600만) ===")
    head = f"  {'순위':<4}{'후보':<12}{'그룹':<8}{'합':>8}  {'성실이 차':>8}"
    print(head)
    dca_sum = totals["성실이"]
    for i, n in enumerate(rank, 1):
        diff = totals[n] - dca_sum
        sgn = "+" if diff >= 0 else ""
        print(f"  {i:<4}{n:<12}{label_of[n]:<8}{totals[n]//10000:>6,}만  "
              f"{sgn}{diff//10000:>5,}만")

    # 체육관별 매트릭스 (단위 만)
    print(f"\n=== 체육관별 잔고 (단위 만) ===")
    short = {g: _short(g) for g in gym_order}
    cols = "".join(f"{short[g]:>9}" for g in gym_order)
    print(f"  {'후보':<12}{'그룹':<8}{cols}{'합':>9}")
    for n in rank:
        cells = "".join(f"{balances[n][g]//10000:>9,}" for g in gym_order)
        print(f"  {n:<12}{label_of[n]:<8}{cells}{totals[n]//10000:>9,}")

    # MD 저장
    md = [
        f"# SPY 리그 — 자산-횡단 검증",
        "",
        f"체육관 기간(QQQ 정의) 그대로, 자산만 **{ASSET}**로 교체.",
        "도전자 10명 (다목적 Top5 + 단일목적 TPE 5시드) · NPC 4명 (B&H/저축왕/성실이/돼지저금통).",
        "시드 100만원 × 6체육관 = 600만원 판돈.",
        "",
        "## 종합 순위 — 6체육관 잔고 합",
        "",
        "| 순위 | 후보 | 그룹 | 잔고 합 (만) | 성실이 차 |",
        "|---:|---|---|---:|---:|",
    ]
    for i, n in enumerate(rank, 1):
        diff = totals[n] - dca_sum
        sgn = "+" if diff >= 0 else ""
        md.append(f"| {i} | {n} | {label_of[n]} | {totals[n]//10000:,} | {sgn}{diff//10000:,} |")
    md += ["", "## 체육관별 잔고 (단위 만)", "",
           "| 후보 | 그룹 | " + " | ".join(short[g] for g in gym_order) + " | 합 |",
           "|---|---|" + "---:|" * (len(gym_order) + 1)]
    for n in rank:
        cells = " | ".join(f"{balances[n][g]//10000:,}" for g in gym_order)
        md.append(f"| {n} | {label_of[n]} | {cells} | **{totals[n]//10000:,}** |")
    md.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nsaved: {OUT_MD.relative_to(_ROOT)} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
