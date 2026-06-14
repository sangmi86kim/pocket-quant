"""챔피언로드 ③ 사천왕 — top10 + 챔피언 + 기준선 4인방 풀라인업 (06-13).

hold-out(2020-07~2026-06)은 06-11에 이미 1회 봉인 해제됐다 — 추가 평가는
오염 우려 없음. 단 이 결과를 보고 적합도/가중치/파라미터를 고치면 반칙이고
다음 알파(에그랩)에서 봉인 다시 못 씀 (AGENTS.md 5번).

라운드 7개 (2020 하반기 ~ 2026 상반기) — 매 라운드 100만원 새로:
  - top10 + 현챔피언 = 11명
  - 어플삭제맨 / 저축왕 / 성실이 / 돼지저금통 = 4명
  - 라운드별 1등 + 합산 잔고 + 국면 라벨 (Regime_Scanner)

산출: reports/league_v1/elite_four_lineup.md (+ .json)
실행: python -m app.league.v1.elite_four_lineup
"""

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]

from app.pocket.battle import _score_position, terminal_balance
from app.pocket.signals import ALL_GENES, combine_positions, positions_with_params
from app.world.data_loader import LoadedGym, get_prices
from app.world.regime import REGIME_LABELS, dominant_regime
from app.league.elite_four import DATA_END, HOLDOUT_START, ROUNDS, TICKER, _loaded_window
from app.league.operations.npcs import npc_graduates

SEED_KRW = 1_000_000
TOP10_JSON = _ROOT / "reports" / "league_v1" / "top10_champions.json"
OUT_MD = _ROOT / "reports" / "league_v1" / "elite_four_lineup.md"


def eval_weights(weights: list[float], lw: LoadedGym) -> int:
    pos = combine_positions(positions_with_params(lw.prices), weights)
    return terminal_balance(_score_position(pos, lw), SEED_KRW)


def main() -> None:
    prices = get_prices(TICKER, "1999-03-10", DATA_END)
    rounds = [(name, start, end, _loaded_window(prices, start, end))
              for name, start, end in ROUNDS]

    top10 = json.loads(TOP10_JSON.read_text(encoding="utf-8"))
    champion_w = [1.0 if g in ("VOL", "REV_RSI", "REV_BB") else 0.0 for g in ALL_GENES]
    strategies = [("현챔피언", champion_w)]
    for i, t in enumerate(top10, 1):
        w = [t["params"][f"w_{g}"] for g in ALL_GENES]
        strategies.append((f"TOP{i:02d}", w))
    npcs = npc_graduates()

    balances: dict[str, dict[str, int]] = {}
    for name, w in strategies:
        balances[name] = {nm: eval_weights(w, lw)
                           for nm, _, _, lw in rounds}
    for g in npcs:
        balances[g["name"]] = {nm: g["evaluator"](lw, SEED_KRW)[1]
                               for nm, _, _, lw in rounds}

    # 국면 라벨 (Regime_Scanner)
    regimes = {nm: REGIME_LABELS[dominant_regime(prices, s, e)]
               for nm, s, e, _ in rounds}

    md = ["# 챔피언로드 ③ 사천왕 — 풀라인업 + 기준선 4인방 (06-13)", ""]
    md.append(f"- 봉인 구간: {HOLDOUT_START} ~ {DATA_END}")
    md.append("- 매 라운드 100만원 시드 (단위 만원)")
    md.append("- 국면 라벨: Regime_Scanner")
    md.append("- ⚠️ 결과 보고 적합도/가중치/파라미터 수정 금지 (AGENTS.md 5번)")
    md.append("")

    md.append("## 종합 — 7라운드 잔고 합")
    md.append("")
    md.append("| 순위 | 후보 | 합 | 차(성실이 대비) |")
    md.append("|---:|---|---:|---:|")
    dca_sum = sum(balances["성실이"].values())
    totals = [(n, sum(balances[n].values())) for n in balances]
    totals.sort(key=lambda x: -x[1])
    for i, (n, t) in enumerate(totals, 1):
        diff = t - dca_sum
        sign = "+" if diff >= 0 else ""
        md.append(f"| {i} | {n} | {t // 10000:,} | {sign}{diff // 10000:,} |")
    md.append("")

    md.append("## 라운드별 잔고 (단위 만원)")
    md.append("")
    round_names = [nm for nm, _, _, _ in rounds]
    head = " | ".join(nm for nm in round_names)
    reg_head = " | ".join(regimes[nm][:2] for nm in round_names)
    md.append(f"| 후보 | {head} | 합 |")
    md.append(f"| 국면 | {reg_head} | |")
    md.append("|---|" + "---:|" * (len(round_names) + 1))
    order = (["현챔피언"] + [f"TOP{i:02d}" for i in range(1, 11)]
             + ["어플삭제맨", "저축왕", "성실이", "돼지저금통"])
    for name in order:
        cells = " | ".join(f"{balances[name][nm] // 10000:,}" for nm in round_names)
        total = sum(balances[name].values()) // 10000
        md.append(f"| {name} | {cells} | **{total:,}** |")
    md.append("")

    md.append("## 라운드별 1등 (전체 풀)")
    md.append("")
    md.append("| 라운드 | 국면 | 1등 | 잔고 | 성실이 | 차 |")
    md.append("|---|---|---|---:|---:|---:|")
    win_counts: dict[str, int] = {}
    for nm in round_names:
        winner = max(balances, key=lambda n: balances[n][nm])
        win_counts[winner] = win_counts.get(winner, 0) + 1
        wb = balances[winner][nm]
        db = balances["성실이"][nm]
        diff = wb - db
        sign = "+" if diff >= 0 else ""
        md.append(f"| {nm} | {regimes[nm]} | {winner} | "
                  f"{wb // 10000:,}만 | {db // 10000:,}만 | "
                  f"{sign}{diff // 10000:,}만 |")
    md.append("")

    md.append("**1등 카운트:**")
    for n, c in sorted(win_counts.items(), key=lambda kv: -kv[1]):
        md.append(f"- {n}: {c}회")
    md.append("")

    regime_dist: dict[str, int] = {}
    for r in regimes.values():
        regime_dist[r] = regime_dist.get(r, 0) + 1
    md.append("**사천왕 국면 분포:**")
    for r, c in sorted(regime_dist.items(), key=lambda kv: -kv[1]):
        md.append(f"- {r}: {c}회")
    md.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    OUT_JSON = OUT_MD.with_suffix(".json")
    OUT_JSON.write_text(json.dumps(
        {"balances": {n: dict(b) for n, b in balances.items()},
         "regimes": regimes, "win_counts": win_counts},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {OUT_MD} (+ .json)")
    print("\n종합 순위:")
    for i, (n, t) in enumerate(totals[:6], 1):
        print(f"  {i}. {n} {t // 10000:,}만원")


if __name__ == "__main__":
    main()
