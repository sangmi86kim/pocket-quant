"""배틀 프론티어 토탈 수익 분석 — 매 세계마다 100만원 판돈, 누적 합 비교 (v1 시즌, 06-13).

기존 battle_frontier_lineup.json 재처리만 (재실행 X).
배틀별 점수가 아니라 "토탈 수익" — 400세계 누적 합 + arena별 분해.

[lab 이주 — 2026-06-14]
시즌 결과 사후 분석은 시즌 어댑터가 아니라 연구·진단(lab) 책임.
v1 시즌 마감 산출물 분석이라 prefix `v1_`로 유지.

산출: reports/league_v1/battle_frontier_total.md (+ .json)
실행: python -m app.lab.season.v1_battle_frontier_total
"""

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]   # app/lab/season/ → 프로젝트 루트

IN_JSON = _ROOT / "reports" / "league_v1" / "battle_frontier_lineup.json"
OUT_MD = _ROOT / "reports" / "league_v1" / "battle_frontier_total.md"

SEED_KRW_MAN = 100   # 한 세계 시드 100만원 (만원 단위)


def main() -> None:
    data = json.loads(IN_JSON.read_text(encoding="utf-8"))
    arenas = data["arenas"]   # {arena: {name: [세계별 잔고 원]}}

    # 후보별 arena별 합 + 총합
    names = list(next(iter(arenas.values())).keys())
    arena_names = list(arenas.keys())

    totals = {}
    for n in names:
        per_arena = {a: sum(arenas[a][n]) for a in arena_names}
        per_arena_n_worlds = {a: len(arenas[a][n]) for a in arena_names}
        seed_sum = sum(per_arena_n_worlds.values()) * 1_000_000
        grand = sum(per_arena.values())
        totals[n] = {"per_arena": per_arena, "grand": grand,
                     "n_worlds": per_arena_n_worlds, "seed_sum": seed_sum}

    md = ["# 배틀 프론티어 — 토탈 수익 (06-13)", ""]
    md.append("- 매 세계마다 100만원 시드 — 종료 잔고 누적 합 비교")
    md.append("- 전천후 200 + bear 100 + rebound 100 = 총 400세계")
    md.append("- **총 판돈 = 400 × 100만원 = 4억원**")
    md.append("")

    md.append("## 종합 — 토탈 잔고 (4억 판돈 → 얼마 회수했나)")
    md.append("")
    md.append("| 순위 | 후보 | 토탈 잔고 (억) | 수익 (만원) | 수익률 |")
    md.append("|---:|---|---:|---:|---:|")
    sorted_names = sorted(names, key=lambda n: -totals[n]["grand"])
    for i, n in enumerate(sorted_names, 1):
        grand = totals[n]["grand"]
        seed = totals[n]["seed_sum"]
        profit = grand - seed
        ret = profit / seed * 100
        sign = "+" if profit >= 0 else ""
        md.append(f"| {i} | {n} | {grand / 1e8:.2f} | {sign}{profit // 10000:,} | "
                  f"{sign}{ret:.1f}% |")
    md.append("")

    md.append("## arena별 잔고 누적 (단위 만원)")
    md.append("")
    md.append("| 후보 | 전천후 (200×100=2억) | bear (100×100=1억) | rebound (100×100=1억) | 합 |")
    md.append("|---|---:|---:|---:|---:|")
    for n in sorted_names:
        cells = []
        for a in arena_names:
            v = totals[n]["per_arena"][a] // 10000
            cells.append(f"{v:,}")
        grand = totals[n]["grand"] // 10000
        md.append(f"| {n} | " + " | ".join(cells) + f" | **{grand:,}** |")
    md.append("")

    md.append("## arena별 수익률 (얼마 늘렸나 — 각 arena 판돈 대비)")
    md.append("")
    md.append("| 후보 | 전천후 | bear | rebound | 평균 |")
    md.append("|---|---:|---:|---:|---:|")
    for n in sorted_names:
        rets = []
        for a in arena_names:
            v = totals[n]["per_arena"][a]
            seed = totals[n]["n_worlds"][a] * 1_000_000
            r = (v - seed) / seed * 100
            rets.append(r)
        avg = sum(rets) / len(rets)
        cells_text = " | ".join(f"{'+'  if r >= 0 else ''}{r:.1f}%" for r in rets)
        sign_avg = "+" if avg >= 0 else ""
        md.append(f"| {n} | {cells_text} | **{sign_avg}{avg:.1f}%** |")
    md.append("")

    md.append("> 위기 + 회복 + 평시 다 합쳐서 보면 누가 진짜 챔피언인지 갈림.")
    md.append("> 어플삭제맨이 bear에서 -50% 맞으니 평시·회복에서 번 게 절반 깎임.")
    md.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    OUT_JSON = OUT_MD.with_suffix(".json")
    OUT_JSON.write_text(json.dumps(totals, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"saved: {OUT_MD} (+ .json)")
    print("\n토탈 수익 순위:")
    for i, n in enumerate(sorted_names[:6], 1):
        grand = totals[n]["grand"]
        profit = grand - totals[n]["seed_sum"]
        print(f"  {i}. {n} {grand / 1e8:.2f}억 (수익 {profit // 10000:+,}만원)")


if __name__ == "__main__":
    main()
