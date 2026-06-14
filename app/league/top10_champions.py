"""sweep_seeds.json에서 잔고 합 상위 10명 챔피언 추출 (06-13).

사용자 안: "챔피언 한 놈만 하란 법 있어? 10명 돌려봐" — 라인업으로 본다.
별도 실행 없이 기존 sweep 결과만 재처리.

산출: reports/league_v1/top10_champions.md + .json
실행: python tools/top10_champions.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8")
    except Exception:
        pass

IN_JSON = Path("reports/league_v1/sweep_seeds.json")
OUT_MD = Path("reports/league_v1/top10_champions.md")
OUT_JSON = OUT_MD.with_suffix(".json")

GENE_ORDER = ["DD", "VOL", "MA", "MOM", "REV_RSI", "REV_BB"]


def normalize_weights(params: dict) -> list[float]:
    """원본 w_* → 정규화(합 1)로 — 비율만 의미."""
    raw = [params[f"w_{g}"] for g in GENE_ORDER]
    s = sum(raw) or 1.0
    return [w / s for w in raw]


def gene_label(norm_w: list[float], threshold: float = 0.10) -> str:
    """주력 유전자만 골라 라벨링 (10% 이상)."""
    pairs = sorted(zip(GENE_ORDER, norm_w), key=lambda kv: -kv[1])
    main = [g for g, w in pairs if w >= threshold]
    return "+".join(main) if main else "분산형"


def main() -> None:
    data = json.loads(IN_JSON.read_text(encoding="utf-8"))

    # 25명 풀 만들기 (5 시드 × top5)
    pool = []
    for r in data:
        for t in r["top5"]:
            pool.append({**t, "seed": r["seed"]})
    pool.sort(key=lambda t: -t["합"])

    # 같은 trial이 시드별로 중복될 일은 없지만 안전하게 시드+trial 기준 dedupe
    seen = set()
    top10 = []
    for t in pool:
        key = (t["seed"], t["trial_number"])
        if key in seen:
            continue
        seen.add(key)
        top10.append(t)
        if len(top10) >= 10:
            break

    # ── md 리포트 ──
    md = ["# 새 챔피언 라인업 — 잔고 합 Top 10 (06-13)", ""]
    md.append("- 출처: 5 시드 분산 실행 25명 풀 (sweep_seeds.json)")
    md.append("- 정렬: 6체육관 잔고 합 (100만원 시드 × 6, 단위 만원)")
    md.append("- 라벨: 정규화 가중치 10% 이상 유전자만 표기 (분산형 = 다 미만)")
    md.append("")

    md.append("## 순위표")
    md.append("")
    md.append("| 순위 | 시드 | trial | 합 | 성실이 | 차 | 주력 유전자 |")
    md.append("|---:|---:|---|---:|---:|---:|---|")
    for i, t in enumerate(top10, 1):
        norm = normalize_weights(t["params"])
        label = gene_label(norm)
        md.append(f"| {i} | {t['seed']} | #{t['trial_number']} | "
                  f"{t['합'] // 10000:,} | {t['성실이_합'] // 10000:,} | "
                  f"+{t['차'] // 10000:,} | {label} |")
    md.append("")

    # 체육관별 잔고 매트릭스
    gym_order = list(top10[0]["per_gym"].keys())
    nick_map = [("닷컴", "닷컴"), ("금융위기", "금융위기"),
                ("회복", "회복"), ("코로나", "코로나"),
                ("상승", "상승"), ("횡보", "횡보")]
    nick = {g: next(short for tok, short in nick_map if tok[:2] in g)
            for g in gym_order}

    md.append("## 체육관별 잔고 (단위 만원)")
    md.append("")
    md.append("| 순위 | trial | " + " | ".join(nick[g] for g in gym_order) + " |")
    md.append("|---:|---|" + "---:|" * len(gym_order))
    for i, t in enumerate(top10, 1):
        cells = " | ".join(f"{t['per_gym'][g] // 10000:,}" for g in gym_order)
        md.append(f"| {i} | #{t['trial_number']} | {cells} |")
    # 성실이 비교선
    sample_dca = top10[0]["성실이_합"] // 10000
    # 체육관별 성실이 잔고는 sweep_seeds.json엔 후보 dca 미저장 → top10[0] 합만 표기
    md.append("")
    md.append(f"> 성실이 합 {sample_dca:,}만원 (모든 시드 동일 — 결정론적 DCA).")
    md.append("")

    # 가중치 매트릭스
    md.append("## 정규화 가중치 (합=1, 비율만 의미)")
    md.append("")
    md.append("| 순위 | trial | " + " | ".join(GENE_ORDER) + " |")
    md.append("|---:|---|" + "---:|" * len(GENE_ORDER))
    for i, t in enumerate(top10, 1):
        norm = normalize_weights(t["params"])
        cells = " | ".join(f"{w:.2f}" for w in norm)
        md.append(f"| {i} | #{t['trial_number']} | {cells} |")
    md.append("")

    # 현 챔피언 비교
    md.append("## 현 챔피언과 비교")
    md.append("")
    md.append("- 현 챔피언: 동일가중 VOL + REV_RSI + REV_BB (각 0.33, 나머지 0)")
    md.append("- 새 라인업 패턴 보기 — 분산도/주력 유전자 변화")
    md.append("")

    # 패턴 카운트
    label_counts = {}
    for t in top10:
        norm = normalize_weights(t["params"])
        label = gene_label(norm)
        label_counts[label] = label_counts.get(label, 0) + 1
    md.append("**주력 유전자 패턴 (top10 카운트):**")
    for label, cnt in sorted(label_counts.items(), key=lambda kv: -kv[1]):
        md.append(f"- {label}: {cnt}명")
    md.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(top10, ensure_ascii=False, indent=2,
                                     default=str), encoding="utf-8")
    print(f"saved: {OUT_MD} (+ .json)")
    print(f"top10 잔고 합: {[t['합'] // 10000 for t in top10]} 만원")


if __name__ == "__main__":
    main()
