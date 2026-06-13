"""챔피언로드 ① 시험장에 top10 + 현 챔피언 + 성실이 입장 (06-13).

흐름: top10_champions.json 읽고 graduates 포맷 변환 → victory_road.run_gate1 호출.
성실이는 victory_road 내부에서 자동으로 1등 후보 매트릭스에 들어감.

실행: python tools/champion_road_lineup.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.backend.genes.signals import ALL_GENES
from app.league.victory_road import run_gate1

TOP10_JSON = _ROOT / "reports" / "league_v1" / "top10_champions.json"


def main() -> None:
    top10 = json.loads(TOP10_JSON.read_text(encoding="utf-8"))

    graduates = [{
        "name": "현챔피언", "label": "기준(동일가중)",
        "weights": [1.0 if g in ("VOL", "REV_RSI", "REV_BB") else 0.0
                     for g in ALL_GENES],
        "params": {}, "mean5": None, "specialist": False,
    }]
    for i, t in enumerate(top10, 1):
        weights = [t["params"][f"w_{g}"] for g in ALL_GENES]
        mean5 = sum(t["values"][:5]) / 5
        graduates.append({
            "name": f"TOP{i:02d}",
            "label": f"#{t['trial_number']}(s{t['seed']})",
            "weights": weights, "params": {},   # 시그널 파라미터 = 기본값(가중치 전용 v2)
            "mean5": mean5, "specialist": False,
        })

    print(f"=== 챔피언로드 ① 입장 명단: {len(graduates)}명 + 성실이 ===")
    print(f"  현챔피언 (동일가중 VOL+REV_RSI+REV_BB)")
    for g in graduates[1:]:
        print(f"  {g['name']} {g['label']} — 인샘플 mean5 {g['mean5'] * 100:+.1f}")
    print()

    run_gate1(graduates)


if __name__ == "__main__":
    main()
