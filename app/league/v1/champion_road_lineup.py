"""챔피언로드 ① v1 시즌 풀라인업 — 현 챔피언 + Top10 + NPC 4인방 입장.

[흐름]
1. top10_champions.json 읽어 가중치 후보 11명(현 챔피언 + Top10) 변환
2. NPC 4인방(어플삭제맨·저축왕·돼지저금통·성실이)도 정식 선수로 명단 추가
3. `victory_road.run_gate1(graduates)` 호출 — graduate dict의 evaluator 키 분기로
   NPC도 같은 매트릭스에 출전(연도별 1등 카운트·국면 라벨 표에 자연 포함).

[load_graduates_from_study]
NSGA-III sqlite 스터디 직접 로드 + summarize_front로 졸업생 명단 만드는 함수는
원래 victory_road.py에 있었으나 학교(`academy.training.nsga3`) 후처리를 시즌 코어가
부르는 의존 정리를 위해 시즌 어댑터로 이주(2026-06-14). battle_frontier.py도 이 경로를
사용한다.

실행: python app/league/v1/champion_road_lineup.py
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")    # type: ignore[union-attr]
    except Exception:
        pass

import optuna

from app.academy.exam.grade import decode_params
from app.academy.training import nsga3
from app.league.operations.npcs import npc_graduates
from app.league.victory_road import run_gate1
from app.pocket.signals import ALL_GENES

TOP10_JSON = _ROOT / "reports" / "league_v1" / "top10_champions.json"

# v1 시즌 NSGA-III sqlite 위치 (옛 victory_road.STORAGE/STUDY 이주)
STORAGE = f"sqlite:///{(_ROOT / 'optuna_pocketquant.db').as_posix()}"
STUDY = "nsga3_v2_weights"      # v1 가중치+파라미터 스터디는 관문 ①에서 전멸 — DB 보존
# 졸업(선발) 필터 허용치. v1의 -5는 파라미터 튜닝으로 부풀린 인샘플 점수에서만
# 가능했던 기준. 정직한 가중치 전용 공간(v2)에서 -10이 현실적인 선발선이다.
GRAD_TOLERANCE = 0.10


def load_graduates_from_study() -> list[dict]:
    """v1 NSGA-III sqlite 스터디 → 챔피언로드 입장 명단(가중치 후보만).

    [스페셜리스트 트랙 — 다양성 보장]
    필터(최악 국면 ≥ -10)는 올라운더 선발 기준이라, "한 국면 몰빵형"(예: bear 1등인데
    상승장 낙제)은 검증장에 입장 못 했다. 하지만 걔들이 Regime Scanner 30% 틸트의
    후보군이므로, front에서 목적별 1등 5명을 필터 무시하고 입장시킨다.
    ⚠️ 단 관문 ①의 생존 기준은 올라운더용(평시 OOS 평균>0)이라 스페셜리스트에겐
    참고 기록일 뿐 — 본판정은 관문 ②(배틀 프론티어)의 전문 국면 합성 세계에서.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=STUDY, storage=STORAGE)
    summary = nsga3.summarize_front(study, tolerance=GRAD_TOLERANCE)
    label_of = {row["number"]: name for name, row in summary["labels"].items()}

    graduates: list[dict] = [{
        "name": "현챔피언(동일가중)", "label": "기준",
        "weights": [1.0 if g in ("VOL", "REV_RSI", "REV_BB") else 0.0 for g in ALL_GENES],
        "params": {}, "mean5": None, "specialist": False,
    }]
    for r in sorted(summary["passed"], key=lambda r: -r["mean5"]):
        w, sig = decode_params(r["params"])
        graduates.append({
            "name": f"#{r['number']}", "label": label_of.get(r["number"], ""),
            "weights": w, "params": sig, "mean5": r["mean5"], "specialist": False,
        })

    # 목적별 1등 스페셜리스트 (front 전체에서, 필터 무시. 이미 명단에 있으면 생략)
    # ⚠️ score 목적 5개만 — turnover 최저는 스페셜리스트가 아니다. turnover를
    # minimize하는 목적이 있어 front의 그 극단은 거의 확실히 "가중치≈0 = 돼지저금통"이고,
    # 필터 무시 입장은 퇴화 후보를 관문에 되올리는 뒷문이 된다(코덱스 리뷰 P2, 06-11).
    # Low-turnover는 summarize_front의 필터 통과자 안에서 라벨로만 뽑는다.
    seen = {g["name"] for g in graduates}
    front = [{"number": t.number, "values": list(t.values), "params": dict(t.params)}
             for t in study.best_trials]
    spec_picks = [(f"{nsga3.OBJECTIVE_NAMES[i]} 1위",
                   max(front, key=lambda r: r["values"][i])) for i in range(5)]
    for title, r in spec_picks:
        if f"#{r['number']}" in seen:
            continue
        seen.add(f"#{r['number']}")
        w, sig = decode_params(r["params"])
        graduates.append({
            "name": f"#{r['number']}", "label": f"★{title}",
            "weights": w, "params": sig,
            "mean5": sum(r["values"][:5]) / 5, "specialist": True,
        })
    return graduates


def main() -> None:
    top10 = json.loads(TOP10_JSON.read_text(encoding="utf-8"))

    graduates: list[dict] = [{
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

    # NPC 4인방 정식 선수로 입장 (사용자 안 2026-06-14)
    graduates.extend(npc_graduates())

    print(f"=== 챔피언로드 ① 입장 명단: {len(graduates)}명 (가중치 후보 + NPC) ===")
    print(f"  현챔피언 (동일가중 VOL+REV_RSI+REV_BB)")
    for g in graduates[1:]:
        suffix = "" if "evaluator" not in g else "  🤖 NPC"
        mean5 = f"인샘플 mean5 {g['mean5'] * 100:+.1f}" if g['mean5'] is not None else ""
        print(f"  {g['name']:<14} {g['label']:<14} {mean5}{suffix}")
    print()

    run_gate1(graduates)


if __name__ == "__main__":
    main()
