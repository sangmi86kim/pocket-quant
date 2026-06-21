"""시즌3 리그 (v3) — 스켈레톤(placeholder).

[학교 ↔ 리그 분리]
- exam(실QQQ 6체육관 졸업시험)은 **학교 졸업 산출물**로 분리됐다
  (`app/academy/exam/graduate.py`, 진단 전용). 리그는 더 이상 exam을 채점하지 않는다.
- 시즌3 리그는 실데이터 두 관문만 본다:
    ① OOS 11년       (victory_road) — 1차 선발 잣대
    ② 사천왕 hold-out (elite_four)   — 최종 판정
  battle_frontier(평행세계 200)는 시즌3 리그에서 일단 뺀다.

[잣대] 종료잔고 **중앙값** (median 목적과 정합, [[median-objective-poc]] 노선).

[상태] 아직 스켈레톤이다 — `run()`은 NotImplementedError. 실제 채점은 14신호 + median으로
top30을 **재선발**한 뒤 채운다(현재 top30은 FEAR_NQ 이전 13신호라 호환 안 됨).

실행(예정): .venv/Scripts/python.exe -m app.league.v3.season3_league
"""
from app.league import elite_four as EF       # ② 사천왕 hold-out
from app.league import victory_road as VR     # ① OOS 11년


SEED_KRW = 1_000_000

# 시즌3 리그 관문 — 순서대로. exam(→학교) · world(평행세계) 없음.
STAGES = (
    ("oos", "OOS 11년", "victory_road"),         # ① 1차 선발
    ("holdout", "사천왕 hold-out", "elite_four"),  # ② 최종 판정
)

# 학교 졸업생(top30) 경로 — 14신호 + median 재선발 후 채운다. None = 아직 미정.
TOP30_PATH = None


def run() -> dict:
    """시즌3 리그 본경기 — OOS → 사천왕. (placeholder)

    TODO(top30 재선발 후 구현):
      1) TOP30_PATH 졸업생 로드 (14신호 가중치 params)
      2) ① OOS    — VR.OOS_YEARS 종료잔고 중앙값으로 줄세워 1차 선발
      3) ② 사천왕 — 선발 통과자를 EF.ROUNDS 종료잔고 중앙값으로 최종 판정
      4) 결과·박스플랏을 app/league/results/ 에 저장
    """
    raise NotImplementedError("시즌3 리그는 스켈레톤 — top30 재선발 후 구현")


def main() -> None:
    print("season3 league (v3) — placeholder", flush=True)
    print(f"  잣대: 종료잔고 median · 시드 {SEED_KRW:,}원", flush=True)
    for i, (key, title, src) in enumerate(STAGES, start=1):
        print(f"  관문 {i} [{key}]: {title}  <- app/league/{src}.py", flush=True)
    print(f"  OOS 연도 {len(VR.OOS_YEARS)}개 · 사천왕 라운드 {len(EF.ROUNDS)}개", flush=True)
    print("  run(): NotImplementedError (스켈레톤)", flush=True)


if __name__ == "__main__":
    main()
