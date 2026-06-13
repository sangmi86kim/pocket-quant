"""데이터 입출력 — 외부 입력 + 분석/리포트 출력.

[2026-06-13 신설] app/league/에서 리포팅 모듈을 분리해 가져왔다.
구조:
- 출력(현재): inspect_front · report_nsga3 · battle_frontier_total
  (스터디 sqlite → CSV/HTML/MD 분석 산출)
- 입력(예정): kis_client.py — KIS API 래퍼. 외인기관/호가 잔량/체결 강도/거래량 급증 등
  가격 외 새 시그널의 원천. 들어오면 신호는 backend/genes/signals.py로 흘러간다.

리포팅 모듈은 워크플로우(`app/league/`)가 만든 산출물(.sqlite/.json)을 사람용
포맷으로 다시 가공만 한다 — 백테스트 계산이나 적합도 평가는 하지 않는다.
"""
