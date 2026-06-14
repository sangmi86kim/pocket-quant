"""리그(챔피언로드) 워크플로우 — 챔피언 발굴 + 관문 ①②③ + 풀라인업 + 스터디 분석.

폴더 규칙(2026-06-13 재구성):
- app/league/ : 리그(NSGA-III 탐색 + 챔피언로드 ① ② ③ + 라인업 + 스터디 분석)
- app/        : 실행 흐름·LLM NPC·도감 등 일반 모듈
- tools/      : 코드/로직 검증용 validator + 진단 + 스모크 (test_*, check_*, smoke_*)
- reports/    : 리그 산출물(.md/.json/.html) + league/ 하위
"""
