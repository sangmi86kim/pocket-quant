"""실험실 — LLM NPC(오박사) + 사후 리포팅(labnotes).

폴더 규칙(2026-06-13 재구성):
- app/lab/   : 오박사 LLM 어댑터 + labnotes 작성. LM Studio 부재 시 조용히 부재 처리.
- app/league/: 리그 워크플로우(NSGA-III + 챔피언로드 + 라인업 + 분석)
- app/       : 모드 디스패처(service.py) + backend
- tools/     : validator + 진단 (test_*, check_*, walk_forward)
"""
