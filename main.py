"""
main.py - 진입점. config.json을 읽어 NSGA-III를 돌린다.

[2026-06-13 정리] 모드 분기 제거. nsga3만 운영한다.
  도감 보기: python -m app.lab.dex
  검증 도구: tools/test_*.py · tools/check_*.py · tools/walk_forward.py
  리그 워크플로우: python app/league/<name>.py
"""
import json
from pathlib import Path

from app.service import run_nsga3

CONFIG_PATH = Path(__file__).parent / "config.json"

# 기본값 — config.json에 없는 키는 이 값을 쓴다.
DEFAULTS = {
    "trials": 600,                # 총 목표 trial 수 — 스터디 재개 시 모자란 만큼만 추가 실행
    "seed": None,                 # 시드 (None=매번 다름)
    "storage": None,              # Optuna storage URL (예: "sqlite:///nsga3.db")
    "study": "nsga3_v2_weights",  # 스터디 이름 (storage 사용 시)
    "tune_params": False,         # True면 시그널 파라미터도 탐색 (v1에서 과적합 — 고도화용)
    "population_size": 50,        # NSGA-III 한 세대 크기
    "early_stop_window": None,    # HV MA(window) 정체 시 self stop (예: 5). None=끔
    "adaptive_mutation": False,   # True면 HV 정체/개선 신호로 mutation_prob 자동 조정
}


def load_config() -> dict:
    """config.json을 읽어 DEFAULTS 위에 덮어쓴 최종 설정을 돌려준다."""
    config = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        # utf-8-sig: 윈도우 편집기가 붙이는 BOM까지 처리 (BOM 없어도 동일 동작)
        config.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig")))
    return config


def main() -> None:
    config = load_config()
    run_nsga3(config["trials"], config["seed"], config["storage"],
              config["study"], config["tune_params"],
              config["population_size"], config["early_stop_window"],
              config["adaptive_mutation"])


if __name__ == "__main__":
    main()
