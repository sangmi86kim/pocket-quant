"""
main.py - 프로그램의 시작점(진입점). 실행 옵션은 config.json에서 읽는다.

3층 구조:
  main.py    = 입력만 받음 (← 이 파일: config.json을 읽어 service에 넘김)
  service.py = 단판/진화 '실행 순서'를 조립
  backend/*  = 실제 기능(데이터·전략·백테스트·GA·계산)

CLI 플래그(argparse)는 쓰지 않는다 — 옵션을 바꾸려면 config.json 값을 고치고
다시 `python main.py` 하면 된다. config.json이 없으면 DEFAULTS로 단판 실행.
여기서는 계산도 흐름 조립도 하지 않는다. 설정을 읽어 어떤 서비스를 부를지만 정한다.
"""
import json
from pathlib import Path

from app.service import run_evolve, run_pokedex, run_single

CONFIG_PATH = Path(__file__).parent / "config.json"

# 기본값 — config.json에 없는 키는 이 값을 쓴다.
DEFAULTS = {
    "mode": "single",     # single(단판) | evolve(진화 GA) | dex(도감)
    "genes": None,        # [단판] 유전자 개수 (None = 랜덤)
    "pop": 20,            # [진화] 개체군 크기
    "generations": 10,    # [진화] 세대 수
    "seed": None,         # 랜덤 시드 (None = 매번 다름, 숫자 = 재현 가능)
    "md": None,           # Markdown 리포트: None=저장 안 함, ""=기본 경로, "경로"=지정 경로
    "capital": None,      # 실전 시뮬 시작 자본(원). 예) 10000000
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
    mode = config["mode"]

    # 설정을 읽어 해당 서비스(실행 흐름)에 넘기기만 한다.
    if mode == "dex":
        run_pokedex()
    elif mode == "evolve":
        run_evolve(config["pop"], config["generations"], config["seed"],
                   config["md"], config["capital"])
    elif mode == "single":
        run_single(config["genes"], config["seed"], config["md"], config["capital"])
    else:
        raise SystemExit(f"[config] 알 수 없는 mode: {mode!r} (single | evolve | dex 중 하나)")


if __name__ == "__main__":
    main()
