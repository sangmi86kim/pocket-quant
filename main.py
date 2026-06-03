"""
main.py - 프로그램의 시작점(진입점). CLI 입력만 받는다.

3층 구조:
  main.py    = CLI 입력만 받음 (← 이 파일: argparse로 인자 받아 service에 넘김)
  service.py = 단판/진화 '실행 순서'를 조립
  backend/*  = 실제 기능(데이터·전략·백테스트·GA·계산)

여기서는 계산도 흐름 조립도 하지 않는다. 입력을 받아 어떤 서비스를 부를지만 정한다.
"""
import argparse

from app.service import run_evolve, run_pokedex, run_single


def main() -> None:
    parser = argparse.ArgumentParser(description="PocketQuant - 전략 포켓몬 스탯 백테스트")
    parser.add_argument("-g", "--genes", type=int, default=None,
                        help="[단판] 유전자 개수 (생략 시 랜덤)")
    parser.add_argument("--evolve", action="store_true",
                        help="진화 모드(단일목적 GA) 실행")
    parser.add_argument("--dex", action="store_true",
                        help="포켓몬 도감(유전자 설명) 출력")
    parser.add_argument("--pop", type=int, default=20, help="[진화] 개체군 크기")
    parser.add_argument("--generations", type=int, default=10, help="[진화] 세대 수")
    parser.add_argument("--seed", type=int, default=None,
                        help="랜덤 시드 고정 (GA 재현용)")
    args = parser.parse_args()

    # 입력을 받아 해당 서비스(실행 흐름)에 넘기기만 한다.
    if args.dex:
        run_pokedex()
    elif args.evolve:
        run_evolve(args.pop, args.generations, args.seed)
    else:
        run_single(args.genes, args.seed)


if __name__ == "__main__":
    main()
