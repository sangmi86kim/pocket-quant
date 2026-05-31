"""
main.py - 프로그램의 시작점(진입점), 게임 진행자 역할

여기서는 어려운 계산을 하지 않습니다.
다른 파일에 있는 기능들을 '순서대로 불러서' 화면에 출력만 합니다.

  1) create_strategy() : 전략 포켓몬 생성   (strategy.py)
  2) challenge()        : 체육관들에 도전    (battle.py)
  3) print(...)         : 결과 출력
"""
import argparse   # 명령줄 옵션(-g 같은 것)을 처리해주는 표준 라이브러리

# app/backend 폴더 안의 기능들을 가져온다.
# main.py는 프로젝트 '루트'에서 실행되므로 'app.backend....' 전체 경로로 가져옴.
from app.backend.battle import challenge
from app.backend.gym import all_gyms
from app.backend.strategy import create_strategy


def run(gene_count: int | None) -> None:
    """실제 게임 한 판을 진행하고 결과를 출력하는 함수."""
    print("=== PocketQuant ===\n")

    # ── 1단계: 전략(포켓몬) 생성 ──
    print("전략 생성")
    strategy = create_strategy(gene_count)
    # " + ".join(...) : 유전자들을 ' + '로 이어 출력. 예) "DD + RSI + MA"
    print(" + ".join(strategy.genes))
    # f"..." : 문자열 안에 {변수}를 끼워넣는 문법(f-string)
    print(f"이름: {strategy.name}\n")

    # ── 2단계: 체육관 도전 ──
    print("체육관 도전")
    report = challenge(strategy, all_gyms())   # 전 체육관 도전 -> 성적표 받기
    for r in report.results:                   # 결과를 하나씩 출력
        print(f"\n{r.gym_name}")
        # 삼항식: 조건이 참이면 앞, 거짓이면 뒤. survived가 True면 "생존"
        print("생존" if r.survived else "사망")

    # ── 3단계: 최종 결과 요약 출력 ──
    print("\n결과")
    print(f"생존 {report.survive_count}")   # @property라 괄호 없이 꺼내 씀
    print(f"사망 {report.death_count}")
    print(f"등급 {report.grade}")


def main() -> None:
    """명령줄 옵션을 읽어서 run()을 호출하는 '입구' 함수."""
    parser = argparse.ArgumentParser(description="PocketQuant - 전략 포켓몬 생존 테스트")
    # -g / --genes 옵션: 유전자 개수를 직접 정하고 싶을 때 사용
    # 예) python main.py -g 3   ->  args.genes 에 3이 들어옴
    # 안 주면 default=None -> create_strategy가 개수를 랜덤으로 정함
    parser.add_argument(
        "-g", "--genes", type=int, default=None,
        help="유전자 개수 (생략 시 랜덤)",
    )
    args = parser.parse_args()   # 실제로 입력된 옵션을 해석
    run(args.genes)              # 해석한 값으로 게임 실행


# 이 파일을 'python main.py'로 직접 실행했을 때만 main()을 부른다.
# (다른 파일이 이 파일을 import만 할 때는 자동 실행되지 않게 하는 관용구)
if __name__ == "__main__":
    main()
