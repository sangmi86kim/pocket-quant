"""
check_signals.py - 시그널 풀 진단 (노출 / 발동률 / 시그널 간 상관)

[왜 상시 도구인가]
2026-06-10 리뷰에서 "시그널 6개처럼 보이지만 실질 자유도 ~2"(DD/MA/VOL/MOM 한
클러스터 + 죽은 RSI/BB)를 임시 스크립트로 발견했다. 시그널을 추가/교체할 때마다
같은 진단을 반복해야 하므로 스크립트로 고정한다.

[측정 항목]
1) 체육관별 노출 — 상시형: 평균 포지션 / 이벤트형(NaN 기권): 발동률 + 발동일 평균 의견
2) 시그널 간 상관 — 이벤트형은 '발동 여부(0/1)'로 변환해 비교
   (발동일 값이 전부 1.0이라 원값으론 상관 계산이 불가능)

[해석 가이드]
- 상시형끼리 상관 > 0.6 = 같은 타입 중복 의심
- 이벤트형 발동이 방어형 포지션과 음의 상관 = 정상 (방어가 도망갈 때 역발상이 출전)
  경계할 것은 '강한 양의 상관'(기존 시그널과 같은 정보 = 추가 가치 없음)

실행: 프로젝트 루트에서  python tests/check_signals.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app.academy.exam import all_gyms
from app.pocket.signals import ALL_GENES, GENE_SIGNALS
from app.world.data import load_gyms


def is_event_signal(series: pd.Series) -> bool:
    """기권(NaN)을 쓰는 이벤트형 시그널인지 판별."""
    return bool(series.isna().any())


def run_check() -> bool:
    loaded = load_gyms(all_gyms())

    frames = []          # 상관 계산용 (이벤트형은 발동 여부로 변환)
    print("=== 1. 체육관별 노출/발동률 ===")
    header = "  ".join(f"{g:>8}" for g in ALL_GENES)
    print(f"{'체육관':<16} {header}")
    for lg in loaded:
        gym, prices = lg.gym, lg.prices
        mask = (prices.index >= pd.Timestamp(gym.start)) & (prices.index <= pd.Timestamp(gym.end))
        cells, corr_cols = [], {}
        for g in ALL_GENES:
            sig = GENE_SIGNALS[g](prices)[mask]
            if is_event_signal(sig):
                fire_rate = sig.notna().mean()
                cells.append(f"{fire_rate:>7.0%}*")           # * = 발동률
                corr_cols[g] = sig.notna().astype(float)      # 발동 여부 0/1
            else:
                cells.append(f"{sig.mean():>8.2f}")           # 평균 포지션
                corr_cols[g] = sig
        frames.append(pd.DataFrame(corr_cols))
        print(f"{gym.name:<16} {'  '.join(cells)}")
    print("  (* = 이벤트형 발동률. 나머지는 평균 포지션 0~1)")

    alldf = pd.concat(frames)
    print(f"\n=== 2. 시그널 간 상관 (전 기간 {len(alldf)}일, 이벤트형은 발동 여부 기준) ===")
    print(alldf.corr().round(2).to_string())

    # 간단 판정: 이벤트형이 기존 상시형과 강한 '양의' 상관이면 경고
    print("\n=== 3. 판정 ===")
    corr = alldf.corr()
    event_genes = [g for g in ALL_GENES
                   if is_event_signal(GENE_SIGNALS[g](loaded[0].prices))]
    ok = True
    for eg in event_genes:
        others = [g for g in ALL_GENES if g not in event_genes]
        worst_g = max(others, key=lambda g: corr.loc[eg, g])
        worst = corr.loc[eg, worst_g]
        verdict = "PASS" if worst < 0.3 else "WARN"
        if worst >= 0.3:
            ok = False
        print(f"  {eg:<8} 상시형과 최대 양의 상관 = {worst:+.2f} ({worst_g})  {verdict}")
    print("  이벤트형이 새 정보를 가져옴" if ok else "  경고: 기존 시그널과 중복 가능성")
    return ok


if __name__ == "__main__":
    # WARN(상시-이벤트 상관 ≥ 0.3)은 진단 정보지 게이트 실패가 아니다.
    # 풀 확장 시 일부 시그널 클러스터링(예: QQQ_SPY ↔ QQQ_DIA)은 자연스러우며
    # GA/TPE가 가중치로 자동 조정. e2e에선 통과, 사람이 콘솔로 판단.
    run_check()
    sys.exit(0)
