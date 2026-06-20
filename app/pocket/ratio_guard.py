"""비율형 시그널 공용 가드.

[왜] 비율 시그널은 num/den 형태인데, 분모가 0이면 ±inf가 나온다. 문제는 inf가
부등호 비교에서 '발동'으로 둔갑한다는 것 — 예: VOL_SPIKE의 `inf > 2.5`는 True라
"배수 정의 불가"인 퇴화일에 매수 의견이 튄다. 시그널 철학상 '정의 불가'는 매수가
아니라 기권(NaN)이 맞다.

[규약] 비율 계산은 항상 finite 이거나 NaN이다 — ±inf는 만들지 않는다. 이 모듈을
거친 비율은 그 규약을 보장한다.

[범위] 여기서 막는 건 '수학적으로 정의 불가'(분모 0 → ±inf)뿐이다. "작지만 0은
아닌 분모"(near-zero)를 기권시킬지는 시그널마다 의미가 달라(거래정지·유동성 고갈
처럼 분모 0 자체가 이벤트일 수도 있음) 전역 기본값으로 박지 않는다 — 필요하면
그 시그널에서 명시적으로 판단한다.
"""
import numpy as np
import pandas as pd


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """num/den을 계산하되 ±inf(분모 0)는 NaN으로 치환해 '정의 불가 → 기권'을 만든다."""
    return (num / den).replace([np.inf, -np.inf], np.nan)
