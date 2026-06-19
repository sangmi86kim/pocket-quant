"""교과서 (textbook) — 평행세계 1권 만들기.

[책임]
`make_world()`가 호출 1번에 합성 QQQ + 야생 정보원(VIX·금리·달러 등)을 **같은 블록
인덱스로 동시에 잘라** 평행세계 1권을 반환한다. 닷컴 패닉 블록에는 VIX 폭증·금리
하락이 같이 들어감 → 야생 시그널이 합성 세계에서도 정상 작동.

N권을 묶어 학기 코스로 엮는 일은 패키지 상위(`curriculum.__init__`)의 책임이다.

[규칙]
- 스트림별 데이터 부재 구간은 NaN 유지 (UUP 2007년 이전 등 — 상장 전 역사 발명 금지)
- `prices.attrs["synthetic"]=True` + `prices.attrs["external_streams"]=dict`로 부착
  → `signals._fetch_external`이 우선 참조
"""
import numpy as np
import pandas as pd

from app.world.data_loader import LoadedGym, get_prices, get_volume
from app.pocket.models import Gym


# 학습 자료 범위 — battle_frontier와 동일 (사천왕 봉인 2020-07 이후 사용 금지)
DATA_START = "1999-03-10"
DATA_END = "2020-06-30"

# 한 달 블록 — 변동성 뭉침 보존 (그보다 짧으면 패턴 깨짐, 길면 다양성 부족)
BLOCK_DAYS = 21

# 지표 예열용 선행 구간 (성적엔 안 들어감)
WARMUP_TDAYS = 270

# 평가 구간 = 2년 (학교 한 학기 길이)
EVAL_TDAYS = 504


# 야생 시그널이 쓰는 외부 정보원 묶음
#   "price"  = 가격형 (QQQ/SPY/TLT/UUP/DIA) → log return으로 잘라 cumprod로 복원
#   "level"  = 절대값형 (VIX/^TNX/거래량) → raw 값 그대로 잘라 raw로 복원
#              ⚠️ 코덱스 연구원 권장: "raw 그대로면 블록 경계 점프 위험 (예: 평온기 VIX 12
#              다음에 위기 VIX 60 갑자기 붙음). 다음 단계에서 log-delta + 클립으로 정교화."
DEFAULT_EXTERNAL_STREAMS = (
    ("^VIX", "level"),     # 공포 지수
    ("^TNX", "level"),     # 10년물 금리 (%)
    ("UUP", "price"),      # 달러 ETF (2007년~, 그 이전은 없음 = NaN 유지)
    ("SPY", "price"),      # S&P 500
    ("TLT", "price"),      # 장기채 (2002년~)
    ("DIA", "price"),      # 다우
    ("QQQ_volume", "volume"),  # VOL_SPIKE용 QQQ 거래량
)


def _to_block_unit(series: pd.Series, stream_type: str) -> pd.Series:
    """타입별 셔플 단위로 변환.

    price → log return (어제 대비 변화율의 로그). cumprod로 복원 시 자연스러움.
    level/volume → raw 값 그대로. VIX 30, 거래량 절대 규모 의미가 유지됨.
    """
    if stream_type == "price":
        return (series / series.shift(1)).apply(np.log).dropna()
    if stream_type in ("level", "volume"):
        return series.dropna()
    raise ValueError(f"unknown stream type: {stream_type!r}")


def _from_block_unit(values: np.ndarray, stream_type: str,
                     init: float = 100.0) -> np.ndarray:
    """블록 단위 → 합성 시계열 복원.

    가격형(price)은 누적 복원하되, 원래 데이터가 없던 자리는 출력도 NaN으로 남긴다.
    UUP/TLT처럼 늦게 생긴 원천의 상장 전 구간을 '변동 없음'으로 발명하지 않기 위해서다.
    """
    if stream_type == "price":
        missing = np.isnan(values)
        path = init * np.exp(np.cumsum(np.where(missing, 0.0, values)))
        path[missing] = np.nan
        return path
    if stream_type in ("level", "volume"):
        # level/volume은 cumsum 안 함 → NaN 전파 없음. 빈칸은 그대로 NaN.
        return np.asarray(values, dtype=float)
    raise ValueError(f"unknown stream type: {stream_type!r}")


def _sample_block_indices(n_days: int, total_days: int, rng) -> np.ndarray:
    """21일 블록 인덱스 추출 — 모든 시계열이 같은 인덱스 공유.

    같은 시점으로 잘라야 cross-asset correlation (여러 자산이 같이 움직이는 정도)
    보존: 닷컴 패닉 블록에 VIX 폭증·금리 하락도 같이 들어감.
    """
    indices = []
    total = 0
    while total < n_days:
        i = rng.integers(0, total_days - BLOCK_DAYS + 1)
        indices.extend(range(i, i + BLOCK_DAYS))
        total += BLOCK_DAYS
    return np.array(indices[:n_days])


def _load_external(streams_spec, start: str, end: str,
                   common_idx: pd.Index) -> dict:
    """야생 정보원 시계열을 받아서 같은 거래일 인덱스로 정렬.

    UUP는 2007년부터, TLT는 2002년부터 — 그 이전 날짜는 NaN으로 유지
    (코덱스 연구원 권장: "없는 역사 만들지 마라, NaN은 그대로 NaN").
    데이터 fetch 자체 실패 시 전체 NaN 시리즈.
    """
    aligned = {}
    for ticker, stream_type in streams_spec:
        try:
            if stream_type == "volume":
                raw = get_volume(ticker.replace("_volume", ""), start, end)
            else:
                raw = get_prices(ticker, start, end)
        except RuntimeError:
            raw = pd.Series(dtype=float)
        if len(raw) == 0:
            aligned[ticker] = (pd.Series(np.nan, index=common_idx), stream_type)
            continue
        block_unit = _to_block_unit(raw, stream_type)
        # 공통 인덱스에 맞춤 — 없던 날짜는 NaN 유지 (백필 금지)
        aligned[ticker] = (block_unit.reindex(common_idx), stream_type)
    return aligned


def make_world(seed: int = 42,
               external_streams=DEFAULT_EXTERNAL_STREAMS,
               start: str = DATA_START, end: str = DATA_END,
               ticker: str = "QQQ",
               n_days: int | None = None) -> LoadedGym:
    """평행세계 1개 만들기 — QQQ + 야생 정보원 같은 블록으로 동시 셔플.

    반환: LoadedGym
      - prices: 합성 QQQ 가격 (100에서 시작)
      - prices.attrs["synthetic"] = True (signals.py 표식)
      - prices.attrs["external_streams"] = {ticker: 합성 시리즈, ...}
        포함 키: QQQ, ^VIX, ^TNX, UUP, SPY, TLT, DIA, QQQ_volume
        → 다음 단계에서 signals._fetch_external이 여기서 읽음

    [흐름]
    1. QQQ + 야생 정보원 다 받기 (시작일 다를 수 있음, 없는 시기는 NaN)
    2. 모두 같은 거래일 인덱스로 정렬
    3. 21일 블록 인덱스 N개 추출 (모든 시계열이 공유)
    4. 그 인덱스로 잘라 이어붙임 (NaN 그대로 보존)
    5. 타입별 복원 (price: cumprod / level·volume: raw)
    6. attrs에 부착해 반환
    """
    if n_days is None:
        n_days = WARMUP_TDAYS + EVAL_TDAYS

    # 1~2. QQQ는 항상 있음 (공통 인덱스의 기준)
    qqq_raw = get_prices(ticker, start, end)
    qqq_returns = (qqq_raw / qqq_raw.shift(1)).apply(np.log).dropna()
    common_idx = qqq_returns.index

    # 야생 정보원 같은 인덱스로 정렬
    aligned = _load_external(external_streams, start, end, common_idx)

    # 3. 같은 블록 인덱스 추출 — 모든 시계열이 이 인덱스 공유
    rng = np.random.default_rng(seed)
    sample_indices = _sample_block_indices(n_days, len(common_idx), rng)

    # 4~5. QQQ 합성가격
    qqq_synth_rets = np.asarray(qqq_returns.values)[sample_indices]
    qqq_synth_prices = _from_block_unit(qqq_synth_rets, "price", init=100.0)
    synth_dates = pd.bdate_range("2001-01-01", periods=n_days)
    prices = pd.Series(qqq_synth_prices, index=synth_dates, name=ticker)

    # 야생 정보원 합성
    external_synth = {ticker: prices.copy()}
    for ext_ticker, (block_unit, stream_type) in aligned.items():
        values = np.asarray(block_unit.values)[sample_indices]
        synth = _from_block_unit(values, stream_type)
        external_synth[ext_ticker] = pd.Series(synth, index=synth_dates)

    # 6. attrs 부착 — 코덱스 연구원 권장: LoadedGym 필드 확장 NO, prices.attrs로만
    prices.attrs["synthetic"] = True
    prices.attrs["external_streams"] = external_synth

    eval_start = prices.index[WARMUP_TDAYS] if n_days > WARMUP_TDAYS else prices.index[0]
    gym = Gym("세계공장#합성", difficulty=0, volatility=0, ticker="SYNTH",
              start=eval_start.strftime("%Y-%m-%d"),
              end=prices.index[-1].strftime("%Y-%m-%d"))
    return LoadedGym(gym=gym, prices=prices)
