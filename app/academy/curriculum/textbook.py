"""교과서 (textbook) — 평행세계 1권 만들기.

[책임]
`make_world()`가 호출 1번에 합성 QQQ + 야생 정보원(VIX·금리·달러 등)을 **같은 블록
인덱스로 동시에 잘라** 평행세계 1권을 반환한다. 닷컴 패닉 블록에는 VIX 폭증·금리
하락이 같이 들어감 → 야생 시그널이 합성 세계에서도 정상 작동.

N권을 묶어 학기 코스로 엮는 일은 `curriculum.course`의 책임이다.

[규칙]
- 스트림별 데이터 부재 구간은 NaN 유지 (UUP 2007년 이전 등 — 상장 전 역사 발명 금지)
- `prices.attrs["synthetic"]=True` + `prices.attrs["external_streams"]=dict`로 부착
  → `signals._fetch_external`이 우선 참조
"""
import numpy as np
import pandas as pd

from app.world.data_loader import LoadedGym, get_prices, get_volume
from app.pocket.models import Gym
from app.world.regime import classify_daily


# 학습 자료 범위 — battle_frontier와 동일 (사천왕 봉인 2020-07 이후 사용 금지)
DATA_START = "1999-03-10"
DATA_END = "2020-06-30"

# 한 달 블록 — 변동성 뭉침 보존 (그보다 짧으면 패턴 깨짐, 길면 다양성 부족)
BLOCK_DAYS = 21

# 지표 예열용 선행 구간 (성적엔 안 들어감)
WARMUP_TDAYS = 270

# 평가 구간 = 2년 (학교 한 학기 길이)
EVAL_TDAYS = 504


# 야생 시그널이 쓰는 외부 정보원 묶음 — 복원 방식은 신호가 절대값을 보느냐로 갈린다.
#   "price"  = 가격형 (QQQ/SPY/TLT/UUP/DIA) → log return으로 잘라 cumprod로 복원 (이음매 연속)
#   "level"  = 변동성 지수형 (VIX/VXN) → raw 값 그대로. FEAR가 절대 임계(>30/>47)로 보므로
#              눈금을 보존해야 한다. 경계 점프는 '그날 값만 보는' 절대 임계엔 무해.
#   "yield"  = 금리형 (^TNX) → 블록을 통째로 평행이동해 이어붙임(offset). US10Y는 '60일 평균
#              대비'로 보므로 절대 레벨은 안 중요하고, 경계 절벽이 만들던 가짜 급락 이벤트를
#              없앤다 (lab/textbook_offset_stitch 실험: 가짜 ~176일/권 제거).
#   "volume" = 거래량형 (QQQ_volume) → raw 값 그대로 (VOL_SPIKE용)
DEFAULT_EXTERNAL_STREAMS = (
    ("^VIX", "level"),     # 공포 지수 (S&P 변동성) → raw, 절대 눈금 보존
    ("^VXN", "level"),     # 공포 지수 (나스닥 변동성, FEAR_NQ용) → raw. 2001년~, 이전은 NaN
    ("^TNX", "yield"),     # 10년물 금리 (%) → offset, 경계 절벽 제거
    ("UUP", "price"),      # 달러 ETF (2007년~, 그 이전은 없음 = NaN 유지)
    ("SPY", "price"),      # S&P 500
    ("TLT", "price"),      # 장기채 (2002년~)
    ("DIA", "price"),      # 다우
    ("QQQ_volume", "volume"),  # VOL_SPIKE용 QQQ 거래량
)

REGIME_STATES = ["bull", "bear", "sideways", "volatile"]
REGIME_INDEX = {s: i for i, s in enumerate(REGIME_STATES)}
SYNTH_START = "2001-01-01"


def _to_block_unit(series: pd.Series, stream_type: str) -> pd.Series:
    """타입별 셔플 단위로 변환.

    price → log return (어제 대비 변화율의 로그). cumprod로 복원 시 자연스러움.
    level/volume → raw 값 그대로. VIX 30, 거래량 절대 규모 의미가 유지됨.
    yield → raw 값. 단 결측은 메운다 — 복원이 블록 평행이동(연속)이라 빈칸이 있으면 끊긴다.
    """
    if stream_type == "price":
        return (series / series.shift(1)).apply(np.log).dropna()
    if stream_type in ("level", "volume"):
        return series.dropna()
    if stream_type == "yield":
        return series.interpolate().dropna()
    raise ValueError(f"unknown stream type: {stream_type!r}")


def _offset_stitch(values: np.ndarray) -> np.ndarray:
    """블록을 통째로 위아래로 밀어 앞 블록 끝에 이어붙인다 (yield 복원 = 경계 절벽 제거).

    블록 내부 일별 등락(트렌드)은 그대로, 블록 b 전체에 (앞 블록 끝값 - b 첫값)을 더해
    이음매를 연속으로 만든다. 평균회귀 앵커는 걸지 않는다 — US10Y는 '60일 평균 대비'만 보고,
    금리는 셔플 누적 표류가 작아 밴드를 잘 유지한다(lab/textbook_offset_stitch 실험 확인).
    """
    out = np.array(values, dtype=float)
    if np.isnan(out).any():    # holiday 정렬 NaN만 메움(상장 전 발명 아님 — 금리는 전구간 존재)
        out = pd.Series(out).interpolate().ffill().bfill().to_numpy(copy=True)
    for s in range(BLOCK_DAYS, len(out), BLOCK_DAYS):
        out[s:s + BLOCK_DAYS] += out[s - 1] - out[s]
    return out


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
    if stream_type == "yield":
        return _offset_stitch(values)
    raise ValueError(f"unknown stream type: {stream_type!r}")


def _sample_block_indices(n_days: int, total_days: int, rng) -> np.ndarray:
    """21일 블록 인덱스 추출 — 모든 시계열이 같은 인덱스 공유.

    같은 시점으로 잘라야 cross-asset correlation (여러 자산이 같이 움직이는 정도)
    보존: 닷컴 패닉 블록에 VIX 폭증·금리 하락도 같이 들어감.
    """
    indices: list[int] = []
    total = 0
    while total < n_days:
        i = rng.integers(0, total_days - BLOCK_DAYS + 1)
        indices.extend(range(i, i + BLOCK_DAYS))
        total += BLOCK_DAYS
    return np.array(indices[:n_days])


def _block_pools_and_chain(qqq_raw: pd.Series, common_idx: pd.Index):
    """국면별 진짜 토막 풀과 블록 단위 전이행렬(P), 초기분포(pi)를 추정한다."""
    labels = classify_daily(qqq_raw).reindex(common_idx)
    lab = labels.to_numpy()
    total = len(common_idx)

    pools: dict[str, list[int]] = {s: [] for s in REGIME_STATES}
    for i in range(total - BLOCK_DAYS + 1):
        end = lab[i + BLOCK_DAYS - 1]
        if isinstance(end, str) and end in pools:
            pools[end].append(i)

    seq = np.array([
        REGIME_INDEX[lab[s + BLOCK_DAYS - 1]]
        for s in range(0, total - BLOCK_DAYS + 1, BLOCK_DAYS)
        if isinstance(lab[s + BLOCK_DAYS - 1], str)
        and lab[s + BLOCK_DAYS - 1] in REGIME_INDEX
    ])

    P = np.zeros((len(REGIME_STATES), len(REGIME_STATES)))
    for a, b in zip(seq[:-1], seq[1:]):
        P[a, b] += 1
    for r in range(len(REGIME_STATES)):
        if P[r].sum() == 0:
            P[r] = 1.0
    P = P / P.sum(axis=1, keepdims=True)
    pi = (np.array([np.mean(seq == i) for i in range(len(REGIME_STATES))])
          if len(seq) else np.ones(len(REGIME_STATES)) / len(REGIME_STATES))
    pi = pi / pi.sum()
    return pools, P, pi


def _load_external(streams_spec, start: str, end: str,
                   common_idx: pd.Index) -> dict:
    """야생 정보원 시계열을 받아서 같은 거래일 인덱스로 정렬.

    UUP는 2007년부터, TLT는 2002년부터 — 그 이전 날짜는 NaN으로 유지
    없는 역사는 만들지 않는다. NaN은 그대로 NaN으로 둔다.
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


def _synth_dates(n_days: int) -> pd.DatetimeIndex:
    return pd.bdate_range(SYNTH_START, periods=n_days)


def _build_external_synth(ticker: str, prices: pd.Series, synth_dates: pd.DatetimeIndex,
                          aligned: dict, sample_indices: np.ndarray) -> dict:
    """attrs에 붙일 합성 외부 스트림 묶음."""
    external_synth = {ticker: prices.copy()}
    for ext_ticker, (block_unit, stream_type) in aligned.items():
        values = np.asarray(block_unit.values)[sample_indices]
        synth = _from_block_unit(values, stream_type)
        external_synth[ext_ticker] = pd.Series(synth, index=synth_dates)
    return external_synth


def _loaded_world(name: str, ticker: str, qqq_returns: pd.Series, aligned: dict,
                  sample_indices: np.ndarray, n_days: int) -> LoadedGym:
    """샘플링된 블록 인덱스를 LoadedGym 1권으로 조립한다."""
    qqq_synth_rets = np.asarray(qqq_returns.values)[sample_indices]
    qqq_synth_prices = _from_block_unit(qqq_synth_rets, "price", init=100.0)
    synth_dates = _synth_dates(n_days)
    prices = pd.Series(qqq_synth_prices, index=synth_dates, name=ticker)

    prices.attrs["synthetic"] = True
    prices.attrs["external_streams"] = _build_external_synth(
        ticker, prices, synth_dates, aligned, sample_indices)

    eval_start = prices.index[WARMUP_TDAYS] if n_days > WARMUP_TDAYS else prices.index[0]
    gym = Gym(name, difficulty=0, volatility=0, ticker="SYNTH",
              start=eval_start.strftime("%Y-%m-%d"),
              end=prices.index[-1].strftime("%Y-%m-%d"))
    return LoadedGym(gym=gym, prices=prices)


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
        포함 키: QQQ, ^VIX, ^VXN, ^TNX, UUP, SPY, TLT, DIA, QQQ_volume
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

    # 4~6. QQQ와 야생 정보원을 같은 블록 순서로 조립하고 attrs에 부착.
    return _loaded_world("세계공장#합성", ticker, qqq_returns, aligned,
                         sample_indices, n_days)


def make_world_rs(seed: int = 42,
                  external_streams=DEFAULT_EXTERNAL_STREAMS,
                  start: str = DATA_START, end: str = DATA_END,
                  ticker: str = "QQQ",
                  n_days: int | None = None,
                  skew: dict | None = None) -> LoadedGym:
    """RS 교과서 1권 — 국면 순서를 먼저 뽑고, 그 국면의 진짜 21일 토막을 끼운다.

    skew: {국면: 배수}. 예: {"bear": 3.0}이면 하락장 토막 출제 확률을 높인다.
    """
    if n_days is None:
        n_days = WARMUP_TDAYS + EVAL_TDAYS

    qqq_raw = get_prices(ticker, start, end)
    qqq_returns = (qqq_raw / qqq_raw.shift(1)).apply(np.log).dropna()
    common_idx = qqq_returns.index
    aligned = _load_external(external_streams, start, end, common_idx)
    pools, P, pi = _block_pools_and_chain(qqq_raw, common_idx)

    if skew:
        m = np.array([skew.get(s, 1.0) for s in REGIME_STATES])
        pi = pi * m
        pi = pi / pi.sum()
        P = P * m[None, :]
        P = P / P.sum(axis=1, keepdims=True)

    rng = np.random.default_rng(seed)
    n_blocks = (n_days + BLOCK_DAYS - 1) // BLOCK_DAYS
    state = int(rng.choice(len(REGIME_STATES), p=pi))
    indices: list[int] = []
    for _ in range(n_blocks):
        pool = pools[REGIME_STATES[state]]
        i0 = (int(pool[rng.integers(0, len(pool))]) if pool
              else int(rng.integers(0, len(common_idx) - BLOCK_DAYS + 1)))
        indices.extend(range(i0, i0 + BLOCK_DAYS))
        state = int(rng.choice(len(REGIME_STATES), p=P[state]))
    sample_indices = np.array(indices[:n_days])

    return _loaded_world("세계공장#RS합성", ticker, qqq_returns, aligned,
                         sample_indices, n_days)
