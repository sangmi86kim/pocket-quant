"""
signals.py - 유전자(시그널)를 '진짜 지표 로직'으로 구현하는 파일

이전엔 유전자가 그냥 점수 라벨이었다 (DD=+20점 식). 이제는 각 유전자가
가격 시계열을 받아 '그날 주식을 얼마나 들고 있을지' = 포지션(0~1) 을 만든다.
  포지션 1.0 = 풀매수,  0.0 = 전액 현금,  0.5 = 반반

[유전자 매핑 — 3타입 × 2마리 (2026-06-10 재배치)]
  💧 위험회피(상시형):
    DD  : 리스크      - 드로다운 스탑. 고점 대비 일정% 빠지면 현금화
    VOL : 시장 상태   - 실현변동성 레짐. 평온=탑승 / 중간=반반 / 격동=현금
  🔥 추세순응(상시형):
    MA  : 추세        - 이평. 가격이 장기 이평 위면 탑승
    MOM : 추세 강화   - 모멘텀. 최근 수익률이 양수면 추세에 더 올라탐
  🌿 역발상(이벤트형 — 발동일만 의견, 평소 기권):
    REV_RSI : 심리    - 과매도(RSI<30) 투매에 매수 의견(1.0)
    REV_BB  : 변동성  - 볼린저 하단 이탈(과대 낙폭)에 매수 의견(1.0)

  ※ 구 RSI(과열 감산)·BB(상단 감산)는 실측 결과 거의 상수 1(죽은 시그널)이고
    과열 회피는 양의 기대수익을 버리는 규칙이라 명단에서 제외했다
    (worklog/2026-06-10_signal_rework_plan.md). 함수는 비교/실험용으로 보존.

[전략 = 유전자들의 '기권 제외 평균' 포지션]
  여러 유전자를 가지면 그날 의견 낸 유전자들의 포지션만 평균낸다.
  예) MA가 1.0(탑승), DD가 0.0(현금화) -> 0.5 (반반).
      DD가 0.0인데 REV_RSI가 발동(1.0)한 투매일 -> 0.5 (패닉 속 부분 진입).
      REV_RSI가 기권(평소)이면 평균에서 빠진다 -> 잉어킹은 벤치에.
  방어(DD/VOL)·공격(MA/MOM)·역발상(REV_*)이 서로 견제하는 구조.
"""
import numpy as np
import pandas as pd

# ── 시그널 튜닝 파라미터 (한곳에 모음 = 진화/실험 시 여기만 만지면 됨) ──
MA_WINDOW = 200                       # MA: 장기 이평 기간(일)
RSI_PERIOD, RSI_OVERBOUGHT = 14, 70   # RSI: 기간 / 과열선 (구 시그널용, 명단 제외)
RSI_OVERSOLD = 30                     # REV_RSI: 과매도선 (이 아래로 투매 시 매수 의견)
BB_PERIOD, BB_K = 20, 2.0             # BB: 기간 / 표준편차 배수 (REV_BB도 같은 밴드 사용)
DD_LIMIT = 0.10                       # DD: 고점 대비 허용 낙폭(넘으면 현금화)
VOL_PERIOD = 20                       # VOL: 실현변동성 측정 기간
VOL_CALM, VOL_STRESSED = 0.010, 0.020 # VOL: 평온/격동 일변동성 임계
MOM_LOOKBACK = 63                     # MOM: 모멘텀 측정 기간(약 3개월)


def _rsi(prices: pd.Series, n: int = RSI_PERIOD) -> pd.Series:
    """RSI(Cutler/SMA 방식). 계산 불가 구간은 중립 50."""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(n, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def signal_MA(prices: pd.Series, window: int = MA_WINDOW) -> pd.Series:
    """추세추종: 가격이 장기 이평(기본 200일) 위면 1, 아니면 0."""
    sma = prices.rolling(window, min_periods=1).mean()
    return (prices > sma).astype(float)


def signal_RSI(prices: pd.Series, n: int = RSI_PERIOD,
               overbought: int = RSI_OVERBOUGHT) -> pd.Series:
    """[명단 제외] 과매수 회피: RSI가 과열선(70) 아래면 탑승(1), 과열이면 현금(0).
    실측 결과 거의 상수 1 + 과열 회피는 기대수익을 버리는 규칙이라 풀에서 뺐다."""
    rsi = _rsi(prices, n)
    return (rsi < overbought).astype(float)


def signal_REV_RSI(prices: pd.Series, n: int = RSI_PERIOD,
                   oversold: int = RSI_OVERSOLD) -> pd.Series:
    """역발상(이벤트형): RSI가 과매도선(기본 30) 아래로 투매되면 매수 의견(1.0).
    평소에는 기권(NaN) — 결합 평균에서 빠진다(현금 앵커 방지)."""
    rsi = _rsi(prices, n)
    pos = pd.Series(np.nan, index=prices.index)
    pos[rsi < oversold] = 1.0
    return pos


def signal_BB(prices: pd.Series, n: int = BB_PERIOD, k: float = BB_K) -> pd.Series:
    """[명단 제외] 볼린저 상단밴드 위(과열)면 현금(0), 그 외엔 탑승(1).
    실측 결과 평균 포지션 0.97 = 사실상 상수 1(정보 없음)이라 풀에서 뺐다."""
    ma = prices.rolling(n, min_periods=1).mean()
    sd = prices.rolling(n, min_periods=1).std().fillna(0)
    upper = ma + k * sd
    return (prices <= upper).astype(float)


def signal_REV_BB(prices: pd.Series, n: int = BB_PERIOD, k: float = BB_K) -> pd.Series:
    """역발상(이벤트형): 가격이 볼린저 하단밴드 아래로 과대 낙폭하면 매수 의견(1.0).
    평소에는 기권(NaN) — 결합 평균에서 빠진다(현금 앵커 방지)."""
    ma = prices.rolling(n, min_periods=1).mean()
    sd = prices.rolling(n, min_periods=1).std().fillna(0)
    lower = ma - k * sd
    pos = pd.Series(np.nan, index=prices.index)
    pos[prices < lower] = 1.0
    return pos


def signal_DD(prices: pd.Series, limit: float = DD_LIMIT) -> pd.Series:
    """드로다운 스탑: 최근 고점 대비 limit(기본 10%) 넘게 빠지면 현금화(0)."""
    peak = prices.cummax()
    drawdown = prices / peak - 1.0          # 0 이하의 값 (예: -0.15 = 고점대비 -15%)
    return (drawdown > -limit).astype(float)


def signal_VOL(prices: pd.Series, n: int = VOL_PERIOD,
               calm: float = VOL_CALM, stressed: float = VOL_STRESSED) -> pd.Series:
    """시장 상태(변동성 레짐): 평온하면 탑승(1.0) / 중간이면 반반(0.5) / 격동이면 현금(0.0)."""
    vol = prices.pct_change().rolling(n, min_periods=1).std().fillna(0)
    pos = pd.Series(0.5, index=prices.index)        # 기본 = 중간 상태
    pos[vol <= calm] = 1.0                            # 평온장 = 위험선호
    pos[vol > stressed] = 0.0                         # 격동장 = 위험회피
    return pos


def signal_MOM(prices: pd.Series, lookback: int = MOM_LOOKBACK) -> pd.Series:
    """추세 강화(모멘텀): 최근 lookback(약 3개월) 수익률이 양수면 탑승(1), 아니면 이탈(0)."""
    momentum = prices / prices.shift(lookback) - 1.0
    return (momentum > 0).astype(float)              # 초기(데이터 부족) 구간은 0


# ─────────────────────────────────────────────────────────────────────
# 야생 포켓퀀트 (2026-06-13 v1.x 시즌 — 외부 정보원 7마리)
# 모두 이벤트형 — 발동일만 의견, 평소 기권. 자산-횡단 알파 후보.
# 외부 시계열은 data_io.data에서 yfinance 캐시로 받음 (캐시 hit이면 빠름).
# ─────────────────────────────────────────────────────────────────────
VOL_SPIKE_THRESHOLD = 2.5    # 거래량 평균 대비 폭증 배수
FEAR_THRESHOLD = 30.0        # VIX 공포 임계
US10Y_DROP_BP = 0.5          # ^TNX 60일 평균 대비 -0.5%p 하락 임계
DXY_UP_PCT = 0.015           # DXY 60일 평균 대비 +1.5% 상승 임계
SPY_TLT_THRESHOLD = 0.01     # TLT/SPY 비율 60일 평균 대비 +1% 임계 (채권 강세)
QQQ_SPY_THRESHOLD = 0.0      # QQQ/SPY 60일 평균 대비 위 = 성장주 우위
QQQ_DIA_THRESHOLD = 0.0      # QQQ/DIA 60일 평균 대비 위 = 테크 우위 (vs 다우 가치)


def _is_synthetic(prices: pd.Series) -> bool:
    """블록 부트스트랩 합성 세계인지 확인한다.

    합성 세계 날짜는 실제 시장 날짜가 아니라 순서 라벨이다. 여기에 실제 VIX/TNX/UUP
    같은 외부 시계열을 붙이면 서로 다른 역사 조각을 섞는 검증 오염이 된다.
    """
    return bool(getattr(prices, "attrs", {}).get("synthetic"))


def _fetch_external(ticker: str, prices: pd.Series) -> pd.Series:
    """외부 시계열을 prices 인덱스에 정렬해 받아오는 헬퍼.
    캐시 hit이면 yfinance 호출 없이 즉시. forward-fill로 holiday 차이 흡수.
    데이터가 요청 기간을 cover 못 하면 (예: UUP는 2007년~) NaN 시리즈 반환 →
    시그널은 자연스러운 기권 처리(이벤트형 패턴과 일관)."""
    from app.backend.data_io.data import get_prices    # 지연 import (순환 회피)
    if _is_synthetic(prices):
        return pd.Series(np.nan, index=prices.index)
    start = prices.index[0].strftime("%Y-%m-%d")
    end = prices.index[-1].strftime("%Y-%m-%d")
    try:
        series = get_prices(ticker, start, end)
    except RuntimeError:
        return pd.Series(np.nan, index=prices.index)
    return series.reindex(prices.index, method="ffill")


def signal_VOL_SPIKE(prices: pd.Series,
                     threshold: float = VOL_SPIKE_THRESHOLD) -> pd.Series:
    """거래량 폭증 + 가격 음봉 = 패닉 매도 → 역발상 매수 의견. 평소 기권."""
    if _is_synthetic(prices):
        return pd.Series(np.nan, index=prices.index)
    from app.backend.data_io.data import get_volume
    ticker = prices.name or "QQQ"
    start = prices.index[0].strftime("%Y-%m-%d")
    end = prices.index[-1].strftime("%Y-%m-%d")
    vol = get_volume(str(ticker), start, end).reindex(prices.index, method="ffill")
    avg = vol.rolling(20, min_periods=10).mean()
    spike = vol / avg
    down = prices.pct_change() < 0
    pos = pd.Series(np.nan, index=prices.index)
    pos[(spike > threshold) & down] = 1.0
    return pos


def signal_FEAR(prices: pd.Series,
                threshold: float = FEAR_THRESHOLD) -> pd.Series:
    """VIX > threshold = 공포 → 역발상 매수 의견 (바닥 신호). 평소 기권."""
    vix = _fetch_external("^VIX", prices)
    pos = pd.Series(np.nan, index=prices.index)
    pos[vix > threshold] = 1.0
    return pos


def signal_US10Y(prices: pd.Series,
                 drop_bp: float = US10Y_DROP_BP) -> pd.Series:
    """^TNX 60일 평균 대비 -drop_bp 이상 하락 = 성장주 우호 → 매수 의견. 평소 기권."""
    tnx = _fetch_external("^TNX", prices)
    ma = tnx.rolling(60, min_periods=20).mean()
    diff = tnx - ma
    pos = pd.Series(np.nan, index=prices.index)
    pos[diff < -drop_bp] = 1.0
    return pos


def signal_DXY(prices: pd.Series,
               up_pct: float = DXY_UP_PCT) -> pd.Series:
    """달러 강세(UUP) 60일 평균 대비 +up_pct 상승 = 안전자산 선호 → 방어 의견 (0). 평소 기권.
    UUP는 Invesco DXY 추종 ETF(2007년~). 그 이전 시기는 데이터 없음 = NaN 기권 처리."""
    dxy = _fetch_external("UUP", prices)
    ma = dxy.rolling(60, min_periods=20).mean()
    pct = (dxy - ma) / ma
    pos = pd.Series(np.nan, index=prices.index)
    pos[pct > up_pct] = 0.0
    return pos


def signal_SPY_TLT(prices: pd.Series,
                   threshold: float = SPY_TLT_THRESHOLD) -> pd.Series:
    """TLT/SPY 비율 60일 평균 대비 +threshold = 채권 강세 → 방어 의견 (0). 평소 기권."""
    spy = _fetch_external("SPY", prices)
    tlt = _fetch_external("TLT", prices)
    ratio = tlt / spy
    ma = ratio.rolling(60, min_periods=20).mean()
    pct = (ratio - ma) / ma
    pos = pd.Series(np.nan, index=prices.index)
    pos[pct > threshold] = 0.0
    return pos


def signal_QQQ_SPY(prices: pd.Series,
                   threshold: float = QQQ_SPY_THRESHOLD) -> pd.Series:
    """QQQ/SPY 비율 60일 평균 대비 위 = 성장주 우위 → 매수 의견. 평소 기권."""
    qqq = _fetch_external("QQQ", prices)
    spy = _fetch_external("SPY", prices)
    ratio = qqq / spy
    ma = ratio.rolling(60, min_periods=20).mean()
    pct = (ratio - ma) / ma
    pos = pd.Series(np.nan, index=prices.index)
    pos[pct > threshold] = 1.0
    return pos


def signal_QQQ_DIA(prices: pd.Series,
                   threshold: float = QQQ_DIA_THRESHOLD) -> pd.Series:
    """QQQ/DIA 비율 60일 평균 대비 위 = 테크 우위(vs 다우 가치) → 매수 의견. 평소 기권."""
    qqq = _fetch_external("QQQ", prices)
    dia = _fetch_external("DIA", prices)
    ratio = qqq / dia
    ma = ratio.rolling(60, min_periods=20).mean()
    pct = (ratio - ma) / ma
    pos = pd.Series(np.nan, index=prices.index)
    pos[pct > threshold] = 1.0
    return pos


# 유전자 이름 -> 시그널 함수.
# 이 레지스트리가 '어떤 유전자가 존재하는가'의 단일 출처(source of truth)다.
# [2026-06-10 재배치] 구 RSI(과열)·BB(상단)는 죽은 시그널이라 제외하고
# 역발상 이벤트형 REV_RSI(과매도 매수)·REV_BB(하단 매수)로 교체. 3타입 × 2마리.
# [2026-06-13 v1.x] 야생 7마리 추가 — 외부 정보원으로 자산-횡단 알파 후보.
GENE_SIGNALS = {
    # 옛 6마리 (가격 기반)
    "DD": signal_DD,                # 💧 위험회피
    "VOL": signal_VOL,              # 💧 위험회피
    "MA": signal_MA,                # 🔥 추세순응
    "MOM": signal_MOM,              # 🔥 추세순응
    "REV_RSI": signal_REV_RSI,      # 🌿 역발상 (이벤트형)
    "REV_BB": signal_REV_BB,        # 🌿 역발상 (이벤트형)
    # v1.x 야생 7마리 (외부 정보원)
    "VOL_SPIKE": signal_VOL_SPIKE,  # 📊 자산 내부 (거래량)
    "FEAR": signal_FEAR,            # 😱 글로벌 (VIX 공포)
    "US10Y": signal_US10Y,          # 💵 글로벌 (10년 금리)
    "DXY": signal_DXY,              # 💰 글로벌 (달러 인덱스)
    "SPY_TLT": signal_SPY_TLT,      # 🏦 글로벌 (채권-주식 상관)
    "QQQ_SPY": signal_QQQ_SPY,      # 🚀 글로벌 (성장주 vs S&P500)
    "QQQ_DIA": signal_QQQ_DIA,      # 🏭 글로벌 (테크 vs 다우 가치)
}

# 사용 가능한 모든 유전자 이름 (v1.x: 13마리).
ALL_GENES = list(GENE_SIGNALS.keys())

# 참고: 유전자 설명 카드(포켓퀀트 도감)는 dex.py(SIGNAL_CARDS)에 있다.


def combine_positions(positions: list[pd.Series],
                      weights: list[float] | None = None) -> pd.Series:
    """
    시그널들의 포지션을 '기권 제외 (가중)평균'으로 합친다.

    [기권(NaN) 규칙]
    상시형 시그널(MA/MOM/DD/VOL)은 매일 0~1 의견을 내지만, 이벤트형 시그널
    (REV_*)은 발동일에만 의견(1.0)을 내고 평소엔 기권(NaN)한다.
    기권을 0으로 취급하면 "의견 없음"이 "현금 가라"로 집계돼 이벤트형 시그널이
    상시 현금 앵커가 되므로(잉어킹 강제 출전 문제), 그날 의견을 낸 시그널들
    끼리만 평균한다 = 기권한 포켓퀀트은 벤치, 출전한 포켓퀀트끼리 싸운다.
      - 전원 기권한 날: 포지션 0.0 (아무도 의견 없으면 들어가지 않는다)

    [weights — NSGA-III 결정변수]
    weights를 주면 '기권 제외 가중평균'이 된다:
        position = Σ wᵢ·posᵢ / Σ wᵢ   (그날 의견 낸 i만 합산)
    분모에 Σw가 있어 가중치의 절대 크기는 의미 없고 비율만 남는다
    = 예산 제약(Σw=1)이 결합식에 내장 → "전부 최대" 퇴화 경사가 없다.
    weights=None(기본)은 기존 동일가중 평균과 비트 단위로 같다(골든 테스트 보호).
      - 그날 의견 낸 시그널들의 가중치 합이 0이면 기권 취급(0.0).
    """
    df = pd.concat(positions, axis=1)
    if weights is None:
        combined = df.mean(axis=1)        # NaN(기권)은 pandas 평균에서 자동 제외
    else:
        w = np.asarray(weights, dtype=float)
        voted = df.notna().to_numpy()                    # 그날 의견 냈는지
        values = df.fillna(0.0).to_numpy()
        denom = (voted * w).sum(axis=1)                  # 출전 시그널들의 가중치 합
        numer = (values * w).sum(axis=1)
        combined = pd.Series(
            np.where(denom > 0, numer / np.where(denom > 0, denom, 1.0), np.nan),
            index=df.index,
        )
    return combined.fillna(0.0).clip(0.0, 1.0)


# NSGA-III가 탐색하는 시그널 파라미터의 기본값/탐색범위 정의는 nsga3.py에 있다.
# 여기서는 "파라미터를 주입해 포지션 목록을 만드는" 입구만 제공한다.
def positions_with_params(prices: pd.Series, params: dict | None = None) -> list[pd.Series]:
    """ALL_GENES 순서대로, 파라미터를 주입한 포지션 목록을 만든다.
    params에 없는 키는 모듈 기본값을 쓴다 (params=None이면 전부 기본값 =
    GENE_SIGNALS 경로와 동일)."""
    p = params or {}
    return [
        signal_DD(prices, limit=p.get("DD_LIMIT", DD_LIMIT)),
        signal_VOL(prices, calm=p.get("VOL_CALM", VOL_CALM),
                   stressed=p.get("VOL_STRESSED", VOL_STRESSED)),
        signal_MA(prices, window=p.get("MA_WINDOW", MA_WINDOW)),
        signal_MOM(prices, lookback=p.get("MOM_LOOKBACK", MOM_LOOKBACK)),
        signal_REV_RSI(prices, oversold=p.get("RSI_OVERSOLD", RSI_OVERSOLD)),
        signal_REV_BB(prices, k=p.get("BB_K", BB_K)),
        # v1.x 야생 7마리 — 파라미터 동결 (탐색공간 추가 시 정정)
        signal_VOL_SPIKE(prices),
        signal_FEAR(prices),
        signal_US10Y(prices),
        signal_DXY(prices),
        signal_SPY_TLT(prices),
        signal_QQQ_SPY(prices),
        signal_QQQ_DIA(prices),
    ]


def combined_position(genes: list[str], prices: pd.Series) -> pd.Series:
    """전략의 유전자들이 만드는 포지션을 합쳐 최종 일별 포지션(0~1)을 만든다."""
    if not genes:                                  # 유전자 없으면 풀매수로 간주
        return pd.Series(1.0, index=prices.index)
    return combine_positions([GENE_SIGNALS[g](prices) for g in genes])
