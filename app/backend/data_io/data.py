"""
data.py - 실제 주가 데이터를 받아오는 파일 (yfinance + 디스크 캐시)

이 파일이 AGENTS.md에서 말하던 '랜덤 보정값 → 실계측값 교체'의 입구다.
이전엔 battle.py가 random.randint(-20,20)으로 운빨 점수를 만들었지만,
이제는 여기서 받아온 진짜 가격으로 진짜 백테스트를 돌린다.

[캐시 전략]
역사적 기간(닷컴/금융위기/코로나/금리쇼크)은 값이 두 번 다시 안 변한다.
그래서 한 번 받으면 data_cache/ 에 CSV로 저장하고, 다음부터는 네트워크 없이 그걸 쓴다.
  - 캐시 있으면        -> 디스크에서 즉시 로드 (오프라인 OK)
  - 캐시 없으면        -> yfinance로 다운로드 후 저장
  - 둘 다 안 되면      -> 명확한 에러 (네트워크 필요)
"""
import atexit
import os
from dataclasses import dataclass

import pandas as pd

from ..core.models import Gym

# 캐시 폴더: 프로젝트 루트의 data_cache/  (이 파일은 app/backend/data_io/ 안에 있음)
# 2026-06-13부터 티커별 서브폴더 구조 — KIS 등 추가 ticker가 섞일 자리 미리 정리.
CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data_cache")
)

# 지표(200일 이평 등)를 데우기 위해 평가 시작일보다 앞쪽으로 더 받는 버퍼 일수.
# 이 버퍼 구간은 지표 워밍업에만 쓰이고, 실제 성적 계산(battle)에서는 잘라낸다.
WARMUP_DAYS = 400


def _cache_path(ticker: str, start: str, end: str) -> str:
    """티커별 서브폴더 안 기간 파일. 예) data_cache/SPY/2000-01-01_2002-12-31.csv"""
    safe = f"{start}_{end}.csv".replace(":", "-")
    return os.path.join(CACHE_DIR, ticker, safe)


def _clean_prices(series: pd.Series, ticker: str) -> pd.Series:
    """캐시/다운로드 가격을 백테스트에 쓸 수 있는 숫자 시계열로 정리한다."""
    series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    series.name = ticker
    return series


def _covers_end(series: pd.Series, end: str) -> bool:
    """캐시가 요청 종료일까지 충분한지 확인한다. 주말 종료일은 직전 영업일이면 충분하다."""
    if series.empty:
        return False
    last_date = pd.Timestamp(series.index.max()).normalize()
    end_date = pd.Timestamp(end).normalize()
    if last_date >= end_date:
        return True
    return len(pd.bdate_range(last_date + pd.Timedelta(days=1), end_date)) == 0


def get_prices(ticker: str, start: str, end: str) -> pd.Series:
    """
    한 티커의 '수정종가(Adjusted Close)' 시계열을 돌려준다.

    반환: pd.Series (인덱스=날짜, 값=가격). 백테스트는 이 한 줄짜리 가격만 쓴다.
    """
    path = _cache_path(ticker, start, end)

    # (1) 캐시 우선 — 있으면 네트워크 없이 즉시 사용
    if os.path.exists(path):
        series = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        series = _clean_prices(series, ticker)
        if _covers_end(series, end):
            return series

    # (2) 캐시 없음 -> yfinance로 다운로드
    #     무거운 라이브러리라 여기서(필요할 때만) import 한다.
    import yfinance as yf

    os.makedirs(os.path.dirname(path), exist_ok=True)   # data_cache/<ticker>/
    yf.set_tz_cache_location(CACHE_DIR)
    _register_yf_cleanup()                              # 종료 시 메타 파일만 청소

    # yfinance의 end는 배타적이므로, 이 함수의 end는 호출자 기준 '포함'으로 맞춘다.
    download_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # auto_adjust=True -> 'Close'가 이미 배당/분할 반영된 수정종가
    df = yf.download(ticker, start=start, end=download_end,
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(
            f"[data] '{ticker}' {start}~{end} 데이터를 받지 못했습니다. "
            f"네트워크 또는 티커/기간을 확인하세요."
        )

    # yfinance가 멀티인덱스 컬럼을 줄 때가 있어 'Close'만 안전하게 추출
    close = df["Close"]
    if isinstance(close, pd.DataFrame):       # 여러 티커 형태로 온 경우
        close = close.iloc[:, 0]
    close = _clean_prices(close, ticker)

    # (3) 캐시에 저장 (다음 실행부턴 오프라인)
    close.to_csv(path)

    return close


# ──────────────────────────────────────────────
# 체육관 + 미리 당겨둔 가격을 한 묶음으로 (= "데이터 땡겨오고" 단계의 결과물)
# 이렇게 미리 로딩해 두면 battle은 I/O 없이 계산만 하면 되고,
# 진화 모드에서 가격을 전략마다 다시 읽지 않아 빠르다(국면당 1번만 로딩).
# ──────────────────────────────────────────────
@dataclass
class LoadedGym:
    gym: Gym                # 체육관 정의(이름/기간/티커)
    prices: pd.Series       # 워밍업 버퍼 포함 전체 가격 시계열


def load_gym(gym: Gym) -> LoadedGym:
    """체육관 하나의 가격을 (워밍업 버퍼 포함) 받아 LoadedGym으로 묶는다."""
    window_start = pd.Timestamp(gym.start)
    fetch_start = (window_start - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    prices = get_prices(gym.ticker, fetch_start, gym.end)
    return LoadedGym(gym=gym, prices=prices)


def load_gyms(gyms: list[Gym]) -> list[LoadedGym]:
    """전 체육관의 가격을 한 번에 당겨온다 (파이프라인의 1단계: 데이터 로딩)."""
    return [load_gym(g) for g in gyms]


# ── 거래량 — VOL_SPIKE 같은 거래량 의존 시그널용 (2026-06-13 신설) ────────
# 일관성: get_prices()와 같은 캐시 디렉토리/파일명 규약, suffix _vol 만 추가.
def get_volume(ticker: str, start: str, end: str) -> pd.Series:
    """티커 일별 거래량 시계열. 캐시: data_cache/<ticker>/<기간>_vol.csv."""
    safe = f"{start}_{end}_vol.csv".replace(":", "-")
    path = os.path.join(CACHE_DIR, ticker, safe)

    if os.path.exists(path):
        series = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
        if _covers_end(series, end):
            series.name = ticker
            return series

    import yfinance as yf
    os.makedirs(os.path.dirname(path), exist_ok=True)
    yf.set_tz_cache_location(CACHE_DIR)
    _register_yf_cleanup()

    download_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=download_end,
                     auto_adjust=True, progress=False)
    if df is None or df.empty or "Volume" not in df.columns:
        raise RuntimeError(
            f"[data] '{ticker}' {start}~{end} 거래량을 받지 못했습니다."
        )
    vol = df["Volume"]
    if isinstance(vol, pd.DataFrame):
        vol = vol.iloc[:, 0]
    vol = pd.to_numeric(vol, errors="coerce").dropna().sort_index()
    vol.name = ticker
    vol.to_csv(path)
    return vol


# ── yfinance 메타 파일 자동 청소 ───────────────────────────────────────────
# yfinance는 매 다운로드마다 data_cache/cookies.db · tkr-tz.db를 만들어 CSV와
# 섞이게 한다 — 폴더가 지저분해지는 주범. 워크플로우 종료 시(atexit) 메타만
# 지운다. 실데이터 CSV(data_cache/<ticker>/*.csv)는 보존 — 다음 실행 캐시.
_YF_CLEANUP_REGISTERED = False


def cleanup_yf_meta() -> None:
    for name in ("cookies.db", "tkr-tz.db"):
        path = os.path.join(CACHE_DIR, name)
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass    # 다른 프로세스가 쥐고 있어도 다음 실행에 다시 시도하면 됨


def _register_yf_cleanup() -> None:
    """첫 yfinance 호출 시 한 번만 등록 — 도감 등 yf 안 쓰는 진입점은 그대로 둠."""
    global _YF_CLEANUP_REGISTERED
    if not _YF_CLEANUP_REGISTERED:
        atexit.register(cleanup_yf_meta)
        _YF_CLEANUP_REGISTERED = True
