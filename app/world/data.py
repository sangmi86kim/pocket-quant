"""
data.py - ?ㅼ젣 二쇨? ?곗씠?곕? 諛쏆븘?ㅻ뒗 ?뚯씪 (yfinance + ?붿뒪??罹먯떆)

???뚯씪??AGENTS.md?먯꽌 留먰븯??'?쒕뜡 蹂댁젙媛????ㅺ퀎痢↔컪 援먯껜'???낃뎄??
?댁쟾??battle.py媛 random.randint(-20,20)?쇰줈 ?대묠 ?먯닔瑜?留뚮뱾?덉?留?
?댁젣???ш린??諛쏆븘??吏꾩쭨 媛寃⑹쑝濡?吏꾩쭨 諛깊뀒?ㅽ듃瑜??뚮┛??

[罹먯떆 ?꾨왂]
??궗??湲곌컙(?룹뺨/湲덉쑖?꾧린/肄붾줈??湲덈━?쇳겕)? 媛믪씠 ??踰??ㅼ떆 ??蹂?쒕떎.
洹몃옒????踰?諛쏆쑝硫?data_cache/ ??CSV濡???ν븯怨? ?ㅼ쓬遺?곕뒗 ?ㅽ듃?뚰겕 ?놁씠 洹멸구 ?대떎.
  - 罹먯떆 ?덉쑝硫?       -> ?붿뒪?ъ뿉??利됱떆 濡쒕뱶 (?ㅽ봽?쇱씤 OK)
  - 罹먯떆 ?놁쑝硫?       -> yfinance濡??ㅼ슫濡쒕뱶 ?????
  - ???????섎㈃      -> 紐낇솗???먮윭 (?ㅽ듃?뚰겕 ?꾩슂)
"""
import atexit
import os
from dataclasses import dataclass

import pandas as pd

from app.pocket.models import Gym

# 罹먯떆 ?대뜑: ?꾨줈?앺듃 猷⑦듃??data_cache/  (???뚯씪? app/backend/data_io/ ?덉뿉 ?덉쓬)
# 2026-06-13遺???곗빱蹂??쒕툕?대뜑 援ъ“ ??KIS ??異붽? ticker媛 ?욎씪 ?먮━ 誘몃━ ?뺣━.
CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data_cache")
)

# 吏??200???댄룊 ??瑜??곗슦湲??꾪빐 ?됯? ?쒖옉?쇰낫???욎そ?쇰줈 ??諛쏅뒗 踰꾪띁 ?쇱닔.
# ??踰꾪띁 援ш컙? 吏???뚮컢?낆뿉留??곗씠怨? ?ㅼ젣 ?깆쟻 怨꾩궛(battle)?먯꽌???섎씪?몃떎.
WARMUP_DAYS = 400


def _cache_path(ticker: str, start: str, end: str) -> str:
    """?곗빱蹂??쒕툕?대뜑 ??湲곌컙 ?뚯씪. ?? data_cache/SPY/2000-01-01_2002-12-31.csv"""
    safe = f"{start}_{end}.csv".replace(":", "-")
    return os.path.join(CACHE_DIR, ticker, safe)


def _clean_prices(series: pd.Series, ticker: str) -> pd.Series:
    """罹먯떆/?ㅼ슫濡쒕뱶 媛寃⑹쓣 諛깊뀒?ㅽ듃???????덈뒗 ?レ옄 ?쒓퀎?대줈 ?뺣━?쒕떎."""
    series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    series.name = ticker
    return series


def _covers_end(series: pd.Series, end: str) -> bool:
    """罹먯떆媛 ?붿껌 醫낅즺?쇨퉴吏 異⑸텇?쒖? ?뺤씤?쒕떎. 二쇰쭚 醫낅즺?쇱? 吏곸쟾 ?곸뾽?쇱씠硫?異⑸텇?섎떎."""
    if series.empty:
        return False
    last_date = pd.Timestamp(series.index.max()).normalize()
    end_date = pd.Timestamp(end).normalize()
    if last_date >= end_date:
        return True
    return len(pd.bdate_range(last_date + pd.Timedelta(days=1), end_date)) == 0


def get_prices(ticker: str, start: str, end: str) -> pd.Series:
    """
    ???곗빱??'?섏젙醫낃?(Adjusted Close)' ?쒓퀎?댁쓣 ?뚮젮以??

    諛섑솚: pd.Series (?몃뜳???좎쭨, 媛?媛寃?. 諛깊뀒?ㅽ듃??????以꾩쭨由?媛寃⑸쭔 ?대떎.
    """
    path = _cache_path(ticker, start, end)

    # (1) 罹먯떆 ?곗꽑 ???덉쑝硫??ㅽ듃?뚰겕 ?놁씠 利됱떆 ?ъ슜
    if os.path.exists(path):
        series = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        series = _clean_prices(series, ticker)
        if _covers_end(series, end):
            return series

    # (2) 罹먯떆 ?놁쓬 -> yfinance濡??ㅼ슫濡쒕뱶
    #     臾닿굅???쇱씠釉뚮윭由щ씪 ?ш린???꾩슂???뚮쭔) import ?쒕떎.
    import yfinance as yf

    os.makedirs(os.path.dirname(path), exist_ok=True)   # data_cache/<ticker>/
    yf.set_tz_cache_location(CACHE_DIR)
    _register_yf_cleanup()                              # 醫낅즺 ??硫뷀? ?뚯씪留?泥?냼

    # yfinance??end??諛고??곸씠誘濡? ???⑥닔??end???몄텧??湲곗? '?ы븿'?쇰줈 留욎텣??
    download_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # auto_adjust=True -> 'Close'媛 ?대? 諛곕떦/遺꾪븷 諛섏쁺???섏젙醫낃?
    df = yf.download(ticker, start=start, end=download_end,
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(
            f"[data] '{ticker}' {start}~{end} ?곗씠?곕? 諛쏆? 紐삵뻽?듬땲?? "
            f"?ㅽ듃?뚰겕 ?먮뒗 ?곗빱/湲곌컙???뺤씤?섏꽭??"
        )

    # yfinance媛 硫?곗씤?깆뒪 而щ읆??以??뚭? ?덉뼱 'Close'留??덉쟾?섍쾶 異붿텧
    close = df["Close"]
    if isinstance(close, pd.DataFrame):       # ?щ윭 ?곗빱 ?뺥깭濡???寃쎌슦
        close = close.iloc[:, 0]
    close = _clean_prices(close, ticker)

    # (3) 罹먯떆?????(?ㅼ쓬 ?ㅽ뻾遺???ㅽ봽?쇱씤)
    close.to_csv(path)

    return close


# ??????????????????????????????????????????????
# 泥댁쑁愿 + 誘몃━ ?밴꺼??媛寃⑹쓣 ??臾띠쓬?쇰줈 (= "?곗씠???↔꺼?ㅺ퀬" ?④퀎??寃곌낵臾?
# ?대젃寃?誘몃━ 濡쒕뵫???먮㈃ battle? I/O ?놁씠 怨꾩궛留??섎㈃ ?섍퀬,
# 吏꾪솕 紐⑤뱶?먯꽌 媛寃⑹쓣 ?꾨왂留덈떎 ?ㅼ떆 ?쎌? ?딆븘 鍮좊Ⅴ??援?㈃??1踰덈쭔 濡쒕뵫).
# ??????????????????????????????????????????????
@dataclass
class LoadedGym:
    gym: Gym                # 泥댁쑁愿 ?뺤쓽(?대쫫/湲곌컙/?곗빱)
    prices: pd.Series       # ?뚮컢??踰꾪띁 ?ы븿 ?꾩껜 媛寃??쒓퀎??


def load_gym(gym: Gym) -> LoadedGym:
    """泥댁쑁愿 ?섎굹??媛寃⑹쓣 (?뚮컢??踰꾪띁 ?ы븿) 諛쏆븘 LoadedGym?쇰줈 臾띕뒗??"""
    window_start = pd.Timestamp(gym.start)
    fetch_start = (window_start - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    prices = get_prices(gym.ticker, fetch_start, gym.end)
    return LoadedGym(gym=gym, prices=prices)


def load_gyms(gyms: list[Gym]) -> list[LoadedGym]:
    """??泥댁쑁愿??媛寃⑹쓣 ??踰덉뿉 ?밴꺼?⑤떎 (?뚯씠?꾨씪?몄쓽 1?④퀎: ?곗씠??濡쒕뵫)."""
    return [load_gym(g) for g in gyms]


# ?? 嫄곕옒????VOL_SPIKE 媛숈? 嫄곕옒???섏〈 ?쒓렇?먯슜 (2026-06-13 ?좎꽕) ????????
# ?쇨??? get_prices()? 媛숈? 罹먯떆 ?붾젆?좊━/?뚯씪紐?洹쒖빟, suffix _vol 留?異붽?.
def get_volume(ticker: str, start: str, end: str) -> pd.Series:
    """?곗빱 ?쇰퀎 嫄곕옒???쒓퀎?? 罹먯떆: data_cache/<ticker>/<湲곌컙>_vol.csv."""
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
            f"[data] '{ticker}' {start}~{end} 嫄곕옒?됱쓣 諛쏆? 紐삵뻽?듬땲??"
        )
    vol = df["Volume"]
    if isinstance(vol, pd.DataFrame):
        vol = vol.iloc[:, 0]
    vol = pd.to_numeric(vol, errors="coerce").dropna().sort_index()
    vol.name = ticker
    vol.to_csv(path)
    return vol


# ?? yfinance 硫뷀? ?뚯씪 ?먮룞 泥?냼 ???????????????????????????????????????????
# yfinance??留??ㅼ슫濡쒕뱶留덈떎 data_cache/cookies.db 쨌 tkr-tz.db瑜?留뚮뱾??CSV?
# ?욎씠寃??쒕떎 ???대뜑媛 吏?遺꾪빐吏??二쇰쾾. ?뚰겕?뚮줈??醫낅즺 ??atexit) 硫뷀?留?
# 吏?대떎. ?ㅻ뜲?댄꽣 CSV(data_cache/<ticker>/*.csv)??蹂댁〈 ???ㅼ쓬 ?ㅽ뻾 罹먯떆.
_YF_CLEANUP_REGISTERED = False


def cleanup_yf_meta() -> None:
    for name in ("cookies.db", "tkr-tz.db"):
        path = os.path.join(CACHE_DIR, name)
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass    # ?ㅻⅨ ?꾨줈?몄뒪媛 伊먭퀬 ?덉뼱???ㅼ쓬 ?ㅽ뻾???ㅼ떆 ?쒕룄?섎㈃ ??


def _register_yf_cleanup() -> None:
    """泥?yfinance ?몄텧 ????踰덈쭔 ?깅줉 ???꾧컧 ??yf ???곕뒗 吏꾩엯?먯? 洹몃?濡???"""
    global _YF_CLEANUP_REGISTERED
    if not _YF_CLEANUP_REGISTERED:
        atexit.register(cleanup_yf_meta)
        _YF_CLEANUP_REGISTERED = True
