"""Fallback công khai: yfinance (không cần tài khoản/token).

Không bắt buộc: nếu thiếu thư viện hoặc lỗi mạng -> log + trả DataFrame rỗng.
Yahoo dùng giá điều chỉnh cho OHLC -> qui về 'close' không đổi, giữ nguyên tương đối.
"""
from __future__ import annotations

import time

import pandas as pd

from app.config import cfg_get
from app.logging_utils import get_logger

log = get_logger(__name__)


def has_yfinance() -> bool:
    try:
        import yfinance  # noqa: F401
        return True
    except Exception:
        return False


def _to_df(sym: str, raw) -> pd.DataFrame:
    if raw is None or (hasattr(raw, "empty") and raw.empty):
        return pd.DataFrame()
    df = raw.reset_index()
    dcol = df.columns[0]
    df = df.rename(columns={dcol: "date"})
    keep = {}
    for want in ["open", "high", "low", "close", "volume"]:
        for c in df.columns:
            if str(c).lower() == want:
                keep[c] = want
    df = df.rename(columns=keep)
    df["symbol"] = sym.upper()
    df["turnover"] = pd.NA  # yfinance không có giá trị giao dịch -> ước lượng volume*close
    return df[["date", "symbol", "open", "high", "low", "close", "volume", "turnover"]]


def fetch_history_yahoo(symbol: str, start: str, end: str, cfg: dict,
                        suffix: str = ".HN") -> pd.DataFrame:
    if not cfg_get(cfg, "data.use_yfinance_fallback", True) or not has_yfinance():
        return pd.DataFrame()
    import yfinance as yf  # noqa: PLC0415
    full = symbol.upper() if "." in symbol else symbol.upper() + suffix
    attempts = int(cfg_get(cfg, "data.max_retries", 3))
    for i in range(1, attempts + 1):
        try:
            raw = yf.download(full, start=start, end=end, progress=False,
                              auto_adjust=True, threads=False)
            df = _to_df(symbol, raw)
            if not df.empty:
                return df
            log.warning("yfinance trả rỗng cho %s", full)
        except Exception as e:  # noqa: BLE001
            log.warning("yfinance lỗi (%d/%d) %s: %s", i, attempts, full, e)
            time.sleep(float(cfg_get(cfg, "data.retry_backoff_seconds", 2.0)) * i)
    return pd.DataFrame()


def fetch_index_yahoo(ticker: str, start: str, end: str, cfg: dict) -> pd.DataFrame:
    """Chỉ số thị trường, ví dụ ^VNINDEX trên Yahoo (công khai)."""
    return fetch_history_yahoo(ticker, start, end, cfg, suffix="")
