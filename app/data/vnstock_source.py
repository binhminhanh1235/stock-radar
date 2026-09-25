"""Nguồn dữ liệu chính: vnstock (miễn phí, không cần tài khoản/API key).

Mọi hàm đều kiểm tra import động; nếu vnstock chưa cài hoặc lỗi mạng thì
trả về DataFrame rỗng + log cảnh báo, KHÔNG làm hỏng pipeline.
Có retry cơ bản với backoff và sleep giữa các request để tôn trọng rate limit.
"""
from __future__ import annotations

import time
from datetime import date

import pandas as pd

from app.config import cfg_get
from app.logging_utils import get_logger

log = get_logger(__name__)


def _retry(fn, cfg, what: str):
    attempts = int(cfg_get(cfg, "data.max_retries", 3))
    backoff = float(cfg_get(cfg, "data.retry_backoff_seconds", 2.0))
    last_err = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - nguồn ngoài, bắt mọi lỗi
            last_err = e
            log.warning("Lần %d/%d gọi %s thất bại: %s", i, attempts, what, e)
            if i < attempts:
                time.sleep(backoff * i)
    raise RuntimeError(f"{what} thất bại sau {attempts} lần: {last_err}")


def has_vnstock() -> bool:
    try:
        import vnstock  # noqa: F401
        return True
    except Exception:
        return False


def fetch_symbol_list(cfg: dict) -> pd.DataFrame:
    """Danh sách cổ phiếu HOSE/HNX/UPCOM. Cột: symbol, exchange."""
    exchanges = cfg_get(cfg, "universe.exchanges", ["HOSE", "HNX", "UPCOM"])
    rows = []
    try:
        from vnstock.core.utils.client import ClientCN  # type: ignore
    except Exception:
        try:
            from vnstock.screener.symbol import SymbolClient  # type: ignore
        except Exception:
            log.warning("vnstock chưa được cài - bỏ qua danh sách symbols từ vnstock")
            return pd.DataFrame(columns=["symbol", "exchange"])

    def _one(exch: str) -> list[dict]:
        out = []
        try:
            df = ClientCN().get_symbols(domain=exch.lower(), sort='asc',
                                        object_type='stock', market_cap=None)
            col = "symbol" if "symbol" in df.columns else df.columns[0]
            for s in df[col].astype(str).str.upper():
                out.append({"symbol": s, "exchange": exch})
        except Exception as e:  # API đổi signature giữa các phiên bản
            log.warning("Không lấy được symbols %s từ vnstock: %s", exch, e)
        return out

    for exch in exchanges:
        try:
            rows.extend(_retry(lambda ex=exch: _one(ex), cfg, f"vnstock.get_symbols({exch})"))
        except RuntimeError:
            pass
        time.sleep(float(cfg_get(cfg, "data.request_sleep_seconds", 0.35)))
    df = pd.DataFrame(rows, columns=["symbol", "exchange"]).drop_duplicates()
    log.info("vnstock trả về %d mã trên %s", len(df), exchanges)
    return df


def fetch_history_vnstock(symbol: str, start: str | date, end: str | date,
                          cfg: dict) -> pd.DataFrame:
    """OHLCV daily một mã từ vnstock. Trả về DataFrame thô (đã thử chuẩn hoá cột)."""
    try:
        from vnstock.core.utils.client import ClientCN  # type: ignore
    except Exception:
        try:
            from vnstock import QuoteHistory  # type: ignore
        except Exception:
            return pd.DataFrame()

    def _one() -> pd.DataFrame:
        try:
            df = ClientCN().get_stock(symbol=symbol, trading_market='vn',
                                      data_type='stock', start=start, end=end)
        except Exception:
            df = QuoteHistory(symbol=symbol).history(start=str(start), end=str(end))
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

    return _retry(_one, cfg, f"vnstock.history({symbol})")


def fetch_index_vnstock(index: str, start: str | date, end: str | date,
                        cfg: dict) -> pd.DataFrame:
    """OHLCV chỉ số (VNIndex...) từ vnstock."""
    try:
        from vnstock.core.utils.client import ClientCN  # type: ignore
    except Exception:
        try:
            from vnstock import QuoteHistory  # type: ignore
        except Exception:
            return pd.DataFrame()

    def _one() -> pd.DataFrame:
        try:
            df = ClientCN().get_index(trading_market=index.upper(), data_type='index',
                                      start=start, end=end)
        except Exception:
            df = QuoteHistory(symbol=index, source='vndirect').history(
                start=str(start), end=str(end))
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

    return _retry(_one, cfg, f"vnstock.index({index})")


def fetch_sectors_vnstock(cfg: dict) -> pd.DataFrame:
    """Danh sách ngành + thành viên nếu vnstock hỗ trợ. Lỗi -> DataFrame rỗng."""
    try:
        from vnstock.core.utils.client import ClientCN  # type: ignore
    except Exception:
        return pd.DataFrame(columns=["symbol", "sector"])

    def _one() -> pd.DataFrame:
        df = ClientCN().get_indicators(domain='vn', indicator='SECTOR')
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

    try:
        df = _retry(_one, cfg, "vnstock.sectors")
        if df.empty:
            return pd.DataFrame(columns=["symbol", "sector"])
        # cấu trúc trả về thay đổi theo phiên bản -> chỉ nhận nếu có đủ cột
        cols = {c.lower(): c for c in df.columns}
        if "symbol" in cols and ("sector" in cols or "industry" in cols):
            sec = cols.get("sector", cols.get("industry"))
            return df.rename(columns={cols["symbol"]: "symbol", sec: "sector"})[
                ["symbol", "sector"]].dropna().drop_duplicates()
    except RuntimeError:
        pass
    return pd.DataFrame(columns=["symbol", "sector"])
