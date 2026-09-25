"""Data pipeline: download -> cache Parquet -> load (PHẦN 1).

Thứ tự ưu tiên nguồn: vnstock -> yfinance fallback.
Nếu cả hai không khả dụng, CLI báo lỗi rõ ràng và hướng dẫn chạy demo offline
(không bao giờ bịa dữ liệu từ trong pipeline).

Benchmark VNIndex:
 1) vnstock get_index('VNINDEX')
 2) yfinance ^VNINDEX
 3) tự xây benchmark equal-weight từ top thanh khoản (ghi rõ trong log + README,
    cột source = 'synthetic_ew_topN').
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from app.config import PROJECT_ROOT, cfg_get, load_config
from app.data import vnstock_source as vns
from app.data import yahoo_source as yah
from app.data.cleaning import clean_ohlcv
from app.logging_utils import get_logger
from app.storage import read_parquet, write_parquet

log = get_logger(__name__)


def data_root(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    p = Path(str(cfg_get(cfg, "data.cache_dir", "data")))
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


# ---------------- paths ----------------

def raw_daily_path(sym: str, cfg=None) -> Path:
    return data_root(cfg) / "raw" / "daily" / f"{sym}.parquet"


def processed_daily_path(cfg=None) -> Path:
    return data_root(cfg) / "processed" / "daily" / "daily_all.parquet"


def index_path(name: str, cfg=None) -> Path:
    return data_root(cfg) / "raw" / "index" / f"{name}.parquet"


def symbols_path(cfg=None) -> Path:
    return data_root(cfg) / "raw" / "symbols" / "symbols.parquet"


def sectors_path(cfg=None) -> Path:
    return data_root(cfg) / "raw" / "symbols" / "sectors.parquet"


def issues_path(cfg=None) -> Path:
    return data_root(cfg) / "raw" / "symbols" / "data_issues.json"


# ---------------- universe ----------------

def get_universe(cfg: dict, refresh: bool = False) -> pd.DataFrame:
    """Danh sách symbol + exchange, cache Parquet để tránh gọi lại nhiều lần."""
    path = symbols_path(cfg)
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
        log.info("Universe nạp từ cache: %d mã", len(df))
        return df
    df = vns.fetch_symbol_list(cfg)
    if df.empty and yah.has_yfinance():
        log.warning("vnstock không trả về universe - KHÔNG tự tạo danh sách mã qua Yahoo "
                    "(không đáng tin). Hãy cài vnstock hoặc cung cấp file symbols thủ công.")
    if not df.empty:
        write_parquet(df, path)
    return df


def load_sectors(cfg: dict) -> pd.Series:
    """symbol -> sector. Thiếu thì trả Series rỗng; mọi mã nhận sector='UNKNOWN'."""
    df = read_parquet(sectors_path(cfg))
    if df.empty:
        return pd.Series(dtype=str)
    return df.set_index("symbol")["sector"].astype(str).str.upper()


# ---------------- download ----------------

def _is_fresh(sym: str, end: str, cfg: dict) -> bool:
    """Cache còn dùng được nếu phiên cuối >= end - 4 ngày."""
    path = raw_daily_path(sym, cfg)
    if not path.exists():
        return False
    try:
        last = pd.read_parquet(path, columns=["date"])["date"].max()
    except Exception:
        return False
    if pd.isna(last):
        return False
    gap = (pd.Timestamp(end) - pd.Timestamp(last)).days
    return gap <= 4


def download_symbol(sym: str, start: str, end: str, cfg: dict,
                    force: bool = False) -> tuple[pd.DataFrame, str]:
    """Tải OHLCV một mã: vnstock trước, yfinance fallback. Trả về (df_clean, source)."""
    if not force and _is_fresh(sym, end, cfg):
        df = pd.read_parquet(raw_daily_path(sym, cfg))
        src = str(df["source"].iloc[-1]) if "source" in df.columns and len(df) else "cache"
        return df, src

    df = pd.DataFrame()
    source = ""
    if vns.has_vnstock():
        try:
            raw = vns.fetch_history_vnstock(sym, start, end, cfg)
            df = clean_ohlcv(raw, source="vnstock")
            df = df[df["symbol"] == sym.upper()]
        except RuntimeError as e:
            log.warning("%s: vnstock thất bại (%s), thử fallback yfinance", sym, e)
            df = pd.DataFrame()
        if not df.empty:
            source = "vnstock"

    if df.empty and cfg_get(cfg, "data.use_yfinance_fallback", True) and yah.has_yfinance():
        try:
            raw = yah.fetch_history_yahoo(sym, start, end, cfg)
            df = clean_ohlcv(raw, source="yfinance")
            # yfinance giá tuyệt đối VND -> đồng bộ thang với vnstock (nghìn đồng)
            mult = float(cfg_get(cfg, "universe.price_unit_multiplier", 1000)) or 1000.0
            med = float(df["close"].median()) if len(df) else 0.0
            if med > 500 * mult:  # rõ ràng là đơn vị đồng
                for c in ["open", "high", "low", "close"]:
                    df[c] = df[c] / mult
                df["turnover"] = pd.NA
            source = "yfinance"
        except Exception as e:  # noqa: BLE001
            log.warning("%s: yfinance cũng lỗi (%s) - ghi nhận missing, không chặn pipeline", sym, e)

    if df.empty:
        log.warning("%s: KHÔNG có dữ liệu từ mọi nguồn (missing_data)", sym)
        return pd.DataFrame(), "none"
    df = df[(pd.to_datetime(df["date"]) >= pd.Timestamp(start)) &
            (pd.to_datetime(df["date"]) <= pd.Timestamp(end))]
    write_parquet(df, raw_daily_path(sym, cfg))
    time.sleep(float(cfg_get(cfg, "data.request_sleep_seconds", 0.35)))
    return df, source


def download_benchmark(cfg: dict, start: str, end: str,
                       daily_all: pd.DataFrame | None = None,
                       force: bool = False) -> pd.DataFrame:
    """VNIndex: vnstock -> yahoo -> synthetic EW top-thanh-khoản (ghi rõ source)."""
    path = index_path("VNINDEX", cfg)
    if path.exists() and not force:
        df = pd.read_parquet(path)
        if not df.empty and pd.to_datetime(df["date"]).max() >= pd.Timestamp(end) - timedelta(days=6):
            return df

    df = pd.DataFrame()
    if vns.has_vnstock():
        try:
            raw = vns.fetch_index_vnstock("VNINDEX", start, end, cfg)
            raw = raw.copy()
            raw["symbol"] = "VNINDEX"
            df = clean_ohlcv(raw, source="vnstock")
        except RuntimeError as e:
            log.warning("VNIndex vnstock lỗi: %s", e)
        if not df.empty:
            df["close"] = pd.to_numeric(df["close"], errors="coerce")

    if df.empty and cfg_get(cfg, "data.use_yfinance_fallback", True) and yah.has_yfinance():
        try:
            raw = yah.fetch_index_yahoo("^VNINDEX", start, end, cfg)
            if not raw.empty:
                raw = raw.copy()
                raw["symbol"] = "VNINDEX"
                df = clean_ohlcv(raw, source="yfinance")
        except Exception as e:  # noqa: BLE001
            log.warning("VNIndex yfinance lỗi: %s", e)

    if df.empty and daily_all is not None and not daily_all.empty:
        log.warning("KHÔNG lấy được VNIndex từ vnstock/yfinance -> "
                    "tự xây benchmark tổng hợp equal-weight từ top cổ phiếu thanh khoản "
                    "(source=synthetic_ew_topN). Tín hiệu RS chỉ mang tính tương đối.")
        df = build_synthetic_benchmark(daily_all, cfg)

    if not df.empty:
        write_parquet(df, path)
    else:
        log.error("Không có benchmark nào khả dụng - RS/regime sẽ bị đánh dấu missing_data")
    return df


def build_synthetic_benchmark(daily_all: pd.DataFrame, cfg: dict, top_n: int = 20) -> pd.DataFrame:
    """Benchmark equal-weight từ top N mã ADV cao nhất, chuẩn hoá về mốc 100."""
    d = daily_all.copy()
    d["turn_est"] = d["volume"] * d["close"]
    adv = d.groupby("symbol")["turn_est"].tail(20).groupby(d["symbol"]).mean().sort_values()
    picks = list(adv.index[-top_n:])
    px = d[d["symbol"].isin(picks)].pivot_table(index="date", columns="symbol", values="close")
    ret = px.pct_change(fill_method=None).mean(axis=1)  # EW daily return
    level = (1 + ret.fillna(0)).cumprod() * 100.0
    out = pd.DataFrame({
        "date": level.index.date, "symbol": "VNINDEX",
        "open": level.values, "high": level.values, "low": level.values,
        "close": level.values, "volume": 0.0, "turnover": pd.NA,
        "source": f"synthetic_ew_top{top_n}",
    })
    return out.dropna(subset=["close"])


def run_download(cfg: dict | None = None, start: str | None = None,
                 end: str | None = None, limit: int | None = None,
                 max_symbols: int | None = None, force: bool = False) -> dict:
    """Tải universe + OHLCV + benchmark + sectors, gộp vào processed/daily/daily_all.parquet."""
    cfg = cfg or load_config()
    start = start or str(cfg_get(cfg, "data.start_date", "2022-01-01"))
    end = end or str(date.today())
    root = data_root(cfg)
    root.joinpath("raw", "daily").mkdir(parents=True, exist_ok=True)

    uni = get_universe(cfg)
    if uni.empty:
        msg = ("Không lấy được danh sách cổ phiếu (vnstock chưa cài hoặc lỗi mạng). "
               "Cài `pip install vnstock` rồi chạy lại, hoặc dùng `python -m app.cli demo-data` "
               "để tạo bộ dữ liệu mẫu OFFLINE phục vụ kiểm thử pipeline.")
        log.error(msg)
        raise SystemExit(msg)

    syms = sorted(uni["symbol"].unique())
    if limit:
        syms = syms[:limit]
    elif max_symbols:
        syms = syms[:max_symbols]

    sector_map = load_sectors(cfg)
    if sector_map.empty and vns.has_vnstock():
        sdf = vns.fetch_sectors_vnstock(cfg)
        if not sdf.empty:
            write_parquet(sdf, sectors_path(cfg))
            sector_map = sdf.set_index("symbol")["sector"]

    frames, ok, failed = [], 0, []
    for i, sym in enumerate(syms, 1):
        df, src = download_symbol(sym, start, end, cfg, force=force)
        if not df.empty:
            frames.append(df)
            ok += 1
        else:
            failed.append(sym)
        if i % 25 == 0 or i == len(syms):
            log.info("Download %d/%d (ok=%d, fail=%d)", i, len(syms), ok, len(failed))

    if not frames:
        raise SystemExit("Không tải được dữ liệu OHLCV nào - dừng pipeline (không bịa dữ liệu).")
    daily = pd.concat(frames, ignore_index=True)
    daily = daily.sort_values(["symbol", "date"]).reset_index(drop=True)
    write_parquet(daily, processed_daily_path(cfg))

    bench = download_benchmark(cfg, start, end, daily_all=daily, force=force)
    meta = {
        "downloaded_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "start": start, "end": end,
        "n_symbols_requested": len(syms), "n_symbols_ok": ok,
        "failed_symbols": failed[:200],
        "benchmark_source": str(bench["source"].iloc[-1]) if len(bench) else "NONE",
        "n_sectors": int(len(sector_map)),
    }
    (root / "raw").mkdir(exist_ok=True)
    with open(root / "raw" / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log.info("Hoàn tất download: %s", meta["n_symbols_ok"]); 
    return meta


# ---------------- load ----------------

def load_daily(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = read_parquet(processed_daily_path(cfg))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_benchmark(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = read_parquet(index_path("VNINDEX", cfg))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def latest_trading_date(cfg: dict | None = None) -> pd.Timestamp:
    """Ngày giao dịch gần nhất có dữ liệu đã ĐÓNG CỬA (as_of mặc định)."""
    df = load_daily(cfg)
    if df.empty:
        raise SystemExit("Chưa có dữ liệu - chạy `python -m app.cli download` trước.")
    return pd.Timestamp(df["date"].max())

