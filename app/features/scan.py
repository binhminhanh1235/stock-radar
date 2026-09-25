"""Quét tín hiệu sớm toàn thị trường + lưu Parquet (PHẦN 4 driver).

BẤT ĐẲNG THỨC QUAN TRỌNG: dữ liệu đưa vào classify_symbol luôn được cắt
date <= as_of_date -> tín hiệu ngày D không bao giờ thấy ngày D+1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import cfg_get, load_config
from app.data.cleaning import per_symbol_quality
from app.data.pipeline import data_root, load_benchmark, load_daily
from app.features.early_wyckoff import Signal, classify_symbol
from app.features.indicators import compute_indicators
from app.logging_utils import get_logger
from app.storage import write_parquet

log = get_logger(__name__)


def market_regime(bench: pd.DataFrame, cfg: dict) -> tuple[str, float | None, list[str]]:
    """BULLISH / NEUTRAL / BEARISH / UNKNOWN từ benchmark (chỉ dùng dữ liệu <= as_of)."""
    missing: list[str] = []
    if bench is None or bench.empty:
        missing.append("benchmark")
        return "UNKNOWN", None, ["thiếu benchmark - market regime = UNKNOWN"]
    b = bench.copy()
    close = b["close"].astype(float)
    ema50 = close.ewm(span=int(cfg_get(cfg, "indicators.ema_mid", 50)), adjust=False).mean()
    ema200 = close.ewm(span=int(cfg_get(cfg, "indicators.ema_long", 200)), adjust=False).mean()
    c, e50, e200 = float(close.iloc[-1]), float(ema50.iloc[-1]), float(ema200.iloc[-1])
    above50, above200 = c > e50, c > e200
    if len(close) < 200:
        missing.append("benchmark < 200 phiên (EMA200 kém tin cậy)")
    if above50 and above200:
        regime = "BULLISH"
    elif above50 or above200:
        regime = "NEUTRAL"
    else:
        regime = "BEARISH"
    note = f"VNIndex={c:.2f} ({'trên' if above50 else 'dưới'} EMA50, {'trên' if above200 else 'dưới'} EMA200)"
    return regime, c, [note] + missing


def scan_all(as_of_date: str | pd.Timestamp, cfg: dict | None = None,
             symbols: list[str] | None = None) -> tuple[list[Signal], dict]:
    """Trả về danh sách Signal cho mọi mã đủ điều kiện + metadata."""
    cfg = cfg or load_config()
    as_of = pd.Timestamp(as_of_date)
    daily = load_daily(cfg)
    if daily.empty:
        raise SystemExit("Chưa có dữ liệu OHLCV - chạy `python -m app.cli download` hoặc demo-data trước.")
    daily = daily[daily["date"] <= as_of]                      # CẮT lookahead tại đây
    if daily.empty:
        raise SystemExit(f"Không có dữ liệu tính đến {as_of.date()}")
    bench = load_benchmark(cfg)
    bench_aligned = None
    if not bench.empty:
        bench = bench[bench["date"] <= as_of]

    quality = per_symbol_quality(daily, cfg, as_of)
    uni = pd.read_parquet(data_root(cfg) / "raw" / "symbols" / "symbols.parquet") \
        if (data_root(cfg) / "raw" / "symbols" / "symbols.parquet").exists() else pd.DataFrame()
    exch_map = dict(zip(uni.get("symbol", []), uni.get("exchange", ""))) if "symbol" in uni.columns else {}
    from app.data.pipeline import load_sectors
    sector_map = load_sectors(cfg)

    exclude_ll = bool(cfg_get(cfg, "universe.exclude_low_liquidity", True))
    signals: list[Signal] = []
    data_issues: list[dict] = []
    syms = symbols or sorted(daily["symbol"].unique())
    for sym in syms:
        g = daily[daily["symbol"] == sym].copy()
        if g.empty:
            continue
        q = quality.loc[sym].to_dict() if sym in quality.index else {"ok": False, "issues": ["không có trong quality table"]}
        if q.get("issues") and exclude_ll and any("thanh khoản" in i or "phiên" in i for i in q["issues"]):
            data_issues.append({"symbol": sym, "issues": q["issues"], "action": "EXCLUDED"})
            continue
        gb = g.sort_values("date").reset_index(drop=True)
        if len(gb) < int(cfg_get(cfg, "universe.require_history_bars", 120)):
            data_issues.append({"symbol": sym, "issues": [f"lịch sử {len(gb)} phiên"], "action": "EXCLUDED_SHORT_HISTORY"})
            continue
        bc = None
        if not bench.empty:
            bb = bench.reindex(bench["date"].searchsorted(gb["date"]) - 1, method=None)
            aligned = bench.set_index("date")["close"].reindex(gb["date"]).ffill()
            bc = aligned.to_numpy()
        ind = compute_indicators(gb, cfg, bench_close=bc)
        sig = classify_symbol(ind, sym, q, cfg, exchange=exch_map.get(sym, ""),
                              sector=str(sector_map.get(sym, "UNKNOWN")))
        signals.append(sig)
        if q.get("warnings"):
            pass
    meta = {"as_of_date": str(as_of.date()), "n_scanned": len(syms), "n_signals": len(signals),
            "data_issues": data_issues,
            "benchmark_source": str(bench["source"].iloc[-1]) if not bench.empty else "NONE"}
    log.info("Scan %s: %d mã, %d signal, %d mã có vấn đề dữ liệu",
             meta["as_of_date"], len(syms), len(signals), len(data_issues))
    return signals, meta


def save_signals(signals: list[Signal], as_of_date: str, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    rows = [s.to_dict() for s in signals]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["reasons"] = df["reasons"].map(lambda x: "; ".join(x) if isinstance(x, list) else str(x))
        df["missing_data"] = df["missing_data"].map(lambda x: "; ".join(x) if isinstance(x, list) else str(x))
    path = data_root(cfg) / "processed" / "signals" / f"signals_{as_of_date}.parquet"
    write_parquet(df, path)
    return df
