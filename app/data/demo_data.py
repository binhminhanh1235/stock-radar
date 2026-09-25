"""Bộ dữ liệu MẪU SINH TỰ (synthetic) - CHỈ dùng để kiểm thử pipeline khi KHÔNG có
mạng/nguồn dữ liệu thật. Đây là dữ liệu GIẢ LẬP được GHI NHÃN RÕ RÀNG
(source='demo_synthetic'), không phải dữ liệu thị trường thật, không dùng để ra quyết định.

Kịch bản dựng sẵn để test bộ nhận diện Wyckoff/VPA:
- SPR01  : nền tích lũy + spring ngày cuối cùng (low thủng support, đóng cửa hồi lên)
- SHK01  : shakeout đâm thủng support volume cao rồi hồi
- TST01  : spring cách đây ~6 phiên + nhịp test volume cạn -> TEST_READY/CONFIRMED
- ACC01  : giảm sâu rồi tạo đáy sau cao hơn + RSI hồi -> EARLY_ACCUMULATION
- SOS01  : breakout qua resistance volume cao + pullback giữ -> CONFIRMED_BUY_CANDIDATE
- BRK01  : gãy nền volume bán cao, đóng cửa dưới support -> AVOID
- LOW01  : thanh khoản rất thấp -> bị loại vì ADV
- BAD01  : nhiều phiên giá lỗi/thiếu -> data_issues
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import cfg_get, load_config
from app.data.cleaning import clean_ohlcv
from app.logging_utils import get_logger
from app.storage import write_parquet

log = get_logger(__name__)

PRICE_MULT_DEFAULT = 1000.0


def _dates(start: str, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _base_ohlcv(dates, close_path, rng, vol_base):
    close = np.asarray(close_path, dtype=float)[:len(dates)]
    n = len(close)
    dates = dates[:n]
    open_ = np.empty(n); high = np.empty(n); low = np.empty(n); vol = np.empty(n)
    open_[0] = close[0] * (1 + rng.normal(0, .003))
    for i in range(1, n):
        gap = rng.normal(0, .004)
        open_[i] = min(max(close[i - 1] * (1 + gap), min(close[i], close[i-1]) * .985),
                       max(close[i], close[i-1]) * 1.015)
    body_hi = np.maximum(open_, close); body_lo = np.minimum(open_, close)
    hi_wick = np.abs(rng.normal(0, .004, n)) * close
    lo_wick = np.abs(rng.normal(0, .004, n)) * close
    high = body_hi + hi_wick
    low = body_lo - lo_wick
    ret = np.diff(close, prepend=close[0]) / np.maximum(close, 1e-9)
    vol = vol_base * np.exp(rng.normal(0, .25, n)) * (1 + 8 * np.abs(ret))
    df = pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low,
                       "close": close, "volume": np.round(vol)})
    return df


def _spring_series(rng, n=260):
    """Giảm từ 30 về 21, đi ngang 23 phiên với biên hẹp, phiên cuối ĐÂM THỦNG support rồi đóng hồi."""
    d = _dates("2024-01-02", n)
    leg1 = np.linspace(30, 21.0, 120) + rng.normal(0, .18, 120)
    base = 21.0
    side = base + np.cumsum(rng.normal(0, .07, n - 120)) * .5
    side = np.clip(side, base - .45, base + 1.4)
    close = np.concatenate([leg1, side])[:n]
    # phiên cuối: spring
    support = float(np.min(close[-24:-1]))
    close[-1] = support + 0.15          # đóng cửa HỒI LÊN trên support
    df = _base_ohlcv(d, close, rng, 6_000_000)
    df.loc[df.index[-1], ["low"]] = support - 0.28   # bóng dưới dài qua support
    df.loc[df.index[-1], "high"] = max(df["high"].iloc[-1], close[-1] + .1)
    df.loc[df.index[-1], "volume"] = 4_500_000       # volume thấp khi xuyên (cạn cung)
    # những phiên gần support volume cạn dần
    df.loc[df.index[-6:-1], "volume"] = np.linspace(5.2e6, 3.4e6, 5)
    return df


def _shakeout_series(rng, n=260):
    d = _dates("2024-01-02", n)
    leg1 = np.linspace(32, 22.0, 120) + rng.normal(0, .2, 120)
    base = 22.0
    side = np.clip(base + np.cumsum(rng.normal(0, .06, n - 120)) * .4, base - .4, base + 1.5)
    close = np.concatenate([leg1, side])
    df = _base_ohlcv(d, close[:n], rng, 7_000_000)
    support = float(np.min(df["low"].iloc[-25:-3]))
    i = n - 3
    df.loc[df.index[i], ["low", "close", "volume"]] = [support - 0.55, support + 0.25, 16_500_000]
    df.loc[df.index[i], "high"] = support + 0.4
    df.loc[df.index[i + 1], "close"] = support + 0.5
    df.loc[df.index[i + 2], "close"] = support + 0.62
    return df


def _test_after_spring_series(rng, n=260):
    d = _dates("2024-01-02", n)
    leg1 = np.linspace(29, 20.5, 115) + rng.normal(0, .18, 115)
    side = np.clip(20.5 + np.cumsum(rng.normal(0, .05, n - 115)) * .4, 20.1, 21.9)
    close = np.concatenate([leg1, side])[:n]
    df = _base_ohlcv(d, close, rng, 6_500_000)
    support = 20.15
    si = n - 7                                   # spring 6 phiên trước
    df.loc[df.index[si], ["low", "close", "volume"]] = [support - 0.30, support + 0.25, 12_000_000]
    ti = n - 2                                   # phiên test volume cạn
    df.loc[df.index[ti], ["low", "close", "open", "volume"]] = [support + 0.05, support + 0.35, support + 0.12, 3_200_000]
    df.loc[df.index[n - 1], ["low", "close", "open", "volume"]] = [support + 0.15, support + 0.42, support + 0.2, 3_600_000]
    return df


def _accumulation_series(rng, n=260):
    d = _dates("2024-01-02", n)
    leg1 = np.linspace(34, 22.0, 130) + rng.normal(0, .22, 130)
    bottom = 22.0 + np.cumsum(rng.normal(.005, .06, n - 130)) * .6
    close = np.concatenate([leg1, np.clip(bottom, 21.6, 25)])[:n]
    df = _base_ohlcv(d, close, rng, 7_000_000)
    # selling climax: phiên giảm mạnh volume cực đại, sau đó không giảm thêm
    ci = 128
    df.loc[df.index[ci], ["close", "low", "volume"]] = [21.7, 21.3, 24_000_000]
    # volume các nhịp giảm sau giảm dần
    df.loc[df.index[ci + 5:ci + 12], "volume"] = np.linspace(9e6, 4.2e6, 7)
    return df


def _sos_pullback_series(rng, n=260):
    d = _dates("2024-01-02", n)
    leg1 = np.linspace(30, 21.0, 110) + rng.normal(0, .18, 110)
    side = np.clip(21.0 + np.cumsum(rng.normal(.004, .05, n - 110)) * .5, 20.8, 23.6)
    close = np.concatenate([leg1, side])[:n]
    df = _base_ohlcv(d, close, rng, 6_800_000)
    res = float(np.max(df["high"].iloc[-130:-25]))
    bi = n - 12                                  # SOS breakout
    df.loc[df.index[bi], ["close", "high", "volume"]] = [res + 0.9, res + 1.05, 18_500_000]
    # pullback volume thấp giữ trên vùng breakout
    pb_close = [res + 0.55, res + 0.35, res + 0.30, res + 0.45, res + 0.60,
                res + 0.50, res + 0.58, res + 0.72, res + 0.85]
    for j, cval in enumerate(pb_close):
        idx = bi + 1 + j
        if idx < n:
            df.loc[df.index[idx], "close"] = cval
            df.loc[df.index[idx], "volume"] = 4_000_000 + j * 300_000
            df.loc[df.index[idx], "low"] = min(cval - .1, df["low"].iloc[idx])
            df.loc[df.index[idx], "high"] = max(cval + .12, df["high"].iloc[idx])
    return df


def _broken_series(rng, n=260):
    d = _dates("2024-01-02", n)
    leg1 = np.linspace(30, 21.5, 120) + rng.normal(0, .18, 120)
    side = np.clip(21.5 + np.cumsum(rng.normal(0, .05, 100)) * .4, 21.2, 22.4)
    crash = np.linspace(21.3, 18.2, n - 220)
    close = np.concatenate([leg1, side, crash])[:n]
    df = _base_ohlcv(d, close, rng, 7_000_000)
    for k in range(n - 18, n - 4):               # chuỗi bán tháo volume cao
        df.loc[df.index[k], "volume"] = 20_000_000 + (k % 3) * 2_000_000
        df.loc[df.index[k], "close"] = min(df["close"].iloc[k], df["low"].iloc[k] + 0.05)
    return df


def build_demo_frames(cfg: dict) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(42)
    frames = {
        "SPR01": _spring_series(rng),
        "SHK01": _shakeout_series(rng),
        "TST01": _test_after_spring_series(rng),
        "ACC01": _accumulation_series(rng),
        "SOS01": _sos_pullback_series(rng),
        "BRK01": _broken_series(rng),
    }
    # LOW01: thanh khoản bỏ đi
    d = _dates("2024-01-02", 260)
    px = 12 + np.cumsum(rng.normal(0, .04, 260))
    f = _base_ohlcv(d, px, rng, 20_000)
    frames["LOW01"] = f
    # BAD01: vài nến lỗi để test cleaning
    f2 = _base_ohlcv(d, 15 + np.cumsum(rng.normal(0, .05, 260)), rng, 3_000_000).copy()
    f2.loc[f2.index[100], "low"] = f2["high"].iloc[100] + 5     # high < low -> bị loại
    f2.loc[f2.index[101], "close"] = -3                          # giá <= 0 -> bị loại
    f2 = f2.drop(index=list(range(150, 175)))                    # thiếu 25 phiên liên tiếp
    frames["BAD01"] = f2

    out = {}
    for sym, f in frames.items():
        f = f.copy()
        f["symbol"] = sym
        f["turnover"] = f["volume"] * f["close"] * PRICE_MULT_DEFAULT
        out[sym] = clean_ohlcv(f, source="demo_synthetic")
    return out


def build_demo_benchmark(cfg: dict, start: str = "2024-01-02", n: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    d = _dates(start, n)
    level = 1200 * np.cumprod(1 + rng.normal(0.0002, 0.008, n))
    df = pd.DataFrame({"date": d, "symbol": "VNINDEX", "open": level, "high": level * 1.003,
                       "low": level * 0.997, "close": level, "volume": 0})
    df["turnover"] = np.nan
    return clean_ohlcv(df, source="demo_synthetic")


def run_demo(cfg: dict | None = None) -> dict:
    """Ghi dữ liệu mẫu vào cache Parquet theo đúng cấu trúc pipeline."""
    cfg = cfg or load_config()
    from app.data import pipeline as pl
    frames = build_demo_frames(cfg)
    daily = pd.concat(frames.values(), ignore_index=True)
    daily = daily.sort_values(["symbol", "date"]).reset_index(drop=True)
    write_parquet(daily, pl.processed_daily_path(cfg))
    for sym, df in frames.items():
        write_parquet(df, pl.raw_daily_path(sym, cfg))
    uni = pd.DataFrame([{"symbol": s, "exchange": "DEMO"} for s in frames])
    write_parquet(uni, pl.symbols_path(cfg))
    bench = build_demo_benchmark(cfg)
    write_parquet(bench, pl.index_path("VNINDEX", cfg))
    meta = {"downloaded_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "start": str(daily["date"].min()), "end": str(daily["date"].max()),
            "n_symbols_requested": len(frames), "n_symbols_ok": len(frames),
            "failed_symbols": [], "benchmark_source": "demo_synthetic", "n_sectors": 0,
            "NOTE": "DỮ LIỆU MẪU SINH TỰ - chỉ để kiểm thử pipeline, không phải dữ liệu thật"}
    import json
    with open(pl.data_root(cfg) / "raw" / "meta.json", "w", encoding="utf-8") as fjson:
        json.dump(meta, fjson, ensure_ascii=False, indent=2)
    log.warning("Đã ghi DEMO DATA (synthetic) - chỉ dùng để kiểm thử end-to-end offline.")
    return meta
