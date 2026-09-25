"""Các chỉ báo kỹ thuật daily (PHẦN 3). KHÔNG dùng dữ liệu tương lai.

Mọi hàm nhận Series/DataFrame đã sắp xếp tăng theo date và CHỈ tính tại ngày cuối
bằng dữ liệu <= ngày đó (rolling/ewm nhân quả). Giá trị tại index i không phụ thuộc
các dòng sau i -> kiểm chứng bằng test_no_lookahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import cfg_get
from app.features.mcdx import mcdx


# ---------------- moving averages ----------------

def sma(s: pd.Series, period: int) -> pd.Series:
    return pd.Series(s).astype(float).rolling(period, min_periods=period).mean()


def ema(s: pd.Series, period: int) -> pd.Series:
    return pd.Series(s).astype(float).ewm(span=period, adjust=False).mean()


# ---------------- volatility ----------------

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR Wilder. df cần high/low/close."""
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ---------------- oscillators ----------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    c = pd.Series(close).astype(float)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(avg_loss.notna(), np.nan)
    out[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    out[(avg_loss == 0) & (avg_gain == 0)] = 50.0
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    c = pd.Series(close).astype(float)
    macd_line = ema(c, fast) - ema(c, slow)
    sig = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig
    return macd_line, sig, hist


# ---------------- volume ----------------

def volume_stats(df: pd.DataFrame, ma_period: int = 20):
    v = df["volume"].astype(float)
    vma = v.rolling(ma_period, min_periods=ma_period).mean()
    vsd = v.rolling(ma_period, min_periods=ma_period).std()
    rvol = v / vma.replace(0, np.nan)
    vz = (v - vma) / vsd.replace(0, np.nan)
    return vma, rvol, vz


def obv(df: pd.DataFrame) -> pd.Series:
    c, v = df["close"].astype(float), df["volume"].astype(float)
    direction = np.sign(c.diff()).fillna(0.0)
    return (direction * v).cumsum()


def slope(s: pd.Series, window: int) -> pd.Series:
    """Độ dốc chuẩn hoá theo std của chính chuỗi cửa sổ (nhân quả)."""
    x = pd.Series(s).astype(float)
    win = x.rolling(window, min_periods=max(3, window // 2))
    # cov(x, t)/var(t) với t = 0..window-1
    t = np.arange(window)
    tv = t.var()
    def _cov(arr):
        return float(np.dot(arr - arr.mean(), t - t.mean()) / window / tv) if len(arr) == window else np.nan
    return win.apply(_cov, raw=True) / x.rolling(window, min_periods=max(3, window // 2)).std().replace(0, np.nan)


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    h, l, c, v = (df[k].astype(float) for k in ["high", "low", "close", "volume"])
    rng = (h - l).replace(0, np.nan)
    mfm = ((c - l) - (h - c)) / rng
    mfv = mfm * v
    return mfv.rolling(period, min_periods=period).sum() / v.rolling(period, min_periods=period).sum().replace(0, np.nan)


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    mf = tp * df["volume"]
    pos = mf.where(tp > tp.shift(1), 0.0)
    neg = mf.where(tp < tp.shift(1), 0.0)
    ratio = pos.rolling(period).sum() / neg.rolling(period).sum().replace(0, np.nan)
    return 100 - 100 / (1 + ratio)


# ---------------- swing structure (fractal, nhân quả) ----------------

def fractal_swings(df: pd.DataFrame, k: int = 2):
    """Trả về (swing_high_idx, swing_low_idx): mảng boolean đánh dấu nến confirmed.

    Nến i được XÁC NHẬN là swing high tại ngày i+k (cần k nến sau) -> khi lấy tín hiệu
    ngày as_of, chỉ swing có index <= len-1-k mới xuất hiện; không lookahead.
    """
    h = df["high"].to_numpy(float); l = df["low"].to_numpy(float)
    n = len(df)
    sh = np.zeros(n, dtype=bool); sl = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        if h[i] == max(h[i - k:i + k + 1]) and h[i] > max(h[i - k], h[i + k]):
            sh[i] = True
        if l[i] == min(l[i - k:i + k + 1]) and l[i] < min(l[i - k], l[i + k]):
            sl[i] = True
    return sh, sl


def detect_range(df: pd.DataFrame, cfg: dict) -> dict:
    """Xác định range tích lũy từ các swing đã confirmed trong lookback.

    Trả về dict: support, resistance, range_height, range_duration, n_test_support,
                 last_swing_low_idx/date, prev_swing_low_idx/date, higher_low(bool)...
    """
    icfg = cfg_get(cfg, "indicators", {})
    scfg = cfg_get(cfg, "signals", {})
    k = int(icfg.get("fractal_k", 2))
    lookback = int(icfg.get("swing_lookback", 60))
    sh, sl = fractal_swings(df, k)
    idx = np.arange(len(df))
    lo_lim = max(0, len(df) - lookback)
    low_i = idx[sl & (idx >= lo_lim)]
    high_i = idx[sh & (idx >= lo_lim)]
    g = df.reset_index(drop=True)

    out = {"support": np.nan, "resistance": np.nan, "range_height": np.nan,
           "range_duration": np.nan, "n_test_support": 0,
           "higher_low": False, "last_swing_low": None, "prev_swing_low": None,
           "swing_highs": [], "swing_lows": []}
    if len(low_i) >= 1:
        lows = [float(g["low"].iloc[i]) for i in low_i]
        # support = vùng đáy cluster: median 2-3 swing low gần nhất
        recent_lows = lows[-3:]
        out["support"] = float(np.median(recent_lows))
        tol = float(scfg.get("support_atr_tolerance", 1.0))
        atr_last = float(g["low"].tail(15).mean() * 0) or None  # placeholder guard
        # số lần test support: số phiên low chạm trong dung sai (dùng ATR bên ngoài truyền vào nếu có)
        out["last_swing_low"] = int(low_i[-1])
        if len(low_i) >= 2:
            out["prev_swing_low"] = int(low_i[-2])
            out["higher_low"] = float(g["low"].iloc[low_i[-1]]) > float(g["low"].iloc[low_i[-2]])
        out["swing_lows"] = [(int(i), float(g["low"].iloc[i]), str(g["date"].iloc[i])) for i in low_i]
    if len(high_i) >= 1:
        highs = [float(g["high"].iloc[i]) for i in high_i[-4:]]
        out["resistance"] = float(np.median(highs))
        out["swing_highs"] = [(int(i), float(g["high"].iloc[i]), str(g["date"].iloc[i])) for i in high_i]
    if not np.isnan(out["support"]) and not np.isnan(out["resistance"]):
        if out["resistance"] <= out["support"]:
            out["resistance"] = float(g["high"].tail(lookback).max())
        out["range_height"] = out["resistance"] - out["support"]
        first = min([i for i, _, _ in out["swing_lows"]] + [i for i, _, _ in out["swing_highs"]]) \
            if (out["swing_lows"] or out["swing_highs"]) else None
        if first is not None:
            out["range_duration"] = int(len(g) - 1 - first)
        near = g["low"] <= out["support"] * 1.01
        out["n_test_support"] = int(near.tail(max(int(scfg.get('range_min_bars', 20)), 1)).sum())
    return out


# ---------------- assembly ----------------

def compute_indicators(g: pd.DataFrame, cfg: dict,
                       bench_close: pd.Series | None = None) -> pd.DataFrame:
    """Tính toàn bộ chỉ báo cho DataFrame OHLCV MỘT mã (đã sort date tăng).

    bench_close: Series close benchmark index theo cùng index positional (đã align).
    Trả về DataFrame copy thêm các cột chỉ báo.
    """
    icfg = cfg_get(cfg, "indicators", {})
    out = g.reset_index(drop=True).copy()
    c = out["close"].astype(float)

    out["sma20"] = sma(c, int(icfg.get("sma_period", 20)))
    out["ema20"] = ema(c, int(icfg.get("ema_short", 20)))
    out["ema50"] = ema(c, int(icfg.get("ema_mid", 50)))
    out["ema200"] = ema(c, int(icfg.get("ema_long", 200)))
    out["atr14"] = atr(out, int(icfg.get("atr_period", 14)))
    out["rsi14"] = rsi(c, int(icfg.get("rsi_period", 14)))

    fast = int(icfg.get("macd_fast", 12)); slow = int(icfg.get("macd_slow", 26))
    sigp = int(icfg.get("macd_signal", 9))
    m, s, hist = macd(c, fast, slow, sigp)
    out["macd"], out["macd_signal"], out["macd_hist"] = m, s, hist
    out["macd_hist_slope"] = hist.diff()

    out["mcdx"] = mcdx(c, atr=out["atr14"], period=int(icfg.get("mcdx", {}).get("period", 14)),
                       mode=str(icfg.get("mcdx", {}).get("mode", "macd_hist_slope_proxy")),
                       macd_hist=hist)

    vma, rvol, vz = volume_stats(out, int(icfg.get("volume_ma_period", 20)))
    out["volume_ma20"] = vma
    out["rvol"] = rvol
    out["volume_zscore_20"] = vz
    out["obv"] = obv(out)
    wsp = int(icfg.get("obv_slope_period", 10))
    out["obv_slope"] = slope(out["obv"], wsp)
    out["cmf20"] = cmf(out, int(icfg.get("cmf_period", 20)))
    out["mfi14"] = mfi(out, int(icfg.get("mfi_period", 14)))

    # Relative strength so với benchmark (chỉ dùng dữ liệu <= ngày hiện tại)
    if bench_close is not None and len(bench_close) == len(out):
        b = pd.Series(np.asarray(bench_close, dtype=float), index=out.index)
        rs = c / b.replace(0, np.nan)
        out["rs"] = rs
        out["rs_slope_10"] = slope(rs, 10)
        w = int(icfg.get("rs_percentile_window", 250))
        out["rs_percentile"] = rs.rolling(w, min_periods=40).apply(
            lambda x: (x[-1] >= np.median(x)) * 50.0 + (np.sum(x < x[-1]) / max(len(x) - 1, 1)) * 50.0,
            raw=True)
    else:
        out["rs"] = np.nan
        out["rs_slope_10"] = np.nan
        out["rs_percentile"] = np.nan

    # biến thể volume của nhịp giảm gần nhất (để so sánh dần cạn)
    dn = out["close"].diff() < 0
    dv = out["volume"].where(dn)
    out["downvol_ma10"] = dv.rolling(10, min_periods=3).mean()
    out["downvol_ma10_prev"] = out["downvol_ma10"].shift(10)
    return out
