"""MCDX - indicator có thể cấu hình, HIỆN LÀ PLACEHOLDER.

Người dùng chưa cung cấp công thức MCDX chuẩn xác. Theo thoả thuận (PHẦN 3.5):
mặc định dùng proxy = độ dốc của MACD histogram đã làm trơn (chuẩn hoá ATR),
có thể đổi sang các mode khác trong config.indicators.mcdx.mode:
  - macd_hist_slope_proxy : slope của MACD histogram (mặc định)
  - volume_momentum       : momentum giá * rvol (VPA-style)
  - tsi_lite              : double-EMA của ROC (Total-Signal-Index rút gọn)

TODO: thay bằng công thức MCDX chính thức khi người dùng cung cấp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mcdx(close: pd.Series, atr: pd.Series | None = None, period: int = 14,
         mode: str = "macd_hist_slope_proxy",
         macd_hist: pd.Series | None = None) -> pd.Series:
    """Trả về chuỗi MCDX (proxy). Càng dương => động lượng càng tích cực."""
    close = pd.Series(close).astype(float)
    if mode == "volume_momentum":
        roc = close.pct_change(period, fill_method=None)
        return roc.rolling(3, min_periods=1).mean() * 100.0

    if mode == "tsi_lite":
        roc = close.diff()
        s1 = roc.ewm(span=period, adjust=False).mean()
        s2 = s1.ewm(span=int(period / 2) + 1, adjust=False).mean()
        sd = roc.abs().ewm(span=period, adjust=False).mean().replace(0, np.nan)
        return (s2 / sd) * 100.0

    # mặc định: slope của MACD histogram (làm trơn, chia ATR để so sánh giữa các mã)
    if macd_hist is None:
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - signal
    hist_sm = macd_hist.rolling(3, min_periods=1).mean()
    slope = hist_sm.diff()
    if atr is not None:
        atr_safe = pd.Series(atr).replace(0, np.nan).ffill()
        slope = slope / atr_safe
    return slope.rolling(5, min_periods=2).mean() * 100.0
