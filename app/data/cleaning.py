"""Chuẩn hoá dữ liệu OHLCV (PHẦN 2).

- Chuẩn hóa tên cột / kiểu dữ liệu, symbol uppercase.
- Loại trùng lặp, sắp xếp theo symbol + date.
- Kiểm tra dữ liệu lỗi: giá <= 0, high < low, close ngoài [low, high].
- Forward-fill tối đa max_ffill_gap phiên CHỈ cho mục đích hiển thị, có cờ ffilled.
- Đánh giá missing data / stale data -> data_issues.
- Lọc thanh khoản ADV20 >= min_adv_vnd (thiếu turnover thì ước lượng volume*close*1000).

Lưu ý đơn vị giá VN: vnstock/VNDirect trả giá ở đơn vị nghìn đồng (21.5 = 21,500đ),
yfinance trả VND tuyệt đối. Hệ số qui đổi về VND nằm trong config.universe.price_unit_multiplier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import cfg_get
from app.logging_utils import get_logger

log = get_logger(__name__)

OHLCV_COLS = ["date", "symbol", "open", "high", "low", "close", "volume", "turnover"]

_ALIAS = {
    "ngay": "date", "ngày": "date", "datetime": "date", "time": "date", "timestamp": "date",
    "ma": "symbol", "ticker": "symbol", "code": "symbol",
    "mo": "open", "giá mở cửa": "open", "开": "open",
    "dong": "close", "đóng": "close", "gc": "close",
    "cao": "high", "thap": "low", "khoi_luong": "volume", "kl": "volume",
    "gtgd": "turnover", "giá trị giao dịch": "turnover", "value": "turnover",
    "amount": "turnover", "gia tri": "turnover",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Đổi tên cột về chuẩn OHLCV_COLS qua bảng alias (không phân biệt hoa/thường, dấu)."""
    df = df.copy()
    renames = {}
    for c in df.columns:
        key = str(c).strip().lower()
        if key in OHLCV_COLS:
            renames[c] = key
        elif key in _ALIAS:
            renames[c] = _ALIAS[key]
    df = df.rename(columns=renames)
    return df


def clean_ohlcv(df: pd.DataFrame, source: str = "unknown") -> pd.DataFrame:
    """Chuẩn hoá kiểu dữ liệu + loại bản ghi lỗi rõ ràng. Trả về DataFrame các cột OHLCV (+source)."""
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLS + ["source"])
    df = normalize_columns(df)
    if "symbol" not in df.columns:
        raise ValueError("Thiếu cột symbol sau chuẩn hoá")
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(
        "Asia/Ho_Chi_Minh").dt.date
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["date", "close"])
    # loại bản ghi lỗi hiển nhiên
    bad = (
        (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (df["high"] < df["low"])
        | (df["close"] > df["high"] + 1e-9)
        | (df["close"] < df["low"] - 1e-9)
        | (df["open"] > df["high"] + 1e-9)
        | (df["open"] < df["low"] - 1e-9)
        | (df["volume"] < 0)
    )
    n_bad = int(bad.sum())
    if n_bad:
        log.warning("Loại %d/%d bản ghi lỗi giá/volume (symbol=%s)", n_bad, before,
                    df.loc[bad, "symbol"].unique()[:5])
    df = df[~bad]
    df = df.drop_duplicates(subset=["symbol", "date"]).sort_values(
        ["symbol", "date"]).reset_index(drop=True)
    df["volume"] = df["volume"].fillna(0.0)
    df["source"] = source
    return df[OHLCV_COLS + ["source"]]


def per_symbol_quality(df: pd.DataFrame, cfg: dict, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Tính bảng chất lượng dữ liệu + thanh khoản theo từng symbol.

    Trả về DataFrame index=symbol với các cột:
      bars, last_date, stale_days, missing_ratio, adv20_vnd, est_turnover,
      avg_price_20, vol_spike_max, ok, issues(list[str]), warnings(list[str]).
    """
    ucfg = cfg_get(cfg, "universe", {})
    price_mult = float(cfg_get(ucfg, "price_unit_multiplier", 1000))
    min_adv = float(cfg_get(ucfg, "min_adv_vnd", 10_000_000_000))
    max_missing = float(cfg_get(ucfg, "max_missing_ratio", 0.10))
    recent_days = int(cfg_get(ucfg, "recent_window_days", 60))
    require_hist = int(cfg_get(ucfg, "require_history_bars", 120))

    rows = []
    as_of = pd.Timestamp(as_of_date)
    for sym, g in df.groupby("symbol", sort=True):
        g = g.sort_values("date")
        issues: list[str] = []
        warns: list[str] = []
        bars = len(g)
        last = pd.Timestamp(g["date"].iloc[-1]) if bars else None
        stale_days = int((as_of - last).days) if last is not None else 9999
        if bars < require_hist:
            warns.append(f"thiếu lịch sử ({bars} < {require_hist} phiên)")
        if last is not None and stale_days > recent_days:
            warns.append(f"dữ liệu cũ (phiên cuối cách as_of {stale_days} ngày)")

        # missing ratio: số ngày âm lịch đã giao dịch dự kiến trong recent window
        recent = g[g["date"] >= (as_of - pd.Timedelta(days=int(recent_days * 1.6))).date()]
        expected = max(int(recent_days * 0.95), 1)  # ~22 phiên/tháng
        got = len(recent)
        missing_ratio = max(0.0, 1.0 - got / expected)
        if missing_ratio > max_missing:
            issues.append(f"thiếu {missing_ratio:.0%} phiên trong {recent_days} ngày gần nhất")

        # turnover & ADV20
        est_turnover = False
        to = pd.to_numeric(g.get("turnover"), errors="coerce")
        if to is None or to.isna().mean() > 0.5 or (to.fillna(0) <= 0).mean() > 0.5:
            est = g["volume"] * g["close"] * price_mult
            est_turnover = True
        else:
            med = float(to[to > 0].median()) if (to > 0).any() else 0.0
            if med < 1e7:  # nghi ngờ đơn vị nghìn đồng
                est = to * 1000.0
                est_turnover = True
            else:
                est = to
        tail = est.tail(20)
        adv20 = float(tail.mean()) if len(tail) else 0.0
        if adv20 < min_adv:
            issues.append(f"thanh khoản thấp: ADV20={adv20/1e9:.1f} tỷ VND < {min_adv/1e9:.0f} tỷ")

        avg_price = float(g["close"].tail(20).mean()) if bars else np.nan
        # suy ra hệ số đơn vị thực tế của nguồn (vnstock: nghìn đồng -> mult~1000;
        # yfinance sau qui đổi cũng ~1000). Không hard-code theo nguồn.
        obs_mult = 1000.0 if (not np.isnan(avg_price) and avg_price > 0 and avg_price < 500) else 1.0
        low_price = (not np.isnan(avg_price)) and avg_price * obs_mult < float(cfg_get(ucfg, "min_price", 5000))
        if low_price:
            warns.append(f"giá quá thấp (~{avg_price*obs_mult:.0f}đ) - giảm confidence")

        # volume bất thường: z-score cực đại trong 60 phiên gần
        v = g["volume"]
        vol_spike_max = 0.0
        if len(v) > 25:
            vm = v.rolling(20).mean()
            vs = v.rolling(20).std()
            z = ((v - vm) / vs.replace(0, np.nan)).tail(60)
            vol_spike_max = float(z.max()) if z.notna().any() else 0.0
        if vol_spike_max > 12:
            warns.append(f"volume tăng đột biến bất thường (z={vol_spike_max:.0f}) - cần kiểm tra tin/corporate action")

        zero_vol = float((g["volume"].tail(20) <= 0).mean())
        if zero_vol > 0.3:
            issues.append(f"nhiều phiên volume=0 ({zero_vol:.0%}/20 phiên gần)")

        rows.append(dict(symbol=sym, bars=bars, last_date=last, stale_days=stale_days,
                         missing_ratio=missing_ratio, adv20_vnd=adv20,
                         est_turnover=est_turnover, avg_price_20=avg_price,
                         low_price=bool(low_price),
                         vol_spike_max=vol_spike_max,
                         ok=len(issues) == 0, issues=issues, warnings=warns))
    q = pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame(
        columns=["bars", "ok", "issues", "warnings"]).rename_axis("symbol")
    return q


def apply_display_ffill(g: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Forward-fill tối đa max_ffill_gap phiên CHO MỤC ĐÍCH HIỂN THỊ, đánh dấu ffilled=True.
    Tín hiệu vẫn tính trên dữ liệu gốc; ffill chỉ tránh NaN lan khi vẽ/tính range."""
    gap = int(cfg_get(cfg, "universe.max_ffill_gap", 2))
    g = g.sort_values("date").copy()
    g["ffilled"] = False
    idx = pd.DatetimeIndex(pd.to_datetime(g["date"]))
    reindexed = g.set_index("date").resample("B")[["open", "high", "low", "close", "volume", "turnover"]]
    filled = reindexed.ffill(limit=gap)
    new_rows = filled.index.difference(idx)
    if len(new_rows):
        add = filled.loc[new_rows].copy()
        add["ffilled"] = True
        add["symbol"] = g["symbol"].iloc[0]
        merged = pd.concat([g.assign(ffilled=False),
                            add.reset_index().rename(columns={"index": "date"})],
                           ignore_index=True).sort_values("date").reset_index(drop=True)
        return merged
    return g
