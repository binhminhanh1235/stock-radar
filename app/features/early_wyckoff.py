"""Nhận diện sớm Wyckoff / VPA (PHẦN 4).

Nguyên tắc:
- Chỉ dùng dữ liệu <= as_of_date (nến as_of đã đóng). Hàm nhận DataFrame đã cắt tới as_of.
- Mỗi mã trả về Signal: stage, early_score, confidence, phase, setup, reasons (tiếng Việt),
  invalidation, các mức giá, missing_data, data_quality.
- Thiếu dữ liệu thành phần => giảm confidence + ghi rõ missing_data, KHÔNG bịa.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.config import cfg_get
from app.features.indicators import detect_range
from app.logging_utils import get_logger

log = get_logger(__name__)

STAGES = ["AVOID", "EARLY_ACCUMULATION", "SPRING_CANDIDATE", "SHAKEOUT_CANDIDATE",
          "TEST_READY", "CONFIRMED_BUY_CANDIDATE"]
STAGE_RANK = {s: i for i, s in enumerate(STAGES)}


@dataclass
class Signal:
    as_of_date: str
    symbol: str
    exchange: str = ""
    sector: str = "UNKNOWN"
    stage: str = "AVOID"
    phase: str = "UNKNOWN"
    setup: str = ""
    early_score: float = 0.0
    confidence: float = 0.0
    support: float = np.nan
    resistance: float = np.nan
    range_height: float = np.nan
    spring_date: str | None = None
    shakeout_date: str | None = None
    test_date: str | None = None
    entry_trigger: float = np.nan
    stop_loss: float = np.nan
    target_1: float = np.nan
    target_2: float = np.nan
    rr_estimate: float = np.nan
    reasons: list[str] = field(default_factory=list)
    invalidation: str = ""
    next_review_date: str | None = None
    missing_data: list[str] = field(default_factory=list)
    data_quality: str = "OK"
    scores: dict = field(default_factory=dict)
    close: float = np.nan
    atr14: float = np.nan
    rvol: float = np.nan
    rsi14: float = np.nan

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _nan(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x)
    except Exception:
        return False


def _clip(x, lo=0.0, hi=100.0) -> float:
    return max(lo, min(hi, x))


# ---------------- event detection (chỉ nhìn quá khứ + nến as_of đã đóng) ----------------

def find_events(ind: pd.DataFrame, cfg: dict) -> dict:
    """Quét 30 phiên gần nhất tìm spring / shakeout / test / SOS (các nến ĐÃ ĐÓNG CỬA)."""
    scfg = cfg_get(cfg, "signals", {})
    g = ind.reset_index(drop=True)
    n = len(g)
    lb = int(scfg.get("test_lookback_days", 10))
    win_start = max(0, n - 1 - (lb + 8))     # cửa sổ sự kiện "tươi"
    old_start = max(0, n - 1 - (lb + 30))    # cửa sổ sự kiện cũ (đã có test kèm theo)
    r_now = detect_range(g, cfg)             # range hiện tại (hỗn hợp cả nến cuối)
    r_prev = detect_range(g.iloc[: n - 1], cfg) if n > 5 else {}  # range TRƯỚC nến as_of
    events = {"spring": None, "shakeout": None, "test": None, "sos": None,
              "range": r_now, "range_prev": r_prev}

    def _pen_breaks(sup, lo_i, hi_i):
        """Tìm các phiên đâm thủng support rồi hồi trong [lo_i, hi_i].

        Support tham chiếu = min(support của range, swing low gần nhất trước đó):
        spring/shakeout là hành vi giá VƯỢT QUA ĐÁY CŨ, nên phải lấy đáy làm chuẩn.
        """
        found = []
        if _nan(sup):
            return found
        prev_lows = [lv for _, lv, _ in (r_prev.get("swing_lows") or []) if lv > 0]
        ref_sup = min([sup] + prev_lows[-2:]) if prev_lows else sup
        max_pen = float(scfg.get("spring_max_atr_penetration", 1.0))
        for i in range(max(lo_i, 1), min(hi_i, n)):
            row = g.iloc[i]
            atr_v = row.get("atr14", np.nan)
            if _nan(atr_v) or atr_v <= 0:
                continue
            sup_i = ref_sup
            pen = sup_i - row["low"]
            if 0.05 * atr_v < pen <= max_pen * atr_v:
                recovered = row["close"] >= sup_i
                closes_back = (i + 1 < n) and bool(g["close"].iloc[i + 1] >= sup_i)
                if not (recovered or closes_back):
                    continue
                vma = row.get("volume_ma20", np.nan)
                vol_ratio = float(row["volume"] / vma) if not _nan(vma) and vma > 0 else np.nan
                lower_wick = min(row["open"], row["close"]) - row["low"]
                body = abs(row["close"] - row["open"])
                kind = "shakeout" if (not _nan(vol_ratio) and vol_ratio >=
                                      float(scfg.get("shakeout_min_volume_ratio", 1.5))) else "spring"
                found.append(dict(kind=kind, idx=i, low=float(row["low"]),
                                  vol=float(row["volume"]), vol_ratio=vol_ratio,
                                  long_wick=bool(lower_wick > body)))
        return found

    def _find_test(ev):
        """Test sau sự kiện: low không thủng thêm >0.2 ATR, volume cạn, nến đóng nửa trên."""
        tmax = float(scfg.get("test_volume_max_ratio", 0.8))
        for i in range(ev["idx"] + 1, n):
            row = g.iloc[i]
            atr_v = row.get("atr14", np.nan)
            vma = row.get("volume_ma20", np.nan)
            if _nan(atr_v) or _nan(vma):
                continue
            if (row["low"] >= ev["low"] - 0.2 * atr_v
                    and row["volume"] <= tmax * vma
                    and (row["close"] >= row["open"]
                         or row["close"] >= (row["high"] + row["low"]) / 2)):
                return dict(idx=i, low=float(row["low"]), vol=float(row["volume"]))
        return None

    # 1) sự kiện tươi theo support của nến as_of (vd spring ngay hôm nay)
    fresh = _pen_breaks(r_now.get("support", np.nan), win_start, n)
    # 2) sự kiện cũ hơn theo range TRƯỚC as_of -> cần có test đi kèm
    stale = _pen_breaks(r_prev.get("support", np.nan), old_start, win_start)
    chosen = None; chosen_test = None
    if fresh:
        chosen = fresh[-1]
        chosen_test = _find_test(chosen)
    elif stale:
        for ev in reversed(stale):
            t = _find_test(ev)
            if t is not None:
                chosen, chosen_test = ev, t
                break
        if chosen is None:
            chosen = stale[-1]
            chosen_test = _find_test(chosen)
    if chosen is not None:
        events[chosen["kind"]] = chosen
        if chosen_test is not None:
            events["test"] = chosen_test
    elif r_prev.get("support") is not None and not _nan(r_prev.get("support", np.nan)):
        # không có xuyên support: cân nhắc "test lại vùng low cũ" (không có spring/shakeout)
        lows = r_prev.get("swing_lows") or []
        if lows:
            ev_low = float(lows[-1][1])
            t = _find_test(dict(idx=max(0, lows[-1][0]), low=ev_low, vol=np.nan))
            if t and t["idx"] >= win_start:
                events["test"] = t

    # SOS: phiên tăng mạnh vượt resistance với volume cao trong 25 phiên gần nhất
    resi = r_prev.get("resistance", np.nan)
    if not _nan(resi):
        look_lo = max(1, n - min(25, n))
        for i in range(look_lo, n):
            row = g.iloc[i]
            prev_c = g["close"].iloc[i - 1]
            gain = row["close"] / prev_c - 1 if prev_c > 0 else 0
            vma = row.get("volume_ma20", np.nan)
            if (row["close"] > resi and gain >= float(cfg_get(cfg, "signals.sos_min_gain_pct", 0.02))
                    and not _nan(vma) and vma > 0
                    and row["volume"] >= float(cfg_get(cfg, "signals.breakout_vol_ratio", 1.5)) * vma):
                events["sos"] = dict(idx=i, level=float(resi), vol=float(row["volume"]))
    return events


def pullback_after_sos(ind: pd.DataFrame, events: dict, cfg: dict) -> dict | None:
    """Nhịp pullback sau SOS: giữ trên vùng breakout/EMA20, volume thấp."""
    sos = events.get("sos")
    if not sos:
        return None
    scfg = cfg_get(cfg, "signals", {})
    n = len(ind)
    start = sos["idx"] + 1
    end = min(n, start + int(scfg.get("pullback_max_bars", 15)) + 1)
    seg = ind.iloc[start:end]
    if seg.empty:
        return None
    vma_last = seg["volume_ma20"].iloc[-1]
    vols_ok = (seg["volume"] <= 0.8 * seg["volume_ma20"]).mean() if not _nan(vma_last) else np.nan
    gg = ind.reset_index(drop=True)
    held = (seg["low"] >= sos["level"] - 0.3 * gg["atr14"].iloc[-1]).all() \
        if not _nan(gg["atr14"].iloc[-1]) else False
    last_held = bool(gg["close"].iloc[-1] >= sos["level"])   # nến as_of vẫn phải trên vùng breakout
    ema_held = (seg["close"] >= ind["ema20"].iloc[start:end] * 0.99).mean() >= 0.8
    return dict(bars=len(seg), low=float(seg["low"].min()),
                vol_dry=float(vols_ok) if not _nan(vols_ok) else np.nan,
                held_support=bool(held and last_held), held_ema=bool(ema_held))


# ---------------- sub-scores 0..100 ----------------

def score_structure(ind: pd.DataFrame, events: dict, cfg: dict, missing: list) -> float:
    r = events["range"]; s = 50.0
    sup, resi = r.get("support", np.nan), r.get("resistance", np.nan)
    if _nan(sup) or _nan(resi):
        missing.append("không xác định được range/support")
        return 30.0
    dur = r.get("range_duration") or 0
    scfg = cfg_get(cfg, "signals", {})
    if dur >= int(scfg.get("range_min_bars", 20)):
        s += 15
    if dur >= 40:
        s += 10
    n_test = r.get("n_test_support", 0)
    s += min(15, n_test * 3)
    if r.get("higher_low"):
        s += 15
    height_pct = (resi - sup) / max(sup, 1e-9)
    if 0.06 <= height_pct <= 0.35:      # biên độ hợp lý cho nền tích luỹ
        s += 10
    elif height_pct > 0.5:
        s -= 10
    return _clip(s)


def score_vpa(ind: pd.DataFrame, events: dict, cfg: dict, missing: list) -> float:
    last = ind.iloc[-1]; s = 40.0
    dry = float(cfg_get(cfg, "signals.dry_up_rvol", 0.7))
    rv_tail = ind["rvol"].tail(6)
    if rv_tail.notna().sum() >= 4:
        frac_low = float((rv_tail <= dry).mean())
        s += 25 * frac_low
    dv_now, dv_prev = last.get("downvol_ma10"), last.get("downvol_ma10_prev")
    if not _nan(dv_now) and not _nan(dv_prev) and dv_prev > 0:
        if dv_now < 0.8 * dv_prev:
            s += 20                          # nhịp giảm gần đây volume cạn dần
        elif dv_now > 1.3 * dv_prev:
            s -= 15                          # áp lực bán gia tăng
    ev = events.get("spring") or events.get("shakeout")
    if ev and not _nan(ev.get("vol_ratio")):
        if ev["kind"] == "spring" and ev["vol_ratio"] < 1.0:
            s += 15                          # spring volume thấp = tốt
        if ev["kind"] == "shakeout" and ev["vol_ratio"] >= 1.5:
            s += 5                           # shakeout vol cao nhưng hồi: trung tính-hơi tốt
    tst = events.get("test")
    if tst and ev and ev["vol"] > 0:
        ratio = tst["vol"] / ev["vol"]
        if ratio <= 0.7:
            s += 20
        elif ratio <= 1.0:
            s += 10
    cmf = last.get("cmf20")
    if not _nan(cmf):
        s += 10 if cmf > 0.05 else (-10 if cmf < -0.1 else 0)
    obv_sl = last.get("obv_slope")
    if not _nan(obv_sl):
        s += 10 if obv_sl > 0 else -5
    return _clip(s)


def score_momentum(ind: pd.DataFrame, cfg: dict, missing: list) -> float:
    last = ind.iloc[-1]; s = 40.0
    r = last["rsi14"]
    if _nan(r):
        missing.append("RSI"); return 40.0
    if r > 45: s += 15
    if r > 50: s += 10
    if r < 28: s -= 15
    hist = ind["macd_hist"]
    if hist.notna().sum() >= 3:
        inc = int((hist.diff().tail(3) > 0).sum())
        s += 8 * inc
        if hist.iloc[-1] > 0: s += 8
    md = last.get("mcdx")
    if not _nan(md):
        s += 10 if md > 0 else -5
    c = last["close"]
    if not _nan(last.get("ema20")): s += 8 if c > last["ema20"] else -8
    if not _nan(last.get("sma20")): s += 4 if c > last["sma20"] else -4
    return _clip(s)


def score_rs(ind: pd.DataFrame, cfg: dict, missing: list) -> float:
    last = ind.iloc[-1]
    if _nan(last.get("rs")):
        missing.append("benchmark/relative strength")
        return 50.0                          # trung tính nhưng đã ghi missing
    s = 40.0
    sl = last.get("rs_slope_10")
    if not _nan(sl): s += 25 if sl > 0 else 0
    pctl = last.get("rs_percentile")
    if not _nan(pctl): s += 30 * float(pctl) / 100.0
    if s == 40.0 and _nan(sl) and _nan(pctl):
        missing.append("RS slope/percentile chưa đủ dữ liệu")
    return _clip(s)


def score_liquidity(q: dict, cfg: dict, missing: list) -> tuple[float, float]:
    """Trả về (điểm, penalty giá thấp). q từ per_symbol_quality."""
    adv = q.get("adv20_vnd", 0.0)
    min_adv = float(cfg_get(cfg, "universe.min_adv_vnd", 1e10))
    s = 30.0
    if adv >= min_adv: s += 30
    if adv >= 3 * min_adv: s += 20
    if adv >= 10 * min_adv: s += 20
    pen = 0.0
    if q.get("low_price"): pen = 0.10
    if q.get("est_turnover"): s -= 10        # turnover phải ước lượng
    return _clip(s), pen


def score_early_event(events: dict, pb: dict | None, cfg: dict) -> float:
    s = 0.0
    if events.get("spring"): s += 40
    if events.get("shakeout"): s += 40
    if events.get("test"): s += 35
    if events.get("sos"): s += 25
    if pb and pb.get("held_support"): s += 20
    return _clip(s)


# ---------------- main classifier ----------------

def classify_symbol(g_ind: pd.DataFrame, symbol: str, quality: dict, cfg: dict,
                    exchange: str = "", sector: str = "UNKNOWN") -> Signal:
    """g_ind: indicators đã tính trên dữ liệu <= as_of_date (nến cuối = as_of, ĐÃ ĐÓNG)."""
    scfg = cfg_get(cfg, "signals", {})
    missing: list[str] = []
    warnings: list[str] = []
    n = len(g_ind)
    as_of = str(pd.Timestamp(g_ind["date"].iloc[-1]).date())
    sig = Signal(as_of_date=as_of, symbol=symbol, exchange=exchange, sector=sector)
    last = g_ind.iloc[-1]
    atr_v = float(last["atr14"]) if not _nan(last["atr14"]) else np.nan
    sig.close = float(last["close"]); sig.atr14 = atr_v
    sig.rvol = float(last["rvol"]) if not _nan(last["rvol"]) else np.nan
    sig.rsi14 = float(last["rsi14"]) if not _nan(last["rsi14"]) else np.nan

    if n < int(cfg_get(cfg, "universe.require_history_bars", 120)):
        missing.append(f"lịch sử ngắn ({n} phiên)")
        warnings.append("thiếu lịch sử - giảm confidence")
    if quality.get("issues"):
        sig.data_quality = "ERROR"
        sig.stage = "AVOID"
        sig.reasons = [f"Dữ liệu lỗi/thanh khoản không đạt: {i}" for i in quality["issues"]]
        sig.confidence = 0.0
        sig.missing_data = missing
        return sig
    if quality.get("warnings"):
        sig.data_quality = "WARN"
        warnings.extend(quality["warnings"])

    events = find_events(g_ind, cfg)
    r = events["range"]
    sig.support = r.get("support", np.nan); sig.resistance = r.get("resistance", np.nan)
    sig.range_height = r.get("range_height", np.nan)
    pb = pullback_after_sos(g_ind, events, cfg)

    for key, attr in (("spring", "spring_date"), ("shakeout", "shakeout_date"), ("test", "test_date")):
        ev = events.get(key)
        if ev:
            setattr(sig, attr, str(pd.Timestamp(g_ind["date"].iloc[ev["idx"]]).date()))

    # -------- AVOID: gãy nền / bán tháo / phân kỳ xấu --------
    avoid_reasons = []
    recent_ev_idx = max([events[k]["idx"] for k in ("spring", "shakeout") if events.get(k)] or [-999])
    has_recent_break = recent_ev_idx >= n - 4          # break còn "tươi" (<=3 phiên trước)
    lows_hist = [lv for _, lv, _ in ((events["range"].get("swing_lows") or []) +
                                     (events.get("range_prev", {}).get("swing_lows") or []))]
    sup_eff = min([sig.support] + [lv for lv in lows_hist if not _nan(lv)]) \
        if not _nan(sig.support) else np.nan
    if not _nan(sup_eff):
        brk_look = g_ind.tail(6)
        below = brk_look["close"] < sup_eff - 0.5 * (atr_v if not _nan(atr_v) else 0)
        heavy_sell = brk_look["volume"] >= float(scfg.get("invalidation_sell_vol_ratio", 1.8)) * brk_look["volume_ma20"]
        if int((below & heavy_sell.fillna(False)).sum()) >= 3 and not has_recent_break:
            avoid_reasons.append("đóng cửa dưới đáy hỗ trợ với volume bán cao bất thường nhiều phiên (gãy nền)")
        if int(below.sum()) >= 4 and not has_recent_break:
            avoid_reasons.append("giá nằm sâu dưới đáy hỗ trợ nhiều phiên, không có dấu hiệu hồi")
    if not _nan(last["close"]) and not _nan(g_ind["ema200"].iloc[-1]) and n >= 200:
        recent_low = float(g_ind["low"].tail(5).min())
        prev_low = float(g_ind["low"].iloc[-30:-5].min()) if n >= 30 else np.nan
        falling = bool(g_ind["close"].iloc[-1] < g_ind["close"].iloc[-6]) if n >= 6 else False
        below_sup = last["close"] < (sup_eff if not _nan(sup_eff) else sig.support) - 0.5 * (atr_v or 0)
        if not _nan(prev_low) and recent_low < prev_low - 0.5 * (atr_v or 0) and falling and below_sup \
                and not has_recent_break:
            avoid_reasons.append("đáy sau thấp hơn đáy trước + đóng cửa sâu dưới support (lower_low)")
    if avoid_reasons:
        sig.stage = "AVOID"; sig.setup = "BROKEN"
        sig.reasons = avoid_reasons + warnings
        sig.invalidation = "Không có tín hiệu tích luỹ; loại khỏi theo dõi."
        sig.confidence = 0.6
        sig.missing_data = missing
        sig.early_score = _clip(100 - 25 * len(avoid_reasons))
        return sig

    # -------- scoring --------
    st = score_structure(g_ind, events, cfg, missing)
    vp = score_vpa(g_ind, events, cfg, missing)
    mo = score_momentum(g_ind, cfg, missing)
    rs = score_rs(g_ind, cfg, missing)
    lq, low_pen = score_liquidity(quality, cfg, missing)
    ee = score_early_event(events, pb, cfg)
    w = cfg_get(cfg, "scoring", {})
    score = (float(w.get("weight_structure", .3)) * st + float(w.get("weight_vpa", .25)) * vp +
             float(w.get("weight_momentum", .15)) * mo + float(w.get("weight_relative_strength", .1)) * rs +
             float(w.get("weight_liquidity", .1)) * lq + float(w.get("weight_early_event", .1)) * ee)
    sig.early_score = round(_clip(score), 1)
    sig.scores = dict(structure=round(st, 1), vpa=round(vp, 1), momentum=round(mo, 1),
                      relative_strength=round(rs, 1), liquidity=round(lq, 1), early_event=round(ee, 1))

    reasons: list[str] = []
    stage = "EARLY_ACCUMULATION"; setup = "BASE_FORMING"
    conf = 0.5 + (sig.early_score - 50) / 100.0

    # ---- EARLY_ACCUMULATION evidence ----
    deep_drop = False
    if n >= 60:
        peak = float(g_ind["close"].tail(int(scfg.get("lookback_6m", 126))).max())
        deep_drop = last["close"] < peak * (1 - float(scfg.get("drawdown_threshold", 0.2)))
    below_ema200 = (not _nan(g_ind["ema200"].iloc[-1])) and last["close"] < g_ind["ema200"].iloc[-1]
    if deep_drop or below_ema200:
        reasons.append("giá đã giảm sâu trước đó (điều kiện cần của pha tích luỹ)")
    r_series = g_ind["rsi14"]
    if r_series.notna().sum() >= 15:
        was_oversold = float(r_series.tail(25).min()) < 30
        if was_oversold and not _nan(last["rsi14"]) and last["rsi14"] > 30:
            reasons.append("RSI hồi lên trên 30 sau khi rơi vào vùng quá bán")
    if not _nan(last.get("sma20")) and n >= 3:
        if last["close"] > last["sma20"] and g_ind["close"].iloc[-4] < g_ind["sma20"].iloc[-4]:
            reasons.append("close vượt lên SMA20 sau chuỗi ngày nằm dưới")
    if r.get("higher_low"):
        reasons.append("đáy sau cao hơn đáy trước (swing low nâng lên)")
    dv_now, dv_prev = last.get("downvol_ma10"), last.get("downvol_ma10_prev")
    if not _nan(dv_now) and not _nan(dv_prev) and dv_prev > 0 and dv_now < 0.8 * dv_prev:
        reasons.append("volume các nhịp giảm gần nhất cạn dần")
    # selling climax rồi không giảm thêm
    if n >= 40:
        tail = g_ind.tail(35)
        zmax = tail["volume_zscore_20"].max()
        if not _nan(zmax) and zmax >= 3:
            imax = int(tail["volume_zscore_20"].idxmax())
            after = g_ind["close"].iloc[imax + 1:].min() if imax + 1 < n else np.nan
            if not _nan(after) and after >= g_ind["close"].iloc[imax] * 0.97:
                reasons.append("xuất hiện phiên khối lượng cực đại (selling climax?) nhưng giá không giảm thêm")

    # ---- SPRING_CANDIDATE ----
    near_support = (not _nan(sig.support)) and (not _nan(atr_v)) and \
        (abs(last["close"] - sig.support) <= float(scfg.get("support_atr_tolerance", 1.0)) * atr_v
         or last["low"] <= sig.support + 0.3 * atr_v)
    has_range = (not _nan(r.get("range_duration"))) and r["range_duration"] >= int(scfg.get("range_min_bars", 20))
    dry_recent = (not _nan(sig.rvol)) and sig.rvol < 0.9
    obv_ok = _nan(last.get("obv_slope")) or last["obv_slope"] >= -0.5
    no_heavy_sell_recent = not ((g_ind["volume"].tail(3) >=
                                 float(scfg.get("invalidation_sell_vol_ratio", 1.8)) *
                                 g_ind["volume_ma20"].tail(3).fillna(1e18)).any())
    rsi_div = False
    if r_series.notna().sum() >= 20 and not _nan(last["rsi14"]):
        lows2 = g_ind["low"].tail(15)
        rsi2 = g_ind["rsi14"].tail(15)
        if lows2.idxmin() == lows2.index[-1] and rsi2.iloc[-1] > rsi2.min() + 3:
            rsi_div = True

    # ---- events based stages ----
    ev_spr, ev_shk, ev_tst = events.get("spring"), events.get("shakeout"), events.get("test")
    chosen_ev = ev_spr or ev_shk
    dist_event = (n - 1 - chosen_ev["idx"]) if chosen_ev else None
    within_lb = dist_event is not None and 1 <= dist_event <= int(scfg.get("test_lookback_days", 10))

    confirmed = None
    if within_lb and ev_tst and (ev_spr or ev_shk):
        ev = ev_spr or ev_shk
        cond_rsi = (not _nan(last["rsi14"])) and last["rsi14"] > 45 and \
            g_ind["rsi14"].iloc[-1] >= g_ind["rsi14"].iloc[-4]
        cond_macd = (g_ind["macd_hist"].diff().tail(2) > 0).any() or \
            (not _nan(last.get("mcdx")) and last["mcdx"] > 0)
        test_vol_low = ev["vol"] > 0 and ev_tst["vol"] <= 0.7 * ev["vol"]
        reclaimed = last["close"] >= sig.support
        if (test_vol_low or ev_tst["vol"] <= float(scfg.get("test_volume_max_ratio", .8)) *
                (last["volume_ma20"] if not _nan(last.get("volume_ma20")) else 1e18)) \
                and reclaimed and cond_rsi and cond_macd:
            confirmed = ("TEST_AFTER_SPRING" if ev_spr else "TEST_AFTER_SHAKEOUT")
    if confirmed is None and events.get("sos") and pb and pb["bars"] >= 2 \
            and pb["held_support"] and (pb.get("vol_dry") or 0) >= 0.5 \
            and (not _nan(last["rsi14"])) and last["rsi14"] >= 45 \
            and (g_ind["macd_hist"].iloc[-1] >= g_ind["macd_hist"].iloc[-4] - 1e-9):
        confirmed = "BU_LPS_PULLBACK"

    if confirmed:
        stage = "CONFIRMED_BUY_CANDIDATE"; setup = confirmed
        conf += 0.15
        reasons.append({
            "TEST_AFTER_SPRING": "test sau spring với volume cạn, RSI/MACD cải thiện",
            "TEST_AFTER_SHAKEOUT": "test sau terminal shakeout, giữ hỗ trợ, động lượng hồi phục",
            "BU_LPS_PULLBACK": "SOS vượt kháng cự volume cao rồi pullback cạn cung giữ trên vùng breakout (LPS)",
        }[confirmed])
        if ev_spr: reasons.append(f"xuất hiện spring ngày {sig.spring_date}")
        if ev_shk: reasons.append(f"xuất hiện shakeout ngày {sig.shakeout_date}")
        if ev_tst: reasons.append(f"nhịp test ngày {sig.test_date} volume thấp, nến đóng ở nửa trên")
    elif ev_tst and (within_lb or chosen_ev is None):
        stage = "TEST_READY"; setup = "TESTING_" + ("SPRING" if ev_spr else "SHAKEOUT")
        conf += 0.08
        kinds = "/".join([k for k, e in (("spring", ev_spr), ("shakeout", ev_shk)) if e]) or "candle low cũ"
        dts = ", ".join([d for d in (sig.spring_date, sig.shakeout_date) if d]) or str(sig.test_date)
        reasons.append(f"có {kinds} ngày {dts} và nhịp test lại với volume cạn hơn")
    elif ev_shk and dist_event is not None and 0 <= dist_event <= 3:
        stage = "SHAKEOUT_CANDIDATE"; setup = "SHAKEOUT_RECOVERY"
        if ev_shk.get("vol_ratio", 0) and ev_shk["vol_ratio"] >= float(scfg.get("shakeout_min_volume_ratio", 1.5)):
            conf += 0.07
            reasons.append("shakeout volume cao nhưng giá đóng cửa/hồi lại trên support ngay sau đó")
        if ev_shk.get("long_wick"):
            reasons.append("nến có bóng dưới dài hơn thân (từ chối giá tại support sâu)")
    elif ev_spr and dist_event is not None and 0 <= dist_event <= 3:
        stage = "SPRING_CANDIDATE"; setup = "SPRING_JUST_FOUND"
        reasons.append("vừa xuất hiện phiên xuyên support nhưng đóng cửa hồi lên (spring tiềm năng/cần test)")
    elif has_range and near_support and dry_recent and no_heavy_sell_recent and (rsi_div or obv_ok):
        stage = "SPRING_CANDIDATE"; setup = "NEAR_SUPPORT_DRYUP"
        reasons.append("giá đang sát support của range với volume cạn dần (rvol < 0.9)")
        if rsi_div: reasons.append("RSI không tạo đáy mới cùng giá - phân kỳ dương nhẹ")
        if not _nan(last.get("obv_slope")) and last["obv_slope"] >= 0:
            reasons.append("OBV không giảm theo giá - dòng tiền đứng ngoài")
    elif deep_drop or below_ema200 or r.get("higher_low") or len(reasons) > 1:
        stage = "EARLY_ACCUMULATION"; setup = "BASE_FORMING"
    else:
        stage = "EARLY_ACCUMULATION"; setup = "WATCH_ONLY"
        conf -= 0.1
        reasons.append("chưa đủ bằng chứng tích luỹ rõ ràng - chỉ đưa vào diện quan sát")

    if _nan(sig.support):
        missing.append("không xác định được support/range")

    # -------- levels --------
    if not _nan(sig.support) and not _nan(atr_v):
        stop = min(sig.support, last["low"]) - 0.35 * atr_v
        trig = max(last["close"], (sig.support + sig.resistance) / 2) + 0.1 * atr_v \
            if stage in ("TEST_READY", "CONFIRMED_BUY_CANDIDATE") else sig.resistance - 0.05 * atr_v \
            if not _nan(sig.resistance) else np.nan
        t1 = sig.resistance if not _nan(sig.resistance) else np.nan
        rh = sig.range_height if not _nan(sig.range_height) else 0.0
        t2 = (t1 + rh) if not _nan(t1) else np.nan
        risk = max(trig - stop, 1e-9) if not _nan(trig) else np.nan
        rr = ((t2 - trig) / risk) if (not _nan(t2) and not _nan(risk) and risk > 0) else np.nan
        sig.stop_loss = round(stop, 3); sig.entry_trigger = round(trig, 3) if not _nan(trig) else np.nan
        sig.target_1 = round(t1, 3) if not _nan(t1) else np.nan
        sig.target_2 = round(t2, 3) if not _nan(t2) else np.nan
        sig.rr_estimate = round(rr, 2) if not _nan(rr) else np.nan
        sig.invalidation = (f"Đóng cửa dưới {round(stop,2)} hoặc xuất hiện phiên bán "
                            f"volume > {scfg.get('invalidation_sell_vol_ratio',1.8)}x trung bình 20 phiên")
    else:
        sig.invalidation = "Không xác định được support - cần bổ sung dữ liệu trước khi theo dõi."

    # -------- phase ước lượng --------
    if stage == "CONFIRMED_BUY_CANDIDATE" and setup == "BU_LPS_PULLBACK":
        sig.phase = "D"
    elif stage == "CONFIRMED_BUY_CANDIDATE":
        sig.phase = "C-D"
    elif stage == "TEST_READY":
        sig.phase = "C"
    elif stage in ("SPRING_CANDIDATE", "SHAKEOUT_CANDIDATE"):
        sig.phase = "B-C"
    elif stage == "EARLY_ACCUMULATION":
        sig.phase = "A-B" if (deep_drop or below_ema200) else "UNKNOWN"
    else:
        sig.phase = "UNKNOWN"

    # -------- confidence adjustments (không bịa: thiếu gì trừ nấy) --------
    conf -= 0.10 * len(missing)
    conf -= low_pen
    if quality.get("est_turnover"):
        conf -= 0.03
    if not _nan(sig.rvol) and sig.rvol > 1.5 and last["close"] < last["open"]:
        conf -= 0.05  # phiên cuối xả mạnh: cẩn trọng hơn
    sig.confidence = round(max(0.0, min(1.0, conf)), 2)
    sig.reasons = (reasons + warnings)[:12]
    review_days = int(np.median([int(h) for h in cfg_get(cfg, "review.horizons_days", [5, 10, 20])]))
    sig.next_review_date = str((pd.Timestamp(as_of) + timedelta(days=max(5, int(round(review_days * 1.4))))).date())
    sig.missing_data = missing
    return sig
