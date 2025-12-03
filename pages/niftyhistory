# streamlit_app.py
"""
NIFTY intraday charts (till 11:30 AM) using NSE chart API (current-day live fetch + local CSV cache).
Adds trend-strength detection per day (score 0-100 + classification).
"""

import streamlit as st
import pandas as pd
import requests
import datetime
import time
import pytz
import matplotlib.pyplot as plt
import numpy as np
import os

st.set_page_config(layout="wide", page_title="NIFTY — Intraday to 11:30 (Trend Strength)")

# ---------------------------
# Helpers: NSE session & fetch
# ---------------------------
def nse_session():
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.nseindia.com",
    }
    session.headers.update(headers)
    # A lightweight warm-up request to populate cookies
    try:
        session.get("https://www.nseindia.com", timeout=5)
    except Exception:
        # ignore network hiccups here; downstream call will show None
        pass
    return session

def fetch_today_intraday_nse():
    """
    Fetch intraday chart JSON for current day using NSE's chart API.
    Returns DataFrame with columns: ['datetime','price','volume'(if available)]
    """
    session = nse_session()
    url = "https://www.nseindia.com/api/chart-databyindex?index=EQUITY%7CNIFTY%2050&preopen=0"
    try:
        r = session.get(url, timeout=8)
        r.raise_for_status()
        j = r.json()
        # The JSON structure may vary; common fields:
        # j['grapthData'] : list of [timestamp, price] OR list of objects
        # Some versions include volume array at j['volume'] or inside grapthData elements
        if "grapthData" in j:
            gd = j["grapthData"]
            # If grapthData is a list of [timestamp, price], create DF
            if len(gd) > 0 and isinstance(gd[0], (list, tuple)) and len(gd[0]) >= 2:
                df = pd.DataFrame(gd)
                df.columns = ["timestamp", "price"] + (["_extra"+str(i) for i in range(df.shape[1]-2)] if df.shape[1]>2 else [])
                # convert millis -> IST
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms") + datetime.timedelta(hours=5, minutes=30)
                df = df[["datetime", "price"]]
                # attempt to add volume if available elsewhere
                if "volume" in j and isinstance(j["volume"], list) and len(j["volume"]) == len(df):
                    df["volume"] = j["volume"]
                else:
                    df["volume"] = np.nan
                return df
            # If grapthData is list of dicts
            elif len(gd) > 0 and isinstance(gd[0], dict):
                records = []
                for rec in gd:
                    # try to find time, price and volume keys
                    ts = rec.get("time") or rec.get("timestamp") or rec.get("x") or rec.get("dt")
                    price = rec.get("close") or rec.get("y") or rec.get("price")
                    vol = rec.get("volume") or rec.get("v")
                    # if timestamp in seconds or ms, handle
                    if ts is None:
                        continue
                    # normalize timestamp -> try as ms int
                    try:
                        ts_int = int(ts)
                        dt = pd.to_datetime(ts_int, unit="ms") + datetime.timedelta(hours=5, minutes=30)
                    except Exception:
                        # fallback: parse string
                        try:
                            dt = pd.to_datetime(ts)
                        except Exception:
                            continue
                    records.append({"datetime": dt, "price": price, "volume": vol})
                if not records:
                    return None
                df = pd.DataFrame(records)
                if "volume" not in df.columns:
                    df["volume"] = np.nan
                return df.sort_values("datetime").reset_index(drop=True)
        # fallback: try parsing generic JSON arrays (rare)
        return None
    except Exception:
        return None

# ---------------------------
# Caching / storage helper
# ---------------------------
CACHE_DIR = ".nse_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_path_for_date(date_obj):
    return os.path.join(CACHE_DIR, f"nifty_{date_obj.isoformat()}.csv")

def get_intraday_for_date(date_obj):
    """
    Return DataFrame for requested date.
    - If file exists in CACHE_DIR, read & return it
    - If date is today, try to fetch live from NSE and save to cache
    - Otherwise return None (past data not available unless previously cached)
    """
    path = cache_path_for_date(date_obj)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=["datetime"])
            # Ensure datetime tz-naive but in IST
            return df
        except Exception:
            # if cache read fails, remove file
            try:
                os.remove(path)
            except Exception:
                pass
    # if date is today, attempt live fetch
    if date_obj == datetime.date.today():
        df = fetch_today_intraday_nse()
        if df is not None and not df.empty:
            # save important columns
            df_to_save = df.copy()
            # ensure datetime column name normalized
            if "datetime" not in df_to_save.columns and 0 in df_to_save.columns:
                df_to_save = df_to_save.rename(columns={0:"datetime"})
            try:
                df_to_save.to_csv(path, index=False)
            except Exception:
                pass
            return df_to_save
    return None

# ---------------------------
# Session / schedule helpers
# ---------------------------
IST = pytz.timezone("Asia/Kolkata")

def now_ist():
    return datetime.datetime.now(IST)

def seconds_until_next_target(hour_targets=(9,17)):
    now = now_ist()
    today = now.date()
    candidates = []
    for h in hour_targets:
        target = IST.localize(datetime.datetime.combine(today, datetime.time(h,0)))
        if target <= now:
            target = target + datetime.timedelta(days=1)
        candidates.append(target)
    next_target = min(candidates)
    return int((next_target - now).total_seconds())

# ---------------------------
# Trend strength calculation
# ---------------------------
def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_vwap(df):
    # For intraday data we may not have high/low/close; we have price per minute.
    # VWAP ~ cumulative(price * vol) / cumulative(vol) — if volume absent, fallback to simple average.
    if "volume" in df.columns and not df["volume"].isna().all():
        pv = (df["price"] * df["volume"]).fillna(0)
        cum_pv = pv.cumsum()
        cum_vol = df["volume"].fillna(0).cumsum()
        # avoid division by zero
        mask = cum_vol > 0
        vwap = pd.Series(np.nan, index=df.index)
        vwap[mask] = (cum_pv[mask] / cum_vol[mask])
        # last VWAP value
        return vwap
    else:
        # fallback to rolling mean as proxy
        return df["price"].expanding().mean()

def compute_trend_strength(df):
    """
    Input: df with columns ['datetime','price','volume' optional]
    Uses the intraday window up to 11:30 (assumed caller clipped)
    Returns: dict {score:0-100, class:..., components: {...}}
    Approach:
      - ema_short_slope : slope of EMA(9) over the window normalized
      - ema_long_slope  : slope of EMA(26) over the window normalized
      - vwap_gap_pct    : (last_price - vwap_last)/vwap_last
      - momentum_pct    : (last - first)/first over window
      - vol_confirm     : normalized avg(volume) (if available) vs its median
      - volatility_penalty: higher ATR reduces score
    All components scaled and combined with weights.
    """
    out = {"score": None, "class": None, "components": {}}
    if df is None or df.empty:
        return out

    working = df.copy().reset_index(drop=True)
    # ensure price numeric
    working["price"] = pd.to_numeric(working["price"], errors="coerce")
    working = working.dropna(subset=["price"])
    if working.empty:
        return out

    # consider only up to 11:30 (should have been clipped by caller)
    # compute EMAs
    ema_short = compute_ema(working["price"], span=9)
    ema_long = compute_ema(working["price"], span=26)
    # slope proxies: difference between last and median (or first) normalized by price range
    last_price = float(working["price"].iloc[-1])
    first_price = float(working["price"].iloc[0])
    price_range = max(1e-6, working["price"].max() - working["price"].min())

    ema9_slope = (ema_short.iloc[-1] - ema_short.iloc[0]) / price_range  # normalized
    ema26_slope = (ema_long.iloc[-1] - ema_long.iloc[0]) / price_range

    # VWAP gap (last price vs VWAP)
    vwap_series = compute_vwap(working)
    vwap_last = float(vwap_series.iloc[-1]) if not vwap_series.isna().all() else np.nan
    if not np.isnan(vwap_last) and vwap_last != 0:
        vwap_gap_pct = (last_price - vwap_last) / vwap_last
    else:
        vwap_gap_pct = (last_price - first_price) / max(1e-6, first_price)

    # session momentum
    momentum_pct = (last_price - first_price) / max(1e-6, first_price)

    # volatility (ATR-like): use rolling range between consecutive prices
    diffs = np.abs(working["price"].diff().fillna(0))
    avg_move = float(diffs.mean())
    # normalize volatility: smaller avg_move -> stronger (less noisy) -> lower penalty
    vol_norm = avg_move / max(1e-6, price_range)

    # volume confirmation
    if "volume" in working.columns and not working["volume"].isna().all():
        avg_vol = float(working["volume"].mean())
        med_vol = float(np.nanmedian(working["volume"].values))
        # vol_conf = avg_vol / (med_vol + 1e-6)
        # normalize vol_conf to a reasonable range (capped)
        vol_conf = (avg_vol / (med_vol + 1e-6)) if med_vol > 0 else 1.0
        vol_conf = min(vol_conf, 5.0)  # cap extreme
    else:
        vol_conf = 1.0  # neutral if unknown

    # Scale components into 0-1 favorable scores
    # ema contribution: positive slope increases score
    ema_score = 0.0
    # combine short & long slopes (greater weight to short)
    # map slope values (which may be +/-) to [0,1] using sigmoid-like mapping
    def slope_to_score(s):
        # s is normalized slope; small s around 0 -> 0.5 ; large positive -> towards 1; large negative -> towards 0
        return 1.0 / (1.0 + np.exp(-4 * s))  # steep logistic

    ema_short_score = slope_to_score(ema9_slope)
    ema_long_score = slope_to_score(ema26_slope)
    ema_score = 0.65 * ema_short_score + 0.35 * ema_long_score

    # vwap score: being above VWAP is bullish (map vwap_gap_pct to score)
    # typical vwap_gap_pct values are small; we squash using tanh
    vwap_score = 0.5 + 0.5 * np.tanh(6 * vwap_gap_pct)  # maps negative to <0.5, positive to >0.5

    # momentum score: strong positive momentum -> higher
    momentum_score = 0.5 + 0.5 * np.tanh(4 * momentum_pct)

    # volume score: vol_conf >1 -> supports the move; map to 0-1
    vol_score = 1.0 - 1.0/(1.0 + (vol_conf-1.0)) if vol_conf >= 1.0 else vol_conf/2.0 + 0.25
    vol_score = float(np.clip(vol_score, 0.0, 1.0))

    # volatility penalty: higher vol_norm reduces final score (map to [0,1] penalty)
    vol_penalty = float(np.clip(np.tanh(6 * vol_norm), 0.0, 1.0))

    # Compose final score: weighted sum of components (bullish orientation)
    # We want score in [0,100] where >60 bullish, <40 bearish, etc.
    w = {
        "ema": 0.38,
        "vwap": 0.28,
        "momentum": 0.20,
        "volume": 0.14
    }
    base_score = (w["ema"] * ema_score + w["vwap"] * vwap_score + w["momentum"] * momentum_score + w["volume"] * vol_score)
    # reduce by volatility penalty (makes trend weaker)
    score = base_score * (1.0 - 0.6 * vol_penalty)  # aggressive penalty factor
    # Convert to 0-100
    score_0_100 = float(np.clip(score * 100.0, 0.0, 100.0))

    # classification thresholds
    if score_0_100 >= 75:
        cl = "Strong Bullish"
    elif score_0_100 >= 60:
        cl = "Mild Bullish"
    elif score_0_100 >= 45:
        cl = "Neutral"
    elif score_0_100 >= 30:
        cl = "Mild Bearish"
    else:
        cl = "Strong Bearish"

    out["score"] = round(score_0_100, 1)
    out["class"] = cl
    out["components"] = {
        "ema9_slope": float(ema9_slope),
        "ema26_slope": float(ema26_slope),
        "ema_short_score": float(round(ema_short_score, 3)),
        "ema_long_score": float(round(ema_long_score, 3)),
        "vwap_gap_pct": float(round(vwap_gap_pct, 5)),
        "vwap_score": float(round(vwap_score, 3)),
        "momentum_pct": float(round(momentum_pct, 5)),
        "momentum_score": float(round(momentum_score, 3)),
        "vol_conf": float(round(vol_conf, 3)),
        "vol_score": float(round(vol_score, 3)),
        "vol_norm": float(round(vol_norm, 6)),
        "volatility_penalty": float(round(vol_penalty, 3))
    }
    return out

# ---------------------------
# UI and plotting
# ---------------------------
st.title("📈 NIFTY — Intraday charts to 11:30 + Trend Strength (NSE)")

# params
cols_ui = st.columns([1, 3])
with cols_ui[0]:
    n_days = st.number_input("Days to show", min_value=5, max_value=60, value=30, step=5)
    show_next = st.checkbox("Show next scheduled refresh time", value=True)
    force_refresh = st.button("Force live fetch for today")
with cols_ui[1]:
    st.write("Notes: NSE public chart API usually returns current-day intraday only. Past days appear if cached in the `.nse_cache` folder.")

if show_next:
    sec = seconds_until_next_target((9,17))
    next_time = now_ist() + datetime.timedelta(seconds=sec)
    st.write(f"Next scheduled refresh (approx): {next_time.strftime('%Y-%m-%d %H:%M:%S %Z')} (in {sec} seconds)")

# dates list (most recent first displayed left-to-right)
dates = [datetime.date.today() - datetime.timedelta(days=i) for i in range(n_days)]
dates = sorted(dates)

# fetch data for each date (cache logic inside)
days_data = []
with st.spinner("Loading cached data / fetching today's intraday..."):
    for d in dates:
        df = get_intraday_for_date(d)
        # ensure datetime column exists and is datetime type
        if df is not None and "datetime" in df.columns:
            # convert to pandas datetime (may already be)
            df["datetime"] = pd.to_datetime(df["datetime"])
            # make sure there's price column
            if "price" not in df.columns and "close" in df.columns:
                df = df.rename(columns={"close":"price"})
        days_data.append((d, df))

# utility to clip to 09:15-11:30 IST
def clip_to_session_df(df):
    if df is None or df.empty:
        return df
    # ensure time in IST: our saved datetimes are naive but assumed IST; treat as such
    ser = pd.to_datetime(df["datetime"])
    # get time part
    mask = ser.dt.time >= datetime.time(9,15)
    mask &= ser.dt.time <= datetime.time(11,30)
    return df.loc[mask].reset_index(drop=True)

# layout grid
cols_per_row = 5
rows = (len(days_data) + cols_per_row - 1) // cols_per_row

st.header("Mini-charts + Trend strength (09:15 → 11:30)")

for r in range(rows):
    cols = st.columns(cols_per_row)
    for c in range(cols_per_row):
        idx = r*cols_per_row + c
        if idx >= len(days_data):
            continue
        d, df = days_data[idx]
        with cols[c]:
            st.subheader(d.strftime("%Y-%m-%d"))
            if df is None or df.empty:
                st.info("No intraday data (not cached and NSE provides only current day).")
                continue
            clipped = clip_to_session_df(df)
            if clipped is None or clipped.empty:
                st.info("No data in 09:15–11:30 window")
                continue

            # compute trend strength
            ts = compute_trend_strength(clipped)

            # badge color mapping
            cls = ts.get("class", "Unknown")
            score = ts.get("score", None)
            if cls == "Strong Bullish":
                color_box = "#1f8b4c"
            elif cls == "Mild Bullish":
                color_box = "#65c466"
            elif cls == "Neutral":
                color_box = "#bfbf00"
            elif cls == "Mild Bearish":
                color_box = "#ff8a5b"
            elif cls == "Strong Bearish":
                color_box = "#d9534f"
            else:
                color_box = "#999999"

            # show score + class
            cols_badge = st.columns([2, 5])
            with cols_badge[0]:
                # numeric badge
                st.markdown(
                    f"<div style='background:{color_box};color:white;padding:8px;border-radius:6px;text-align:center'>"
                    f"<strong style='font-size:18px'>{score if score is not None else 'NA'}</strong><br><small>Trend</small></div>",
                    unsafe_allow_html=True
                )
            with cols_badge[1]:
                st.markdown(f"**{cls}**")
                # small components summary
                comps = ts.get("components", {})
                if comps:
                    st.caption(
                        f"EMA9_slope={comps.get('ema9_slope'):.4f}  "
                        f"VWAP_gap={comps.get('vwap_gap_pct'):.4f}  "
                        f"Momentum={comps.get('momentum_pct'):.4f}"
                    )

            # plot price + volume
            fig, (ax0, ax1) = plt.subplots(2,1, figsize=(4,2.6), gridspec_kw={'height_ratios':[2,0.8]})
            times = pd.to_datetime(clipped["datetime"])
            xlabels = times.dt.strftime("%H:%M")
            ax0.plot(xlabels, clipped["price"], linewidth=1.0)
            # overlay VWAP
            try:
                v = compute_vwap(clipped)
                ax0.plot(xlabels, v, linewidth=0.9, linestyle="--", alpha=0.8)
            except Exception:
                pass
            ax0.set_xticks([xlabels.iloc[0], xlabels.iloc[len(xlabels)//2], xlabels.iloc[-1]])
            ax0.tick_params(axis='x', rotation=45, labelsize=7)
            ax0.set_ylabel("Price", fontsize=8)

            # volume
            if "volume" in clipped.columns and not clipped["volume"].isna().all():
                ax1.bar(xlabels, clipped["volume"].fillna(0), width=0.6)
            ax1.set_xticks([xlabels.iloc[0], xlabels.iloc[len(xlabels)//2], xlabels.iloc[-1]])
            ax1.tick_params(axis='x', rotation=45, labelsize=7)
            ax1.set_ylabel("Vol", fontsize=8)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

st.write("---")
st.header("How Trend Strength is computed (short)")
st.markdown("""
The score (0–100) is a weighted composite of:
- **EMA slopes** (fast + slow): measures short & medium momentum.
- **VWAP gap**: whether price is above/below value area (supports trend).
- **Session momentum**: last vs first price in the 09:15–11:30 window.
- **Volume confirmation**: whether volume supports the price move.
- **Volatility penalty**: noisy, choppy moves reduce the score.

**Interpretation**
- 75–100: Strong Bullish — high confidence trend to the upside
- 60–74: Mild Bullish
- 45–59: Neutral
- 30–44: Mild Bearish
- 0–29: Strong Bearish

Notes & caveats:
- This is a heuristic composite for intraday trend *strength*, intended to help prioritise which days show clear directional bias early (till 11:30). It is **not** a trading signal by itself — combine with price action, risk management and confirmation.
- Metric depends on data quality (volumes especially). If volume is missing for a day, volume-based component becomes neutral.
""")

st.success("Trend strength detection added. The app caches today's intraday in `.nse_cache`; over days this builds a local history.")
