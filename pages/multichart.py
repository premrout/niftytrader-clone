"""
Streamlit app: NIFTY intraday (09:15->11:30) trend strength
- Uses Yahoo Finance ^NSEI (5m interval, 60d range) — reliable & free
- Caches per-date CSVs under .yahoo_cache (fetch once per day or on demand)
- Computes trend strength using user's original logic

Usage: `streamlit run streamlit_app_yahoo.py`

Notes:
- Fetching 60d at 5m interval returns ~60 * (6.5*12) ~ a few thousand rows; Yahoo is stable.
- The app will fetch when you click "Force refresh cache" or when today's file missing.
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

st.set_page_config(layout="wide", page_title="NIFTY — Intraday (Yahoo 5m) Trend Strength")

# ---------------------------
# Config
# ---------------------------
CACHE_DIR = ".yahoo_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
SYMBOL = "^NSEI"
INTERVAL = "5m"  # chosen per user's A option
RANGE = "60d"
IST = pytz.timezone("Asia/Kolkata")

# ---------------------------
# Yahoo fetch helpers
# ---------------------------

def fetch_intraday_yahoo(interval="5m", range_days="60d", symbol=SYMBOL, timeout=15):
    """
    Fetch intraday series from Yahoo Finance chart API.
    Returns DataFrame with columns: datetime (tz-aware IST), price, volume
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range_days}"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    # navigate to result
    result = j.get("chart", {}).get("result")
    if not result:
        raise ValueError("No result in Yahoo response")
    res0 = result[0]
    timestamps = res0.get("timestamp")
    if timestamps is None:
        # Possibly an extended result under indicators -> adjclose only (rare)
        raise ValueError("No timestamps in Yahoo response")

    indicators = res0.get("indicators", {}).get("quote", [])[0]
    closes = indicators.get("close")
    volumes = indicators.get("volume")

    df = pd.DataFrame({
        "datetime": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Asia/Kolkata"),
        "price": closes,
        "volume": volumes
    })
    # drop rows where price is None (market closed periods)
    df = df.dropna(subset=["price"]).reset_index(drop=True)
    return df


def split_and_cache_by_date(df, cache_dir=CACHE_DIR):
    """
    Split DataFrame into per-date CSVs (date in IST) and save into cache.
    Overwrites existing files for the dates present in df.
    Returns list of dates saved.
    """
    df = df.copy()
    df["date"] = df["datetime"].dt.tz_convert("Asia/Kolkata").dt.date
    saved = []
    for d, grp in df.groupby("date"):
        path = os.path.join(cache_dir, f"nifty_{d.isoformat()}.csv")
        # ensure datetime stored in ISO format with timezone info
        out = grp.drop(columns=["date"]).copy()
        out.to_csv(path, index=False)
        saved.append(d)
    return saved


# ---------------------------
# Cache helpers
# ---------------------------

def cache_path_for_date(date_obj, cache_dir=CACHE_DIR):
    return os.path.join(cache_dir, f"nifty_{date_obj.isoformat()}.csv")


def load_cached_date(date_obj, cache_dir=CACHE_DIR):
    path = cache_path_for_date(date_obj, cache_dir)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        # ensure datetime parsed
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
            # stored timezone info may be present; convert to tz-aware IST
            try:
                # attempt to parse any tz info
                df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
            except Exception:
                # if already tz-aware or naive, attach IST
                df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize("Asia/Kolkata")
        else:
            # fallback: construct naive datetimes if first column looks like a timestamp
            df["datetime"] = pd.to_datetime(df.iloc[:,0]).dt.tz_localize("Asia/Kolkata")
        # ensure price and volume columns present
        if "price" not in df.columns and "close" in df.columns:
            df = df.rename(columns={"close":"price"})
        if "volume" not in df.columns:
            df["volume"] = np.nan
        return df
    except Exception:
        return None


def ensure_cache_for_recent_days(days=60):
    """
    If multiple dates missing, fetch the 60d Yahoo dataset once and split into per-date cache files.
    Returns set of dates available in cache after the operation.
    """
    # Determine which dates already cached
    today = datetime.date.today()
    dates = [today - datetime.timedelta(days=i) for i in range(days)]
    missing = [d for d in dates if not os.path.exists(cache_path_for_date(d))]
    if not missing:
        return set(dates)

    # Fetch once and split
    try:
        df_all = fetch_intraday_yahoo(interval=INTERVAL, range_days=RANGE)
    except Exception as e:
        st.warning(f"Failed to fetch Yahoo intraday: {e}")
        return set([d for d in dates if os.path.exists(cache_path_for_date(d))])

    saved = split_and_cache_by_date(df_all)
    return set(saved)


# ---------------------------
# Time helpers
# ---------------------------

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
# Trend strength computation (unchanged logic, slightly adapted)
# ---------------------------

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def compute_vwap(df):
    if "volume" in df.columns and not df["volume"].isna().all():
        pv = (df["price"] * df["volume"]).fillna(0)
        cum_pv = pv.cumsum()
        cum_vol = df["volume"].fillna(0).cumsum()
        mask = cum_vol > 0
        vwap = pd.Series(np.nan, index=df.index)
        vwap[mask] = (cum_pv[mask] / cum_vol[mask])
        return vwap
    else:
        return df["price"].expanding().mean()


def compute_trend_strength(df):
    out = {"score": None, "class": None, "components": {}}
    if df is None or df.empty:
        return out

    working = df.copy().reset_index(drop=True)
    working["price"] = pd.to_numeric(working["price"], errors="coerce")
    working = working.dropna(subset=["price"])
    if working.empty:
        return out

    ema_short = compute_ema(working["price"], span=9)
    ema_long = compute_ema(working["price"], span=26)
    last_price = float(working["price"].iloc[-1])
    first_price = float(working["price"].iloc[0])
    price_range = max(1e-6, working["price"].max() - working["price"].min())

    ema9_slope = (ema_short.iloc[-1] - ema_short.iloc[0]) / price_range
    ema26_slope = (ema_long.iloc[-1] - ema_long.iloc[0]) / price_range

    vwap_series = compute_vwap(working)
    vwap_last = float(vwap_series.iloc[-1]) if not vwap_series.isna().all() else np.nan
    if not np.isnan(vwap_last) and vwap_last != 0:
        vwap_gap_pct = (last_price - vwap_last) / vwap_last
    else:
        vwap_gap_pct = (last_price - first_price) / max(1e-6, first_price)

    momentum_pct = (last_price - first_price) / max(1e-6, first_price)

    diffs = np.abs(working["price"].diff().fillna(0))
    avg_move = float(diffs.mean())
    vol_norm = avg_move / max(1e-6, price_range)

    if "volume" in working.columns and not working["volume"].isna().all():
        avg_vol = float(working["volume"].mean())
        med_vol = float(np.nanmedian(working["volume"].values))
        vol_conf = (avg_vol / (med_vol + 1e-6)) if med_vol > 0 else 1.0
        vol_conf = min(vol_conf, 5.0)
    else:
        vol_conf = 1.0

    def slope_to_score(s):
        return 1.0 / (1.0 + np.exp(-4 * s))

    ema_short_score = slope_to_score(ema9_slope)
    ema_long_score = slope_to_score(ema26_slope)
    ema_score = 0.65 * ema_short_score + 0.35 * ema_long_score

    vwap_score = 0.5 + 0.5 * np.tanh(6 * vwap_gap_pct)
    momentum_score = 0.5 + 0.5 * np.tanh(4 * momentum_pct)

    if vol_conf >= 1.0:
        vol_score = 1.0 - 1.0/(1.0 + (vol_conf-1.0))
    else:
        vol_score = vol_conf/2.0 + 0.25
    vol_score = float(np.clip(vol_score, 0.0, 1.0))

    vol_penalty = float(np.clip(np.tanh(6 * vol_norm), 0.0, 1.0))

    w = {"ema": 0.38, "vwap": 0.28, "momentum": 0.20, "volume": 0.14}
    base_score = (w["ema"] * ema_score + w["vwap"] * vwap_score + w["momentum"] * momentum_score + w["volume"] * vol_score)
    score = base_score * (1.0 - 0.6 * vol_penalty)
    score_0_100 = float(np.clip(score * 100.0, 0.0, 100.0))

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
# UI
# ---------------------------

st.title("📈 NIFTY — Intraday to 11:30 + Trend Strength (Yahoo 5m, cached)")

cols_ui = st.columns([1, 3])
with cols_ui[0]:
    n_days = st.number_input("Days to show", min_value=5, max_value=60, value=30, step=5)
    show_next = st.checkbox("Show next scheduled refresh time", value=True)
    force_refresh = st.button("Force refresh cache (fetch 60d from Yahoo)")
with cols_ui[1]:
    st.write("Notes: Using Yahoo Finance ^NSEI with 5-minute resolution (60 days). Cached per-date in `.yahoo_cache`.")

if show_next:
    sec = seconds_until_next_target((9,17))
    next_time = now_ist() + datetime.timedelta(seconds=sec)
    st.write(f"Next scheduled refresh (approx): {next_time.strftime('%Y-%m-%d %H:%M:%S %Z')} (in {sec} seconds)")

# If user forces refresh, fetch full 60d and split
if force_refresh:
    with st.spinner("Fetching 60 days of 5-minute data from Yahoo and caching per date..."):
        try:
            df_all = fetch_intraday_yahoo(interval=INTERVAL, range_days=RANGE)
            saved_dates = split_and_cache_by_date(df_all)
            st.success(f"Cached data for {len(saved_dates)} dates (most recent: {max(saved_dates) if saved_dates else 'none'}).")
        except Exception as e:
            st.error(f"Failed to refresh cache: {e}")

# Ensure cache exists for recent window (try background fill but avoid aggressive network ops)
with st.spinner("Ensuring recent dates cached (will fetch from Yahoo if needed)..."):
    available_dates = ensure_cache_for_recent_days(days=60)

# Prepare list of dates to display
dates = [datetime.date.today() - datetime.timedelta(days=i) for i in range(n_days)]
dates = sorted(dates)

# Load cached data for each requested date
days_data = []
for d in dates:
    df = load_cached_date(d)
    days_data.append((d, df))

# utility to clip to 09:15-11:30 IST

def clip_to_session_df(df):
    if df is None or df.empty:
        return df
    ser = pd.to_datetime(df["datetime"]).dt.tz_convert("Asia/Kolkata")
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
                st.info("No intraday data (not cached). Try 'Force refresh cache' to fetch 60d from Yahoo.")
                continue
            clipped = clip_to_session_df(df)
            if clipped is None or clipped.empty:
                st.info("No data in 09:15–11:30 window")
                continue

            ts = compute_trend_strength(clipped)

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

            cols_badge = st.columns([2, 5])
            with cols_badge[0]:
                st.markdown(
                    f"<div style='background:{color_box};color:white;padding:8px;border-radius:6px;text-align:center'>"
                    f"<strong style='font-size:18px'>{score if score is not None else 'NA'}</strong><br><small>Trend</small></div>",
                    unsafe_allow_html=True
                )
            with cols_badge[1]:
                st.markdown(f"**{cls}**")
                comps = ts.get("components", {})
                if comps:
                    st.caption(
                        f"EMA9_slope={comps.get('ema9_slope'):.4f}  "
                        f"VWAP_gap={comps.get('vwap_gap_pct'):.4f}  "
                        f"Momentum={comps.get('momentum_pct'):.4f}"
                    )

            fig, (ax0, ax1) = plt.subplots(2,1, figsize=(4,2.6), gridspec_kw={'height_ratios':[2,0.8]})
            times = pd.to_datetime(clipped["datetime"]).dt.tz_convert("Asia/Kolkata")
            xlabels = times.dt.strftime("%H:%M")
            ax0.plot(xlabels, clipped["price"], linewidth=1.0)
            try:
                v = compute_vwap(clipped)
                ax0.plot(xlabels, v, linewidth=0.9, linestyle="--", alpha=0.8)
            except Exception:
                pass
            ax0.set_xticks([xlabels.iloc[0], xlabels.iloc[len(xlabels)//2], xlabels.iloc[-1]])
            ax0.tick_params(axis='x', rotation=45, labelsize=7)
            ax0.set_ylabel("Price", fontsize=8)

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

st.success("Trend strength detection added. The app caches per-day intraday in `.yahoo_cache`.")
