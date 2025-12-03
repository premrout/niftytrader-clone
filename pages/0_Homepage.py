# app.py
"""
Streamlit app: NIFTY last 30 trading days intraday charts (open -> 11:30)
- Attempts to fetch minute OHLCV for each trading day
- Shows 30 mini-charts on a single page
- Auto-refresh scheduled for next 09:00 IST and 17:00 IST using st_autorefresh
- Provides volume analysis and suggested entry/exit windows
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta
import pytz
import math
import io
import requests
import sys
import traceback

# --- Optional: if you want to use an NSE-focused library (preferred if available).
# We'll attempt to use openchart (if installed) otherwise fallback to yfinance.
try:
    import openchart  # optional library; if available use it (github: marketcalls/openchart)
    HAS_OPENCHART = True
except Exception:
    HAS_OPENCHART = False

# fallback
try:
    import yfinance as yf
    HAS_YFINANCE = True
except Exception:
    HAS_YFINANCE = False

st.set_page_config(page_title="NIFTY — 30-day intraday (to 11:30)", layout="wide")

# -------------------------
# Utility: IST timezone helpers
# -------------------------
IST = pytz.timezone("Asia/Kolkata")

def now_ist():
    return datetime.now(IST)

def seconds_until_next_target(hour_targets=(9,17)):
    """Return seconds until next target time (in IST). Targets are hour values (24h)."""
    now = now_ist()
    today = now.date()
    # candidate datetimes:
    candidates = []
    for h in hour_targets:
        target_dt = IST.localize(datetime.combine(today, time(h,0,0)))
        if target_dt <= now:
            target_dt = target_dt + timedelta(days=1)
        candidates.append(target_dt)
    next_target = min(candidates)
    delta = next_target - now
    return int(delta.total_seconds())

# -------------------------
# Data fetchers
# -------------------------
# User-specified params
SYMBOL_FOR_YFINANCE = "^NSEI"  # Yahoo ticker for Nifty 50 index (used when falling back)

@st.cache_data(ttl=60*60)  # cache for 1 hour
def fetch_intraday_for_date(date_obj):
    """
    Fetch minute-level intraday OHLCV for NIFTY for a single date.
    - Tries openchart (if available) to get 1m data from NSE
    - Otherwise tries yfinance (may return 1m only for recent days)
    Returns DataFrame with index as timestamp (localized to IST) and columns: ['Open','High','Low','Close','Volume']
    """
    date_str = date_obj.strftime("%Y-%m-%d")
    # 1) Try openchart if available
    if HAS_OPENCHART:
        try:
            # the library's API: openchart.history(symbol, timeframe='1m', start=..., end=...) -- adjust if library differs
            # We'll try to be defensive.
            df = openchart.history('NIFTY 50', timeframe='1m', start=date_str, end=date_str)
            if df is not None and not df.empty:
                # ensure index tz -> IST
                if df.index.tzinfo is None:
                    df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
                else:
                    df.index = df.index.tz_convert('Asia/Kolkata')
                return df[['Open','High','Low','Close','Volume']]
        except Exception:
            # fall through to next method
            st.write("OpenChart fetch failed for", date_str)
            st.write(traceback.format_exc())

    # 2) Try yfinance fallback
    if HAS_YFINANCE:
        try:
            # yfinance fetch; 1m interval usually available for up to last 7 days only on free endpoints.
            start = date_obj.strftime("%Y-%m-%d")
            end = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
            ticker = yf.Ticker(SYMBOL_FOR_YFINANCE)
            # yfinance: history(interval='1m', start=..., end=...)
            df = ticker.history(interval="1m", start=start, end=end, auto_adjust=False, prepost=False)
            if df is not None and not df.empty:
                # Yahoo returns index in UTC or local - convert to IST
                if df.index.tzinfo is None:
                    df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
                else:
                    df.index = df.index.tz_convert('Asia/Kolkata')
                # Ensure required columns
                df = df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"})
                return df[['Open','High','Low','Close','Volume']]
        except Exception:
            st.write("yfinance fetch failed for", date_str)
            st.write(traceback.format_exc())

    # 3) As last resort, return empty
    return pd.DataFrame(columns=['Open','High','Low','Close','Volume'])

@st.cache_data(ttl=60*60*24)
def get_last_n_trading_days(n=30):
    """
    Return a list of last n trading dates (business days excluding weekends). This is a simple approximation:
    we will pick last n weekdays; if you need exchange holidays removed, integrate exchange calendar.
    """
    days = []
    cur = now_ist().date()
    while len(days) < n:
        if cur.weekday() < 5:  # Mon-Fri
            days.append(cur)
        cur = cur - timedelta(days=1)
    days = sorted(days)
    return days

# -------------------------
# Analysis functions
# -------------------------
def clip_to_session(df):
    """Keep rows from market open (09:15) to 11:30 inclusive (IST)"""
    if df.empty:
        return df
    # ensure index tz-aware in IST
    idx = df.index.tz_convert('Asia/Kolkata')
    mask = (idx.time >= time(9,15)) & (idx.time <= time(11,30))
    return df.loc[mask]

def aggregate_per_minute(df):
    """Return per-minute OHLCV (if input is higher granularity). Here we assume df is already 1m"""
    return df.resample('1T').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()

def compute_volume_profile(all_day_frames):
    """
    all_day_frames: list of DataFrames (each with minute index localized to IST in same date) clipped to 09:15-11:30
    Returns avg_volume_by_minute: Series indexed by minute-of-day string 'HH:MM' -> average volume
    """
    frames = []
    for df in all_day_frames:
        if df.empty: continue
        df2 = df.copy()
        # minute key
        df2['minute_key'] = df2.index.strftime('%H:%M')
        frames.append(df2[['minute_key','Volume']])
    if not frames:
        return pd.Series(dtype=float)
    big = pd.concat(frames)
    avg = big.groupby('minute_key')['Volume'].mean().sort_index()
    return avg

def detect_entry_exit(avg_volume_series, top_percentile=90, taper_threshold_pct=0.4):
    """
    Suggest entry and exit times:
    - entry: minutes where average volume is above top_percentile percentile (e.g., 90th)
    - exit: heuristic: minutes after the main peak where volume drops below taper_threshold_pct * peak_volume
    Returns dict: {'peak_minutes':[...], 'suggested_entry': first_peak_minute, 'suggested_exit': minute_when_volume_drops}
    """
    if avg_volume_series.empty:
        return {}
    peak_vol = avg_volume_series.max()
    threshold = np.percentile(avg_volume_series.values, top_percentile)
    peak_minutes = avg_volume_series[avg_volume_series >= threshold].index.tolist()
    # take earliest peak minute as entry
    suggested_entry = peak_minutes[0] if peak_minutes else avg_volume_series.idxmax()
    # find exit: after the peak, find first minute where volume <= taper_threshold_pct * peak_vol
    try:
        idx_peak = list(avg_volume_series.index).index(suggested_entry)
    except ValueError:
        idx_peak = avg_volume_series.idxmax()
        idx_peak = list(avg_volume_series.index).index(idx_peak)
    exit_minute = None
    for i in range(idx_peak+1, len(avg_volume_series)):
        if avg_volume_series.iloc[i] <= taper_threshold_pct * peak_vol:
            exit_minute = avg_volume_series.index[i]
            break
    if exit_minute is None:
        # fallback: last minute in the window
        exit_minute = avg_volume_series.index[-1]
    return {
        "peak_minutes": peak_minutes,
        "suggested_entry": suggested_entry,
        "suggested_exit": exit_minute,
        "peak_volume": float(peak_vol),
        "threshold_for_peak": float(threshold)
    }

# -------------------------
# UI & Main flow
# -------------------------
st.title("NIFTY last 30 days — intraday charts (open → 11:30)")

col1, col2 = st.columns([1,3])
with col1:
    st.markdown("**Parameters**")
    n_days = st.number_input("Days to show", min_value=5, max_value=60, value=30, step=5)
    refresh_note = st.checkbox("Show next auto-refresh time", value=True)
    re_fetch = st.button("Force re-fetch data now (may take a minute)")

with col2:
    st.markdown("**Notes**")
    st.write("""
    - The app attempts to fetch minute-level data (1m) for NIFTY from an NSE-focused library (if available).
    - If that fails, it falls back to Yahoo Finance minute data (may be limited to recent days).
    - For guaranteed historical minute/tick access consider a paid data vendor or NSE's paid products.
    """)

# calculate next refresh interval (seconds)
sec_to_next = seconds_until_next_target((9,17))
if refresh_note:
    next_time = now_ist() + timedelta(seconds=sec_to_next)
    st.write(f"Next scheduled refresh at (IST): **{next_time.strftime('%Y-%m-%d %H:%M:%S')}** — in {sec_to_next} seconds.")

# trigger auto-refresh: st_autorefresh will rerun app after given interval
# set a max interval of 24 hours; if sec_to_next is 0 small, set to 5 seconds min
interval_seconds = max(5, min(sec_to_next, 24*3600))
count = st.experimental_data_editor  # avoid linter warning; not used
from streamlit_autorefresh import st_autorefresh
# The streamlit_autorefresh helper: pip install streamlit-autorefresh
st_autorefresh(interval=interval_seconds * 1000, key="datarefresh")

# main data fetch & plotting
with st.spinner("Fetching data and preparing charts..."):
    dates = get_last_n_trading_days(n_days)
    day_frames = []
    failed_days = []
    for d in dates:
        try:
            df = fetch_intraday_for_date(pd.to_datetime(d))
            df_clip = clip_to_session(df)
            if df_clip.empty:
                failed_days.append(d)
            day_frames.append((d, df_clip))
        except Exception as e:
            failed_days.append(d)
            day_frames.append((d, pd.DataFrame()))
    # Analysis: compute average volume by minute
    clipped_frames = [f for (_, f) in day_frames if not f.empty]
    avg_vol = compute_volume_profile(clipped_frames)
    entry_exit = detect_entry_exit(avg_vol)

# show analysis
st.header("Volume analysis (09:15 → 11:30 IST)")
if avg_vol.empty:
    st.write("No minute-level volume data available. Check data source or try force re-fetch.")
else:
    # display top 5 high-volume minutes
    top5 = avg_vol.sort_values(ascending=False).head(5)
    st.metric("Peak average volume (minute)", f"{int(entry_exit.get('peak_volume', 0))}")
    st.write("Top 5 minutes by average volume (HH:MM → avg volume):")
    st.dataframe(top5.reset_index().rename(columns={'minute_key':'Minute','Volume':'AvgVolume'}).head(10))

    st.markdown(f"**Suggested entry minute:** `{entry_exit.get('suggested_entry')}`  \n**Suggested exit minute:** `{entry_exit.get('suggested_exit')}`")
    st.write("Heuristic: entry is earliest minute in top 90th percentile by average volume. Exit is first minute after the peak where volume falls below 40% of the peak.")

# Plot 30 mini-charts on a single page in a grid
st.header("30 mini-charts (market open → 11:30 IST)")

# number of columns for grid
cols_per_row = 5
rows = math.ceil(len(day_frames) / cols_per_row)

for r in range(rows):
    cols = st.columns(cols_per_row)
    for c in range(cols_per_row):
        idx = r*cols_per_row + c
        if idx >= len(day_frames):
            continue
        d, df = day_frames[idx]
        with cols[c]:
            st.markdown(f"**{d.strftime('%Y-%m-%d')}**")
            if df.empty:
                st.write("No data")
                continue
            # simple price + volume chart
            fig, ax = plt.subplots(2,1, figsize=(3,2.5), gridspec_kw={'height_ratios':[2,1]}, dpi=100)
            ax0, ax1 = ax
            # Price line
            ax0.plot(df.index.time, df['Close'], linewidth=0.8)
            ax0.set_xticks([df.index.time[0], df.index.time[len(df.index)//2], df.index.time[-1]])
            ax0.tick_params(axis='x', labelrotation=45)
            ax0.set_ylabel("Price")
            ax0.set_title("")
            # Volume bars
            ax1.bar(df.index.time, df['Volume'], width=0.0008)
            ax1.set_xticks([df.index.time[0], df.index.time[len(df.index)//2], df.index.time[-1]])
            ax1.tick_params(axis='x', labelrotation=45)
            ax1.set_ylabel("Vol")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

st.write("---")
st.markdown("**Deployment notes & caveats**")
st.write("""
- NSE does not provide unlimited free intraday minute history via public APIs; for reliable, long-range minute/tick history you may need a paid feed or an authorized vendor. (This app attempts open libraries first; fallback to Yahoo may not have old minute data.) :contentReference[oaicite:1]{index=1}
- If you need guaranteed historical minute data for many past days, consider a data vendor or using a recorded dataset (e.g., Kaggle datasets, or buying historical ticks). :contentReference[oaicite:2]{index=2}
- The entry/exit detection here is a **heuristic** based purely on average minute volume across days; you should combine it with price movement, VWAP, and risk management rules before trading.
""")

st.markdown("**How to run on Streamlit Cloud**")
st.write("""
1. Create a GitHub repo with this `app.py`.  
2. Add `requirements.txt` with:  
