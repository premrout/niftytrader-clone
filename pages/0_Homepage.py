# app.py
"""
Streamlit app: NIFTY last N trading days intraday charts (09:15 -> 11:30 IST).
This is a simplified, parse-safe version intended to avoid SyntaxError issues.
It attempts to fetch data using yfinance as a fallback. If you prefer a
different data provider, integrate it in fetch_intraday_for_date().
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, time, timedelta
import pytz
import math
import traceback

# Optional imports: guard them so missing optional packages do not raise SyntaxError
HAS_YFINANCE = False
try:
    import yfinance as yf
    HAS_YFINANCE = True
except Exception:
    HAS_YFINANCE = False

# Optional autorefresh helper (streamlit-autorefresh). If missing, app still runs.
HAS_AUTORE = False
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTORE = True
except Exception:
    HAS_AUTORE = False

st.set_page_config(page_title="NIFTY — intraday 09:15→11:30", layout="wide")
IST = pytz.timezone("Asia/Kolkata")

def now_ist():
    return datetime.now(IST)

def seconds_until_next_target(hours=(9, 17)):
    now = now_ist()
    today = now.date()
    candidates = []
    for h in hours:
        try:
            dt = IST.localize(datetime.combine(today, time(h, 0, 0)))
        except Exception:
            dt = IST.localize(datetime.combine(today, time(h, 0, 0)))
        if dt <= now:
            dt = dt + timedelta(days=1)
        candidates.append(dt)
    next_target = min(candidates)
    return int((next_target - now).total_seconds())

@st.cache_data(ttl=60*60)
def get_last_n_weekdays(n=30):
    days = []
    cur = now_ist().date()
    while len(days) < n:
        if cur.weekday() < 5:
            days.append(cur)
        cur = cur - timedelta(days=1)
    days = sorted(days)
    return days

@st.cache_data(ttl=60*60)
def fetch_intraday_for_date(date_obj):
    """
    Return minute-level DataFrame for the given date (if available).
    Columns: ['Open','High','Low','Close','Volume'], index tz-aware in IST.
    This uses yfinance fallback; it may not return 1m for older dates.
    """
    empty = pd.DataFrame(columns=['Open','High','Low','Close','Volume'])
    if not HAS_YFINANCE:
        return empty
    try:
        start = date_obj.strftime("%Y-%m-%d")
        end = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
        ticker = yf.Ticker("^NSEI")  # Yahoo ticker for NIFTY index
        df = ticker.history(interval="1m", start=start, end=end, auto_adjust=False, prepost=False)
        if df is None or df.empty:
            return empty
        # Ensure tz: yfinance usually returns tz-aware UTC or naive; convert to IST
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
        else:
            df.index = df.index.tz_convert("Asia/Kolkata")
        # Keep only required columns and ensure names are consistent
        df = df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"})
        df = df[['Open','High','Low','Close','Volume']]
        return df
    except Exception:
        # Return empty if any error while fetching so the app remains parse-safe
        return empty

def clip_session(df):
    if df.empty:
        return df
    try:
        idx = df.index.tz_convert("Asia/Kolkata")
    except Exception:
        idx = df.index
    mask = (idx.time >= time(9,15)) & (idx.time <= time(11,30))
    return df.loc[mask]

def compute_avg_volume(frames):
    frames_clean = []
    for f in frames:
        if f is None or f.empty:
            continue
        tmp = f.copy()
        tmp['minute_key'] = tmp.index.strftime("%H:%M")
        frames_clean.append(tmp[['minute_key','Volume']])
    if not frames_clean:
        return pd.Series(dtype=float)
    big = pd.concat(frames_clean, ignore_index=True)
    avg = big.groupby('minute_key')['Volume'].mean().sort_index()
    return avg

def detect_entry_exit(avg_vol_series):
    result = {}
    if avg_vol_series.empty:
        return result
    arr = avg_vol_series.values
    try:
        threshold = np.percentile(arr, 90)
    except Exception:
        threshold = avg_vol_series.max()
    peaks = avg_vol_series[avg_vol_series >= threshold].index.tolist()
    suggested_entry = peaks[0] if peaks else avg_vol_series.idxmax()
    peak_vol = float(avg_vol_series.max())
    taper_threshold = 0.4 * peak_vol
    # find exit after entry
    idxs = list(avg_vol_series.index)
    try:
        start_idx = idxs.index(suggested_entry)
    except ValueError:
        start_idx = idxs.index(avg_vol_series.idxmax())
    exit_minute = None
    for i in range(start_idx+1, len(idxs)):
        if avg_vol_series.iloc[i] <= taper_threshold:
            exit_minute = idxs[i]
            break
    if exit_minute is None:
        exit_minute = idxs[-1]
    result = {
        "peak_volume": peak_vol,
        "threshold_for_peak": float(threshold),
        "suggested_entry": suggested_entry,
        "suggested_exit": exit_minute,
        "peak_minutes": peaks
    }
    return result

# ----------------- UI -----------------
st.title("NIFTY — last N days intraday (09:15 → 11:30 IST)")

params_col, notes_col = st.columns([1, 3])
with params_col:
    n_days = st.number_input("Days to show", min_value=5, max_value=60, value=30, step=5)
    show_next = st.checkbox("Show next scheduled refresh", value=True)
    force_refetch = st.button("Force re-fetch")
with notes_col:
    st.markdown("Notes:")
    st.write("- This is a simplified app. If yfinance is not installed or minute data isn't available for old days, some charts will show `No data`.")
    st.write("- For production historical minute data use a paid vendor or exchange data feed.")

# compute refresh interval and use st_autorefresh if available
sec_to_next = seconds_until_next_target((9,17))
if show_next:
    st.write("Next refresh in (approx):", str(timedelta(seconds=sec_to_next)))
if HAS_AUTORE:
    sec = max(5, min(sec_to_next, 24*3600))
    st_autorefresh(interval=sec * 1000, key="auto")

# fetch data
with st.spinner("Fetching data..."):
    days = get_last_n_weekdays(n_days)
    day_frames = []
    for d in days:
        df = fetch_intraday_for_date(pd.to_datetime(d))
        clipped = clip_session(df)
        day_frames.append((d, clipped))

    frames_only = [f for (_, f) in day_frames if not f.empty]
    avg_vol = compute_avg_volume(frames_only)
    entry_exit = detect_entry_exit(avg_vol)

st.header("Volume analysis (09:15–11:30 IST)")
if avg_vol.empty:
    st.info("No minute-volume data available. Try installing yfinance or use a data provider.")
else:
    st.metric("Peak avg volume (minute)", f"{int(entry_exit.get('peak_volume', 0))}")
    top5 = avg_vol.sort_values(ascending=False).head(5)
    st.dataframe(top5.reset_index().rename(columns={'minute_key':'Minute','Volume':'AvgVolume'}))
    st.markdown(f"**Suggested entry:** `{entry_exit.get('suggested_entry')}`  —  **Suggested exit:** `{entry_exit.get('suggested_exit')}`")

st.header("Mini-charts (09:15 → 11:30)")
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
            if df is None or df.empty:
                st.write("No data")
                continue
            # plot small chart
            fig, axes = plt.subplots(2, 1, figsize=(3, 2.4), gridspec_kw={'height_ratios':[2,1]})
            ax0, ax1 = axes
            # convert index to times for x axis
            times = df.index.strftime("%H:%M")
            ax0.plot(times, df['Close'], linewidth=0.9)
            ax0.set_xticks([times[0], times[len(times)//2], times[-1]])
            ax0.tick_params(axis='x', rotation=45, labelsize=7)
            ax0.set_ylabel("Price", fontsize=8)
            ax1.bar(times, df['Volume'], width=0.6)
            ax1.set_xticks([times[0], times[len(times)//2], times[-1]])
            ax1.tick_params(axis='x', rotation=45, labelsize=7)
            ax1.set_ylabel("Vol", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

st.write("---")
st.markdown("If you still see a SyntaxError after replacing the file, check logs for the file/line and paste that line here (I will help fix it).")
