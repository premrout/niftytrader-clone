# app.py
"""
Streamlit app: NIFTY last N trading days intraday charts (09:15 -> 11:30 IST)
This version is cleaned to avoid SyntaxError issues and kept minimal but functional.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, time, timedelta
import pytz
import math
import traceback

# Optional data sources
try:
    import yfinance as yf
    HAS_YFINANCE = True
except Exception:
    HAS_YFINANCE = False

st.set_page_config(page_title="NIFTY — 30-day intraday (to 11:30)", layout="wide")

IST = pytz.timezone("Asia/Kolkata")


def now_ist():
    return datetime.now(IST)


def seconds_until_next_target(hour_targets=(9, 17)):
    now = now_ist()
    today = now.date()
    candidates = []
    for h in hour_targets:
        candidate = IST.localize(datetime.combine(today, time(h, 0, 0)))
        if candidate <= now:
            candidate = candidate + timedelta(days=1)
        candidates.append(candidate)
    next_target = min(candidates)
    return int((next_target - now).total_seconds())


@st.cache_data(ttl=60 * 60)
def get_last_n_trading_days(n=30):
    """Return last n weekdays (approximation)."""
    days = []
    cur = now_ist().date()
    while len(days) < n:
        if cur.weekday() < 5:
            days.append(cur)
        cur = cur - timedelta(days=1)
    days = sorted(days)
    return days


@st.cache_data(ttl=60 * 60)
def fetch_intraday_for_date_yahoo(date_obj):
    """
    Try fetching intraday 1m data from yfinance for the given date.
    NOTE: yfinance 1m is usually available for recent days only.
    Returns a dataframe indexed by tz-aware datetime in IST with columns
    ['Open','High','Low','Close','Volume'] or empty DataFrame.
    """
    if not HAS_YFINANCE:
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    start = date_obj.strftime("%Y-%m-%d")
    end = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        ticker = yf.Ticker("^NSEI")  # Nifty index on Yahoo
        df = ticker.history(interval="1m", start=start, end=end, auto_adjust=False, prepost=False)
        if df is None or df.empty:
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
        else:
            df.index = df.index.tz_convert("Asia/Kolkata")
        df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception:
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])


def clip_to_session(df):
    """Keep only 09:15 -> 11:30 (inclusive) rows."""
    if df is None or df.empty:
        return df
    idx = df.index.tz_convert("Asia/Kolkata")
    mask = (idx.time >= time(9, 15)) & (idx.time <= time(11, 30))
    return df.loc[mask]


def compute_average_volume_by_minute(frames):
    """frames: list of dataframes already clipped to session."""
    if not frames:
        return pd.Series(dtype=float)
    parts = []
    for df in frames:
        if df is None or df.empty:
            continue
        s = pd.Series(df['Volume'].values, index=df.index.strftime("%H:%M"))
        parts.append(s)
    if not parts:
        return pd.Series(dtype=float)
    big = pd.concat(parts, axis=1).fillna(0)
    avg = big.mean(axis=1)
    # group by minute label so that ordering is by time
    avg.index.name = 'minute'
    avg = avg.groupby(avg.index).mean()
    return avg


def detect_entry_exit(avg_series, top_percentile=90, taper_ratio=0.4):
    if avg_series is None or avg_series.empty:
        return {}
    values = avg_series.values
    threshold = np.percentile(values, top_percentile)
    peak_vol = values.max()
    above = avg_series[avg_series >= threshold]
    if above.empty:
        suggested_entry = avg_series.idxmax()
    else:
        suggested_entry = above.index[0]
    # find exit after suggested_entry where volume <= taper_ratio * peak_vol
    idxs = list(avg_series.index)
    try:
        start_idx = idxs.index(suggested_entry)
    except ValueError:
        start_idx = np.argmax(values)
    exit_minute = None
    for i in range(start_idx + 1, len(idxs)):
        if avg_series.iloc[i] <= taper_ratio * peak_vol:
            exit_minute = avg_series.index[i]
            break
    if exit_minute is None:
        exit_minute = avg_series.index[-1]
    return {
        "suggested_entry": suggested_entry,
        "suggested_exit": exit_minute,
        "peak_volume": float(peak_vol),
        "threshold": float(threshold),
    }


# ---- UI ----
st.title("NIFTY — last trading days intraday (09:15 → 11:30 IST)")

col1, col2 = st.columns([1, 3])
with col1:
    n_days = st.number_input("Days to show", min_value=5, max_value=60, value=30, step=5)
    show_next = st.checkbox("Show next auto-refresh time", value=True)
    force_refetch = st.button("Force refetch")

with col2:
    st.write("This app fetches minute-level data (tries Yahoo fallback). For robust historical minute data use a vendor.")

# compute next refresh seconds (used only for user info)
sec_to_next = seconds_until_next_target((9, 17))
if show_next:
    next_dt = now_ist() + timedelta(seconds=sec_to_next)
    st.write(f"Next scheduled refresh (IST): {next_dt.strftime('%Y-%m-%d %H:%M:%S')} (in {sec_to_next} sec)")

# Try to auto-refresh using streamlit_autorefresh if available; else skip automatic refresh
try:
    from streamlit_autorefresh import st_autorefresh
    interval_seconds = max(5, min(sec_to_next, 24 * 3600))
    st_autorefresh(interval=interval_seconds * 1000, key="refresh")
except Exception:
    # If library missing, we simply don't auto-refresh
    pass

with st.spinner("Fetching data..."):
    days = get_last_n_trading_days(n_days)
    day_frames = []
    for d in days:
        df = fetch_intraday_for_date_yahoo(pd.to_datetime(d))
        dfc = clip_to_session(df)
        day_frames.append((d, dfc))

    clipped_frames = [df for (_, df) in day_frames if df is not None and not df.empty]
    avg_vol = compute_average_volume_by_minute(clipped_frames)
    entry_exit = detect_entry_exit(avg_vol)

st.header("Volume analysis (09:15 → 11:30 IST)")
if avg_vol.empty:
    st.write("No minute-level volume data available for these days.")
else:
    st.metric("Peak avg minute volume", f"{int(entry_exit.get('peak_volume', 0))}")
    top5 = avg_vol.sort_values(ascending=False).head(5)
    st.dataframe(top5.rename_axis("Minute").reset_index().rename(columns={0: "AvgVolume"}))
    st.markdown(f"**Suggested entry:** `{entry_exit.get('suggested_entry')}` — **Suggested exit:** `{entry_exit.get('suggested_exit')}`")
    st.write("Heuristic: entry = earliest minute in the top 90th percentile of average volume; exit = first minute after peak where volume drops below 40% of peak.")

st.header("Mini charts (open → 11:30)")
cols_per_row = 5
rows = math.ceil(len(day_frames) / cols_per_row)

for r in range(rows):
    cols = st.columns(cols_per_row)
    for c in range(cols_per_row):
        idx = r * cols_per_row + c
        if idx >= len(day_frames):
            continue
        d, df = day_frames[idx]
        with cols[c]:
            st.markdown(f"**{d.strftime('%Y-%m-%d')}**")
            if df.empty:
                st.write("No data")
                continue
            x = df.index.strftime("%H:%M")
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(3, 2.5), gridspec_kw={'height_ratios': [2, 1]})
            ax0.plot(x, df['Close'], linewidth=0.8)
            ax0.set_xticks([x.iloc[0], x.iloc[len(x) // 2], x.iloc[-1]])
            ax0.tick_params(axis='x', rotation=45)
            ax0.set_ylabel("Price")
            ax1.bar(x, df['Volume'], width=0.6)
            ax1.set_xticks([x.iloc[0], x.iloc[len(x) // 2], x.iloc[-1]])
            ax1.tick_params(axis='x', rotation=45)
            ax1.set_ylabel("Vol")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

st.write("---")
st.info("If you still get a SyntaxError after replacing the file, check the app logs (Manage app → Logs) for the exact file/line reported.")
