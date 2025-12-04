# streamlit_app_yahoo.py (Updated – Non-blocked Yahoo CSV API)

import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime, timedelta, date
import pytz
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="NIFTY Intraday (Yahoo 5m – 60 Days)", layout="wide")

CACHE_DIR = ".yahoo_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------
# Fetch 5-minute intraday using Yahoo CSV (NO RATE LIMITS)
# ---------------------------------------------------------------
def fetch_yahoo_intraday_5m(days=60):
    symbol = "^NSEI"

    end = int(time.time())
    start = int((datetime.utcnow() - timedelta(days=days)).timestamp())

    url = (
        f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
        f"?interval=5m&events=history&includeAdjustedClose=true"
        f"&period1={start}&period2={end}"
    )

    r = requests.get(url, timeout=10)
    r.raise_for_status()

    df = pd.read_csv(pd.compat.StringIO(r.text))

    # Convert timestamp
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
    df = df.rename(columns={"Close": "price"})

    df["date"] = df["Datetime"].dt.date
    return df

# ---------------------------------------------------------------
# Trend Score (Your Logic)
# ---------------------------------------------------------------
def calculate_trend_strength(df):
    if df is None or len(df) < 10:
        return None, "No Data"

    df = df.copy()

    df["returns"] = df["price"].pct_change()
    momentum = df["returns"].rolling(10).mean().iloc[-1] * 1000
    volatility = df["returns"].rolling(10).std().iloc[-1] * 1000

    if pd.isna(momentum) or pd.isna(volatility) or volatility == 0:
        return None, "Insufficient Data"

    score = max(0, min(100, (momentum / volatility) * 50 + 50))

    if score >= 70:
        label = "Strong Uptrend"
    elif score >= 55:
        label = "Mild Uptrend"
    elif score > 45:
        label = "Sideways"
    elif score > 30:
        label = "Mild Downtrend"
    else:
        label = "Strong Downtrend"

    return round(score, 2), label

# ---------------------------------------------------------------
# Cache Handler
# ---------------------------------------------------------------
def load_cached(date_obj):
    path = f"{CACHE_DIR}/{date_obj}.csv"
    return pd.read_csv(path) if os.path.exists(path) else None

def save_cache(date_obj, df):
    df.to_csv(f"{CACHE_DIR}/{date_obj}.csv", index=False)

# ---------------------------------------------------------------
# Load all Yahoo data once (60 days)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_yahoo_all():
    df = fetch_yahoo_intraday_5m(days=60)
    return df

all_df = load_yahoo_all()
all_df_by_date = {d: g.reset_index(drop=True) for d, g in all_df.groupby("date")}

# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.title("NIFTY Intraday Charts – Yahoo Finance 5m (60 Days)")

available_dates = sorted(all_df_by_date.keys())

selected_date = st.date_input("Select Date", value=available_dates[-1], min_value=available_dates[0], max_value=available_dates[-1])

# ---------------------------------------------------------------
# Get Data for Selected Date
# ---------------------------------------------------------------
if selected_date in all_df_by_date:
    day_df = all_df_by_date[selected_date]
else:
    day_df = None
    st.error("No data available for this date.")

# ---------------------------------------------------------------
# Show Trend Strength
# ---------------------------------------------------------------
score, label = calculate_trend_strength(day_df)

col1, col2 = st.columns(2)
col1.metric("Trend Score", score if score else "-")
col2.metric("Trend Label", label)

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
if day_df is not None and len(day_df) > 0:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(day_df["Datetime"], day_df["price"])
    ax.set_title(f"NIFTY Price on {selected_date}")
    ax.set_xlabel("Time (IST)")
    ax.set_ylabel("Price")
    st.pyplot(fig)
else:
    st.warning("No intraday data available.")
