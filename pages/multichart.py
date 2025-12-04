import streamlit as st
import pandas as pd
import yfinance as yf
import time
from datetime import datetime, timedelta, date
import pytz
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="NIFTY Intraday (Yahoo 5m – 60 Days)", layout="wide")

CACHE_DIR = ".yahoo_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------
# Fetch 5-minute intraday using yfinance (reliable, no 429)
# ---------------------------------------------------------------
def fetch_yahoo_intraday_5m(days=60):
    ticker = yf.Ticker("^NSEI")

    df = ticker.history(
        period=f"{days}d",
        interval="5m",
    )

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    df.rename(columns={"Close": "price"}, inplace=True)

    # Ensure timezone conversion
    if df["Datetime"].dt.tz is None:
        df["Datetime"] = df["Datetime"].dt.tz_localize("UTC")

    df["Datetime"] = df["Datetime"].dt.tz_convert("Asia/Kolkata")
    df["date"] = df["Datetime"].dt.date

    return df

# ---------------------------------------------------------------
# Trend Score
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

if not all_df_by_date:
    st.error("No data available from Yahoo Finance.")
else:
    available_dates = sorted(all_df_by_date.keys())

    selected_date = st.date_input(
        "Select Date",
        value=available_dates[-1],
        min_value=available_dates[0],
        max_value=available_dates[-1],
    )

    # ---------------------------------------------------------------
    # Get Data for Selected Date
    # ---------------------------------------------------------------
    day_df = all_df_by_date.get(selected_date, None)

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
