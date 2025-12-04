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

    st.subheader("Last 40 Days – Multi Chart View")

    import plotly.graph_objects as go

    # Track clicked chart
    if "selected_day" not in st.session_state:
        st.session_state.selected_day = None

    last_40_days = available_dates[-40:]

    rows = [last_40_days[i:i+4] for i in range(0, len(last_40_days), 4)]

    for row in rows:
        cols = st.columns(4)
        for idx, d in enumerate(row):
            df_day = all_df_by_date[d]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_day["Datetime"], y=df_day["price"], mode="lines"))
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=20, b=10))
            if cols[idx].plotly_chart(fig, use_container_width=True, key=f"chart_{d}"):
                pass
            if cols[idx].button(f"Enlarge {d}"):
                st.session_state.selected_day = d

    if st.session_state.selected_day:
        st.subheader(f"Enlarged View – {st.session_state.selected_day}")
        df_big = all_df_by_date[st.session_state.selected_day]
        big = go.Figure()
        big.add_trace(go.Scatter(x=df_big["Datetime"], y=df_big["price"], mode="lines"))
        big.update_layout(height=600, title=f"NIFTY – {st.session_state.selected_day}")
        st.plotly_chart(big, use_container_width=True)
