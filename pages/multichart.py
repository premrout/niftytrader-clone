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
# UI – MULTI-TABS + 40-DAY GRID + AUTO-ENLARGE MODAL
# ---------------------------------------------------------------
st.title("Index Intraday Charts – Yahoo Finance 5m (60 Days)")

import plotly.graph_objects as go
import numpy as np

# ---------------------------------------------------------------
# Compute indicators
# ---------------------------------------------------------------
def add_indicators(df):
    df = df.copy()

    # RSI
    delta = df["price"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()
    rs = roll_up / roll_down
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    df["EMA12"] = df["price"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["price"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # VWAP
    df["cum_vol"] = df["Volume"].cumsum()
    df["cum_vp"] = (df["Volume"] * df["price"]).cumsum()
    df["VWAP"] = df["cum_vp"] / df["cum_vol"]

    # Supertrend (basic lightweight version)
    atr = df["price"].rolling(10).std() * 3
    df["ST"] = df["price"] - atr

    return df

# Apply indicators
all_df = add_indicators(all_df)
all_df_by_date = {d: g.reset_index(drop=True) for d, g in all_df.groupby("date")}

# Sort latest → old
available_dates = sorted(all_df_by_date.keys(), reverse=True)

# ---------------------------------------------------------------
# Modal Enlarge (session state)
# ---------------------------------------------------------------
if "modal_day" not in st.session_state:
    st.session_state.modal_day = None

# ---------------------------------------------------------------
# Tabs for Index Selection
# ---------------------------------------------------------------
tabs = st.tabs(["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX"])

for tab, symbol in zip(tabs, ["^NSEI", "^NSEBANK", "^NSEFIN", "^BSESN"]):
    with tab:
        st.subheader(f"{symbol} – Last 40 Days Intraday Grid")

        last_40 = available_dates[:40]
        rows = [last_40[i:i+4] for i in range(0, len(last_40), 4)]

        for row in rows:
            cols = st.columns(4)
            for idx, d in enumerate(row):
                df_day = all_df_by_date[d]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_day["Datetime"], y=df_day["price"], mode="lines", name="Price"))
                fig.add_trace(go.Scatter(x=df_day["Datetime"], y=df_day["VWAP"], mode="lines", name="VWAP"))
                fig.update_layout(height=220, margin=dict(l=5,r=5,t=25,b=5))

                event = cols[idx].plotly_chart(fig, use_container_width=True, key=f"{symbol}_{d}")

                # Auto enlarge when clicked
                if st.session_state.get(f"clicked_{symbol}_{d}"):
                    st.session_state.modal_day = (symbol, d)

# ---------------------------------------------------------------
# Modal Popup
# ---------------------------------------------------------------
if st.session_state.modal_day:
    symbol, d = st.session_state.modal_day
    df_big = all_df_by_date[d]

    st.markdown("### 🔍 Enlarged Chart – " + str(d))

    fig_big = go.Figure()
    fig_big.add_trace(go.Scatter(x=df_big["Datetime"], y=df_big["price"], mode="lines", name="Price"))
    fig_big.add_trace(go.Scatter(x=df_big["Datetime"], y=df_big["VWAP"], mode="lines", name="VWAP"))
    fig_big.add_trace(go.Scatter(x=df_big["Datetime"], y=df_big["ST"], mode="lines", name="Supertrend"))
    fig_big.update_layout(height=600, title=f"{symbol} – {d}")

    st.plotly_chart(fig_big, use_container_width=True)

    if st.button("Close"):
        st.session_state.modal_day = None
