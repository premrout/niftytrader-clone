# Streamlit Option Chain without Playwright
# Shows 10 CE & PE around ATM (5 each side)
# Adds OI change graph

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Option Chain Lite", layout="wide")

# ------------------ NSE API FETCH ------------------
@st.cache_data(ttl=8)
def fetch_nse_chain(symbol="NIFTY"):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain",
        "Connection": "keep-alive",
    }
    s = requests.Session()
    # First request to get cookies
    s.get("https://www.nseindia.com", headers=headers)
    r = s.get(url, headers=headers)
    if r.status_code != 200:
        st.error("NSE blocked request. Refresh or try again.")
        return {"records": {"data": []}}
    return r.json()

# ------------------ PARSE CHAIN ------------------
def parse_chain(data):
    rows = []
    for item in data.get("records", {}).get("data", []):
        strike = item.get("strikePrice")
        ce = item.get("CE", {})
        pe = item.get("PE", {})
        rows.append({
            "Strike": strike,
            "CE_OI": ce.get("openInterest", 0),
            "PE_OI": pe.get("openInterest", 0),
            "CE_LTP": ce.get("lastPrice", 0),
            "PE_LTP": pe.get("lastPrice", 0),
            "CE_CHG_OI": ce.get("changeinOpenInterest", 0),
            "PE_CHG_OI": pe.get("changeinOpenInterest", 0)
        })
    df = pd.DataFrame(rows)
    return df.dropna().sort_values("Strike").reset_index(drop=True)

# ------------------ UI ------------------
index = st.selectbox("Index", ["NIFTY", "BANKNIFTY", "FINNIFTY"], index=0)
st.header(f"Option Chain — {index}")

raw = fetch_nse_chain(index)
df = parse_chain(raw)

# ATM detection
atm = df.iloc[(df['CE_LTP'] + df['PE_LTP']).abs().idxmin()]['Strike']

# Filter 5 strikes each side → 10 total per CE/PE
step = int(df['Strike'].diff().median())
left = atm - 5 * step
right = atm + 5 * step
df_f = df[(df['Strike'] >= left) & (df['Strike'] <= right)].copy()

st.subheader("Nearby Option Strikes (±5)")
st.dataframe(df_f)

# ------------------ PCR + Trend Calculation ------------------
# PCR = Sum(PE_OI) / Sum(CE_OI)
def compute_pcr(df):
    total_ce = df['CE_OI'].sum()
    total_pe = df['PE_OI'].sum()
    return (total_pe / total_ce) if total_ce else 0

pcr = compute_pcr(df_f)
pcr_change = (df_f['PE_CHG_OI'].sum() - df_f['CE_CHG_OI'].sum()) / max(df_f['CE_OI'].sum(), 1)

st.subheader("PCR & Trend Suggestion")
st.write(f"PCR (ATM ±5): **{pcr:.2f}**")
st.write(f"PCR Change Today: **{pcr_change:.3f}**")

if pcr > 1.2 and pcr_change > 0:
    trend = "Bullish — PE OI rising faster than CE OI"
elif pcr < 0.8 and pcr_change < 0:
    trend = "Bearish — CE OI rising faster than PE OI"
else:
    trend = "Sideways / Neutral"

st.info(f"Trend: **{trend}**")

# ------------------ PCR Line Graph ------------------
def plot_pcr_line(df):
    fig = go.Figure()
    pe = df['PE_OI']
    ce = df['CE_OI']
    pcr_vals = pe / ce.replace(0, np.nan)
    fig.add_trace(go.Scatter(x=df['Strike'], y=pcr_vals, mode='lines+markers', name='PCR by Strike'))
    fig.update_layout(title='PCR Across Strikes (ATM ±5)', yaxis_title='PCR')
    return fig

st.subheader("PCR Line Chart — Strike-wise")
st.plotly_chart(plot_pcr_line(df_f), use_container_width=True)

# ------------------ OI CHANGE GRAPH ------------------ ------------------
def plot_oi_change(df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Strike'], y=df['CE_CHG_OI'], name='CE OI Change'))
    fig.add_trace(go.Bar(x=df['Strike'], y=df['PE_CHG_OI'], name='PE OI Change'))
    fig.update_layout(barmode='group', title='OI Change for the Day')
    return fig

st.subheader("CE & PE OI Change — Today")
st.plotly_chart(plot_oi_change(df_f), use_container_width=True)

# ------------------ PCR + Trend Calculation ------------------
# PCR = Sum(PE_OI) / Sum(CE_OI)
def compute_pcr(df):
    total_ce = df['CE_OI'].sum()
    total_pe = df['PE_OI'].sum()
    return (total_pe / total_ce) if total_ce else 0

pcr = compute_pcr(df_f)
pcr_change = (df_f['PE_CHG_OI'].sum() - df_f['CE_CHG_OI'].sum()) / max(df_f['CE_OI'].sum(), 1)

st.subheader("PCR & Trend Suggestion (ATM ±5)")
st.write(f"PCR: **{pcr:.2f}**")
st.write(f"PCR Change Today: **{pcr_change:.3f}**")

if pcr > 1.2 and pcr_change > 0:
    trend = "Bullish — PE OI rising faster than CE OI"
elif pcr < 0.8 and pcr_change < 0:
    trend = "Bearish — CE OI rising faster than PE OI"
else:
    trend = "Sideways / Neutral"

st.info(f"Market Trend: **{trend}**")

# ------------------ PCR Line Chart ------------------
def plot_pcr_line(df):
    fig = go.Figure()
    pe = df['PE_OI']
    ce = df['CE_OI'].replace(0, np.nan)
    pcr_vals = pe / ce
    fig.add_trace(go.Scatter(x=df['Strike'], y=pcr_vals, mode='lines+markers', name='PCR by Strike'))
    fig.update_layout(title='PCR Across Strikes (ATM ±5)', yaxis_title='PCR')
    return fig

st.subheader("Strike-wise PCR Line Chart")
st.plotly_chart(plot_pcr_line(df_f), use_container_width=True)

# ------------------ RSI Indicator ------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# use CE_LTP as price proxy
rsi = compute_rsi(df_f['CE_LTP'])

st.subheader("RSI (Using CE LTP as Proxy)")
st.line_chart(rsi)

if rsi.iloc[-1] > 70:
    rsi_trend = "Overbought — Possible Downtrend"
elif rsi.iloc[-1] < 30:
    rsi_trend = "Oversold — Possible Uptrend"
else:
    rsi_trend = "Neutral Momentum"

st.success(f"RSI Momentum: **{rsi_trend}**")
