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

# ------------------ OI CHANGE GRAPH ------------------
def plot_oi_change(df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Strike'], y=df['CE_CHG_OI'], name='CE OI Change'))
    fig.add_trace(go.Bar(x=df['Strike'], y=df['PE_CHG_OI'], name='PE OI Change'))
    fig.update_layout(barmode='group', title='OI Change for the Day')
    return fig

st.subheader("CE & PE OI Change — Today")
st.plotly_chart(plot_oi_change(df_f), use_container_width=True)
