import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Option Chain (NSE API)", layout="wide")

# ------------------------ Fetch NSE Option Chain ------------------------
@st.cache_data(ttl=10)
def fetch_nse_chain(symbol="NIFTY"):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    s = requests.Session()
    s.get("https://www.nseindia.com", headers=headers)
    r = s.get(url, headers=headers)
    return r.json()

# ------------------------ Parse Data ------------------------
def parse_chain(data):
    """Extracts nearest expiry CE/PE and returns DataFrame."""
    records = data.get("records", {})
    exp_list = records.get("expiryDates", [])
    if not exp_list:
        return pd.DataFrame()

    nearest_expiry = exp_list[0]  # Auto-select nearest expiry
    
    rows = []
    for item in records.get("data", []):
        if item.get("expiryDate") != nearest_expiry:
            continue
        
        strike = item.get("strikePrice")
        ce = item.get("CE", {})
        pe = item.get("PE", {})

        rows.append({
            "Strike": strike,
            "CE_LTP": ce.get("lastPrice", 0),
            "PE_LTP": pe.get("lastPrice", 0),
            "CE_OI": ce.get("openInterest", 0),
            "PE_OI": pe.get("openInterest", 0),
            "CE_CHG_OI": ce.get("changeinOpenInterest", 0),
            "PE_CHG_OI": pe.get("changeinOpenInterest", 0)
        })

    df = pd.DataFrame(rows)
    return df.dropna().sort_values("Strike").reset_index(drop=True), nearest_expiry

# ------------------------ UI ------------------------
index = st.selectbox("Select Index", ["NIFTY", "BANKNIFTY", "FINNIFTY"], index=0)
st.title(f"Option Chain — {index}")

raw = fetch_nse_chain(index)
df, expiry = parse_chain(raw)

if df.empty:
    st.error("No data received from NSE. Try again.")
    st.stop()

st.write(f"### Nearest Expiry: **{expiry}**")

# ------------------------ ATM logic ------------------------
atm_row = df.iloc[(df["CE_LTP"] + df["PE_LTP"]).abs().idxmin()]
atm = atm_row["Strike"]

step = int(df["Strike"].diff().median())
left = atm - 5 * step
right = atm + 5 * step

df_f = df[(df["Strike"] >= left) & (df["Strike"] <= right)].copy()

# ------------------------ Display Option Table ------------------------
st.subheader("Nearest 10 Strikes (5 CE + 5 PE around ATM)")
st.dataframe(df_f, use_container_width=True)

# ------------------------ OI Change Chart ------------------------
def plot_oi_change(df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Strike'], y=df['CE_CHG_OI'], name='CE OI Change'))
    fig.add_trace(go.Bar(x=df['Strike'], y=df['PE_CHG_OI'], name='PE OI Change'))
    fig.update_layout(
        barmode='group',
        title="Open Interest Change for the Day",
        xaxis_title="Strike Price",
        yaxis_title="OI Change",
        height=350
    )
    return fig

st.subheader("CE & PE OI Change (Today)")
st.plotly_chart(plot_oi_change(df_f), use_container_width=True)

# ------------------------ OI Bar Graph ------------------------
def plot_oi(df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Strike"], y=df["CE_OI"], name="CE OI"))
    fig.add_trace(go.Bar(x=df["Strike"], y=df["PE_OI"], name="PE OI"))
    fig.update_layout(
        barmode="group",
        title="Open Interest (CE / PE)",
        height=350
    )
    return fig

st.subheader("Open Interest (CE & PE)")
st.plotly_chart(plot_oi(df_f), use_container_width=True)

# ------------------------ Download CSV ------------------------
csv = df_f.to_csv(index=False)
st.download_button("Download CSV", data=csv, file_name=f"{index}_{expiry}.csv")

st.caption(f"Data Source: NSE | Last Updated: {datetime.now().strftime('%H:%M:%S')}")

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
    headers = {"User-Agent": "Mozilla/5.0"}
    s = requests.Session()
    r = s.get("https://www.nseindia.com", headers=headers)
    r = s.get(url, headers=headers)
    data = r.json()
    return data

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

