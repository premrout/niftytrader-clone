import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import time
import plotly.graph_objects as go

st.set_page_config(page_title="Option Chain (NSE API)", layout="wide")

# ---------------- NSE API DIRECT SCRAPER ----------------
@st.cache_data(ttl=5)
def fetch_nse_option_chain(symbol: str):
    """
    Fetch option chain from NSE India WITHOUT Playwright.
    Uses official API endpoint.
    """
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "accept": "application/json",
        "accept-language": "en-US,en;q=0.9",
    }

    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers)
        response = session.get(url, headers=headers, timeout=10)
        data = response.json()
    except Exception as e:
        return None, f"Failed fetching NSE API data: {e}"

    try:
        rows = data["records"]["data"]
        oc_list = []
        for row in rows:
            strike = row.get("strikePrice")

            ce = row.get("CE", {})
            pe = row.get("PE", {})

            oc_list.append({
                "Strike": strike,
                "CE_LTP": ce.get("lastPrice", 0),
                "CE_OI": ce.get("openInterest", 0),
                "PE_LTP": pe.get("lastPrice", 0),
                "PE_OI": pe.get("openInterest", 0),
            })

        df = pd.DataFrame(oc_list)
        df = df.sort_values("Strike").reset_index(drop=True)
        return df, None

    except Exception as e:
        return None, f"Parsing error: {e}"

# --------------- Metrics ----------------
def compute_pcr(df):
    ce = df["CE_OI"].sum()
    pe = df["PE_OI"].sum()
    return round(pe / ce, 2) if ce > 0 else np.nan


def compute_max_pain(df):
    strikes = df["Strike"].unique()
    pain = []

    for s in strikes:
        total = 0
        for _, r in df.iterrows():
            k = r["Strike"]
            ce_oi = r["CE_OI"]
            pe_oi = r["PE_OI"]

            ce_pain = max(s - k, 0) * ce_oi
            pe_pain = max(k - s, 0) * pe_oi
            total += ce_pain + pe_pain
        pain.append((s, total))

    pain_df = pd.DataFrame(pain, columns=["Strike", "Pain"])
    return int(pain_df.sort_values("Pain").iloc[0]["Strike"])

# --------------- Plots ----------------
def plot_oi_grouped(df):
    fig = go.Figure()
    fig.add_bar(x=df["Strike"], y=df["CE_OI"], name="CE OI")
    fig.add_bar(x=df["Strike"], y=df["PE_OI"], name="PE OI")
    fig.update_layout(barmode="group", height=350, template="plotly_dark")
    return fig


def plot_oi_stacked(df):
    fig = go.Figure()
    fig.add_bar(x=df["Strike"], y=df["CE_OI"], name="CE OI")
    fig.add_bar(x=df["Strike"], y=df["PE_OI"], name="PE OI")
    fig.update_layout(barmode="stack", height=350, template="plotly_dark")
    return fig


def plot_net_heatmap(df):
    df2 = df.copy()
    df2["NET"] = df2["PE_OI"] - df2["CE_OI"]
    fig = go.Figure(
        data=go.Heatmap(
            z=[df2["NET"].tolist()],
            x=df2["Strike"].tolist(),
            y=["Net OI"],
            colorscale="RdBu",
            zmid=0,
        )
    )
    fig.update_layout(height=150, template="plotly_dark")
    return fig


# ---------------- UI ----------------
st.title("📉 NSE Option Chain — Fast API (No Playwright)")

with st.sidebar:
    st.header("Controls")

    index_choice = st.selectbox(
        "Index",
        ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"],
        index=0
    )

    refresh_sec = st.selectbox("Auto Refresh (sec)", [3, 5, 8, 10, 15], index=1)
    strikes_each = st.slider("Strikes each side of ATM", 3, 15, 6)

    if st.button("Force Refresh"):
        fetch_nse_option_chain.clear()
        st.rerun()

df, error = fetch_nse_option_chain(index_choice)

if error:
    st.error(error)
    st.stop()

# ATM detection
df["Sum"] = df["CE_LTP"] + df["PE_LTP"]
atm = int(df.loc[df["Sum"].idxmin()]["Strike"])

# Filter strikes around ATM
step = int(df["Strike"].diff().median())
left = atm - strikes_each * step
right = atm + strikes_each * step
fdf = df[(df["Strike"] >= left) & (df["Strike"] <= right)].copy()

# Metrics
pcr = compute_pcr(fdf)
max_pain = compute_max_pain(fdf)

c1, c2, c3 = st.columns(3)
c1.metric("ATM Strike", atm)
c2.metric("PCR", pcr)
c3.metric("Max Pain", max_pain)

st.markdown("---")

# Plots
col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(plot_oi_grouped(fdf), use_container_width=True)
    st.plotly_chart(plot_oi_stacked(fdf), use_container_width=True)

with col2:
    st.plotly_chart(plot_net_heatmap(fdf), use_container_width=True)

# Table
st.subheader("Option Chain Table")
st.dataframe(
    fdf[["Strike", "CE_LTP", "CE_OI", "PE_LTP", "PE_OI"]],
    use_container_width=True,
    hide_index=True,
)

# CSV Download
st.download_button(
    "Download CSV",
    fdf.to_csv(index=False),
    file_name=f"{index_choice}_option_chain.csv",
    mime="text/csv",
)

st.caption(
    f"Source: NSE India | Last Updated: {datetime.now().strftime('%H:%M:%S')} | Auto-refresh: {refresh_sec}s"
)

# Auto-refresh
time.sleep(refresh_sec)
st.rerun()

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

