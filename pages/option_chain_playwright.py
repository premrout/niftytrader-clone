# pages/option_chain_no_playwright.py
"""
Streamlit Option Chain (NSE API) — fixed version
- Auto-refresh works (st_autorefresh)
- No StreamlitDuplicateElementId errors
- All charts have unique keys
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Option Chain (NSE API)", layout="wide")

# ------------------ Auto Refresh ------------------
st_autorefresh = st.experimental_rerun if False else None
st_autorefresh = getattr(st, "autorefresh", None)
if st_autorefresh:
    st.autorefresh(interval=15000, key="autorefresh_15s")   # 15 seconds refresh


# ---------- Utilities ----------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com"
}

def safe_session_get(session: requests.Session, url: str, headers=None, timeout=8, retries=3):
    headers = headers or HEADERS
    for i in range(retries):
        try:
            r = session.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Failed to GET {url} after {retries} attempts")


@st.cache_data(ttl=8, show_spinner=False)
def fetch_nse_chain(symbol: str = "NIFTY"):
    base = "https://www.nseindia.com"
    url = f"{base}/api/option-chain-indices?symbol={symbol}"
    s = requests.Session()
    try:
        s.get(base, headers=HEADERS, timeout=5)
    except Exception:
        pass
    r = safe_session_get(s, url, headers=HEADERS, timeout=10, retries=4)
    return r.json()


def parse_chain(json_data):
    rows = []
    records = json_data.get("records", {})
    data_list = records.get("data", [])
    expiry_dates = records.get("expiryDates", [])
    nearest_expiry = expiry_dates[0] if expiry_dates else None

    for item in data_list:
        if nearest_expiry and item.get("expiryDate") != nearest_expiry:
            continue

        strike = item.get("strikePrice")
        ce = item.get("CE", {}) or {}
        pe = item.get("PE", {}) or {}

        rows.append({
            "Strike": strike,
            "CE_LTP": ce.get("lastPrice", 0.0),
            "PE_LTP": pe.get("lastPrice", 0.0),
            "CE_OI": ce.get("openInterest", 0),
            "PE_OI": pe.get("openInterest", 0),
            "CE_CHG_OI": ce.get("changeinOpenInterest", 0),
            "PE_CHG_OI": pe.get("changeinOpenInterest", 0),
            "expiryDate": item.get("expiryDate")
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Strike"] = pd.to_numeric(df["Strike"], errors="coerce")
    for col in ["CE_LTP", "PE_LTP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in ["CE_OI", "PE_OI", "CE_CHG_OI", "PE_CHG_OI"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df = df.dropna(subset=["Strike"]).sort_values("Strike").reset_index(drop=True)
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


# ------------------ UI ------------------
st.title("📊 Option Chain — NSE (No Playwright)")
st.caption("Nearest expiry, ±N strikes around ATM. Auto-refresh every 15s")

with st.sidebar:
    st.markdown("## Controls")
    index_choice = st.selectbox("Index", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
    strikes_each_side = st.slider("Strikes each side of ATM", 3, 12, 5)
    if st.button("Force refresh"):
        fetch_nse_chain.clear()


# ------------------ Fetch chain ------------------
st.info("Fetching option-chain from NSE...", icon="🔁")

try:
    raw = fetch_nse_chain(index_choice)
except Exception as e:
    st.error(f"Failed to fetch from NSE: {e}")
    st.stop()

df_all = parse_chain(raw)
if df_all.empty:
    st.error("No option chain data parsed.")
    st.stop()

expiry_dates = raw.get("records", {}).get("expiryDates", [])
nearest_expiry = expiry_dates[0] if expiry_dates else None
st.markdown(f"**Expiry used:** {nearest_expiry}")

spot_val = raw.get("records", {}).get("underlyingValue")
if spot_val:
    st.metric(label=f"{index_choice} Spot", value=f"{spot_val:,.2f}")


# ------------------ ATM ------------------
if spot_val:
    strikes = df_all["Strike"]
    atm = int(min(strikes, key=lambda s: abs(s - spot_val)))
else:
    df_all["mid_sum"] = df_all["CE_LTP"] + df_all["PE_LTP"]
    atm = int(df_all.loc[df_all["mid_sum"].idxmin(), "Strike"])

step = int(df_all["Strike"].diff().median())
left_bound = atm - strikes_each_side * step
right_bound = atm + strikes_each_side * step

df_window = df_all[(df_all["Strike"] >= left_bound) & (df_all["Strike"] <= right_bound)].reset_index(drop=True)


# ------------------ Table ------------------
st.subheader(f"Nearby Option Strikes around ATM {atm}")

def highlight_atm(row):
    return ["background-color: #1f2a30; color: yellow" if row["Strike"] == atm else "" for _ in row]

styled = df_window.style.apply(highlight_atm, axis=1)
st.dataframe(styled, use_container_width=True)


# ------------------ OI Change Chart ------------------
st.subheader("CE & PE ΔOI — Today")

fig_oi = go.Figure()
fig_oi.add_trace(go.Bar(x=df_window["Strike"], y=df_window["CE_CHG_OI"], name="CE ΔOI"))
fig_oi.add_trace(go.Bar(x=df_window["Strike"], y=df_window["PE_CHG_OI"], name="PE ΔOI"))
fig_oi.update_layout(barmode="group", template="plotly_dark", height=360)

st.plotly_chart(fig_oi, use_container_width=True, key="oi_chart")


# ------------------ PCR ------------------
st.subheader("PCR (Window)")

pcr_total_ce = df_window["CE_OI"].sum()
pcr_total_pe = df_window["PE_OI"].sum()
pcr = pcr_total_pe / pcr_total_ce if pcr_total_ce else 0

st.markdown(f"- **PCR:** {pcr:.3f}")

pcr_line = df_window["PE_OI"] / df_window["CE_OI"].replace(0, np.nan)
fig_pcr = go.Figure()
fig_pcr.add_trace(go.Scatter(x=df_window["Strike"], y=pcr_line, mode="lines+markers"))
fig_pcr.update_layout(template="plotly_dark", height=260)

st.plotly_chart(fig_pcr, use_container_width=True, key="pcr_chart")


# ------------------ RSI ------------------
df_window["SYM_PRICE"] = (df_window["CE_LTP"] + df_window["PE_LTP"]) / 2
df_window["RSI"] = compute_rsi(df_window["SYM_PRICE"])

fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df_window["Strike"], y=df_window["RSI"], mode="lines+markers"))
fig_rsi.update_layout(template="plotly_dark", height=260)

st.subheader("RSI (Synthetic)")
st.plotly_chart(fig_rsi, use_container_width=True, key="rsi_chart")


# ------------------ Download ------------------
csv = df_window.to_csv(index=False)
st.download_button("Download CSV", csv, "option_chain.csv")
