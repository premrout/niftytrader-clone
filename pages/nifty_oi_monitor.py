import streamlit as st
import requests
import pandas as pd
import time

import streamlit as st

def top_nav():
    st.markdown("""
    <style>
        .nav-container {
            background-color: #161A1D;
            padding: 10px;
            border-bottom: 1px solid #30363D;
            position: sticky;
            top: 0;
            z-index: 999;
        }
        .nav-item {
            color: white;
            margin-right: 20px;
            font-size: 18px;
            font-weight: 600;
            text-decoration: none;
        }
        .nav-item:hover { color: #00E3FF; }
    </style>

    <div class="nav-container">
        <a class="nav-item" href="/?page=home">Home</a>
        <a class="nav-item" href="/?page=option-chain">Option Chain</a>
        <a class="nav-item" href="/?page=oi-monitor">OI Monitor</a>
        <a class="nav-item" href="https://github.com/premrout/niftytrader-clone">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

top_nav()

# ---------- CONFIG ----------
REFRESH_SECONDS = 15

NSE_OC_URL = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
NSE_QUOTE_URL = "https://www.nseindia.com/api/quote-derivative?symbol=NIFTY"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain"
}

# ---------- Helpers ----------
def create_session():
    s = requests.Session()
    try:
        s.get("https://www.nseindia.com", headers=HEADERS, timeout=5)
    except:
        pass
    return s

def safe_json_request(session, url, retries=6, timeout=10):
    for attempt in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=timeout)
            if r.text.strip().startswith("<"):
                session = create_session()
                time.sleep(1)
                continue
            return r.json()
        except:
            session = create_session()
            time.sleep(1)
    raise RuntimeError("NSE JSON not available")

def safe_float(x):
    try:
        return float(x or 0)
    except:
        try:
            return float(str(x).replace(",", ""))
        except:
            return 0.0

def safe_int(x):
    try:
        return int(float(str(x).replace(",", "")))
    except:
        return 0

# ---------- Fetchers ----------
def get_atm_strike_and_spot():
    s = create_session()
    data = safe_json_request(s, NSE_QUOTE_URL)
    spot = safe_float(data.get("underlyingValue"))
    atm = round(spot / 50) * 50
    return atm, spot

def fetch_option_chain_df():
    s = create_session()
    data = safe_json_request(s, NSE_OC_URL)
    records = data.get("records", {})
    expiry_dates = records.get("expiryDates", [])
    nearest_expiry = expiry_dates[0]

    rows = []
    for item in records.get("data", []):
        if item.get("expiryDate") != nearest_expiry:
            continue

        strike = item.get("strikePrice")
        ce = item.get("CE") or {}
        pe = item.get("PE") or {}

        ce_ltp = safe_float(ce.get("lastPrice"))
        pe_ltp = safe_float(pe.get("lastPrice"))
        ce_oi = safe_int(ce.get("openInterest"))
        pe_oi = safe_int(pe.get("openInterest"))
        ce_chg_oi = safe_int(ce.get("changeinOpenInterest"))
        pe_chg_oi = safe_int(pe.get("changeinOpenInterest"))

        ce_value = ce_ltp * ce_oi
        pe_value = pe_ltp * pe_oi

        ce_chg_value = ce_ltp * ce_chg_oi
        pe_chg_value = pe_ltp * pe_chg_oi

        diff_value = ce_chg_value - pe_chg_value
        diff_percent = (diff_value / pe_chg_value * 100) if pe_chg_value else 0

        rows.append([
            strike,
            ce_ltp, pe_ltp,
            ce_oi, pe_oi,
            ce_value, pe_value,
            ce_chg_value, pe_chg_value,
            diff_value, diff_percent
        ])

    df = pd.DataFrame(rows, columns=[
        "Strike", "CE_LTP", "PE_LTP",
        "CE_OI", "PE_OI",
        "CE_VALUE", "PE_VALUE",
        "CE_CHG_VALUE", "PE_CHG_VALUE",
        "DIFF_VALUE", "DIFF_PERCENT"
    ])
    return df.sort_values("Strike")

# ---------- Trend Engine ----------
def trend_suggestion_combined(df):
    total_ce_value = df["CE_VALUE"].sum()
    total_pe_value = df["PE_VALUE"].sum()
    total_ce_chg = df["CE_CHG_VALUE"].sum()
    total_pe_chg = df["PE_CHG_VALUE"].sum()
    total_ce_oi = df["CE_OI"].sum()
    total_pe_oi = df["PE_OI"].sum()

    pcr = total_pe_oi / total_ce_oi if total_ce_oi else 0

    if (total_pe_value > total_ce_value) and (total_pe_chg > total_ce_chg):
        value_trend = "Bullish (PE value & change dominate)"
        value_bias = 1
    elif (total_ce_value > total_pe_value) and (total_ce_chg > total_pe_chg):
        value_trend = "Bearish (CE value & change dominate)"
        value_bias = -1
    else:
        value_trend = "Neutral"
        value_bias = 0

    if pcr > 1.25:
        pcr_trend = "Bullish PCR"
        pcr_bias = 1
    elif pcr < 0.75:
        pcr_trend = "Bearish PCR"
        pcr_bias = -1
    else:
        pcr_trend = "Neutral"
        pcr_bias = 0

    score = value_bias + pcr_bias
    if score >= 2:
        final = "📈 Strong Bullish"
    elif score == 1:
        final = "🔼 Bullish Bias"
    elif score == 0:
        final = "⚖ Neutral"
    elif score == -1:
        final = "🔽 Bearish Bias"
    else:
        final = "📉 Strong Bearish"

    return {
        "pcr": pcr,
        "value_trend": value_trend,
        "pcr_trend": pcr_trend,
        "final": final,
        "ce_val": total_ce_value,
        "pe_val": total_pe_value,
        "ce_chg": total_ce_chg,
        "pe_chg": total_pe_chg
    }

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Nifty OI Monitor", layout="wide")

st.title("📊 NIFTY OI MONITOR (Streamlit Version)")
st.caption("Auto-refreshing every 15 seconds")

placeholder = st.empty()

while True:
    with placeholder.container():

        atm, spot = get_atm_strike_and_spot()
        df = fetch_option_chain_df()

        st.subheader(f"Spot: `{spot}` | ATM: `{atm}`")

        meta = trend_suggestion_combined(df)

        # --- Trend Cards ---
        st.write("### Trend Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("PCR", f"{meta['pcr']:.2f}", None)
        col2.metric("Value Trend", meta["value_trend"])
        col3.metric("Final Trend", meta["final"])

        # --- Table ---
        st.write("### Option Chain (Nearest 10 strikes around ATM)")
        lower = df[df["Strike"] < atm].tail(5)
        mid = df[df["Strike"] == atm]
        upper = df[df["Strike"] > atm].head(5)
        df10 = pd.concat([lower, mid, upper])

        st.dataframe(df10, use_container_width=True)

time.sleep(refresh_interval)
st.rerun()
