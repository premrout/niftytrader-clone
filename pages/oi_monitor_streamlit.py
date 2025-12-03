import streamlit as st
import pandas as pd
import requests
from datetime import datetime

st.set_page_config(page_title="NSE Live Data", layout="wide")

st.title("📈 NSE India – Real-Time Market Data")

# NSE requires proper headers, otherwise you get 403/blocked
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9",
}

# API endpoints
NSE_QUOTE_API = "https://www.nseindia.com/api/quote-equity?symbol={symbol}"
NSE_INDICES_API = "https://www.nseindia.com/api/allIndices"

@st.cache_data(ttl=15)
def get_stock_data(symbol):
    url = NSE_QUOTE_API.format(symbol=symbol.upper())
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=HEADERS)

    res = session.get(url, headers=HEADERS)
    res.raise_for_status()
    return res.json()

@st.cache_data(ttl=15)
def get_index_data():
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=HEADERS)

    res = session.get(NSE_INDICES_API, headers=HEADERS)
    res.raise_for_status()
    return res.json()

# ---------------- UI ----------------

tab1, tab2 = st.tabs(["📊 Stock Quote", "📉 Market Indices"])

with tab1:
    symbol = st.text_input("Enter Stock Symbol (e.g., TCS, INFY, RELIANCE)", "RELIANCE")

    if st.button("Fetch Live Stock Data"):
        try:
            data = get_stock_data(symbol)
            info = data.get("priceInfo", {})
            meta = data.get("info", {})

            st.subheader(f"Live Price for {meta.get('companyName', symbol)}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Last Price", info.get("lastPrice"))
            col2.metric("Open", info.get("open"))
            col3.metric("Day High", info.get("intraDayHighLow", {}).get("max"))

            col4, col5, col6 = st.columns(3)
            col4.metric("Day Low", info.get("intraDayHighLow", {}).get("min"))
            col5.metric("Previous Close", info.get("previousClose"))
            col6.metric("Change (%)", info.get("pChange"))

            st.write("Raw Data:")
            st.json(data)

        except Exception as e:
            st.error(f"Error fetching data: {e}")

with tab2:
    st.subheader("📉 NSE Market Indices (Live)")

    try:
        indices = get_index_data()

        idx_list = indices.get("data", [])
        df = pd.DataFrame(idx_list)
        df = df[["index", "last", "variation", "percentChange"]]

        st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading indices: {e}")

st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
