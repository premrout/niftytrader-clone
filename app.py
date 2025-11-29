# app.py
import streamlit as st
import pandas as pd
import time
from datetime import datetime
from utils import nse_api
from utils.charts import oi_grouped_chart  # reuse earlier charts util (if not present, create utils/charts.py)

st.set_page_config(page_title="NiftyTrader Clone", layout="wide")

# ---------- Dark theme CSS (compact) ----------
st.markdown("""
    <style>
    .stApp { background-color: #0f1720; color: #e6eef6; }
    .card { background: #0b1220; border-radius: 8px; padding: 12px; margin-bottom: 10px; border:1px solid rgba(255,255,255,0.03); }
    .kpi { font-size:20px; font-weight:700; }
    .muted { color: #9aa6b2; }
    </style>
    """, unsafe_allow_html=True)

# ---------- Header ----------
col1, col2 = st.columns([3,1])
with col1:
    st.markdown("<h1 style='margin:0'>NiftyTrader Clone</h1>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Realtime analytics powered by NSE public endpoints</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'><b>Last update</b><br><span class='muted'>{}</span></div>".format(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")), unsafe_allow_html=True)

st.write("")

# ---------- Live KPIs (NIFTY & BANKNIFTY) ----------
k1, k2, k3, k4 = st.columns(4)
try:
    nifty_quote = nse_api.get_quote("NIFTY 50")
except Exception:
    nifty_quote = {}
try:
    bank_quote = nse_api.get_quote("BANKNIFTY")
except Exception:
    bank_quote = {}

def fmt(val):
    try:
        return f"{float(val):,.2f}"
    except Exception:
        return "-"

with k1:
    st.markdown("<div class='card'><div class='kpi'>NIFTY</div><div class='muted'>LTP</div></div>", unsafe_allow_html=True)
    st.metric(label="NIFTY LTP", value=fmt(nifty_quote.get("lastPrice")), delta=f"{nifty_quote.get('pChange','-')}%")
with k2:
    st.markdown("<div class='card'><div class='kpi'>BANKNIFTY</div><div class='muted'>LTP</div></div>", unsafe_allow_html=True)
    st.metric(label="BANKNIFTY LTP", value=fmt(bank_quote.get("lastPrice")), delta=f"{bank_quote.get('pChange','-')}%")
with k3:
    st.markdown("<div class='card'><div class='kpi'>PCR (est)</div><div class='muted'>Put/Call Ratio</div></div>", unsafe_allow_html=True)
    st.metric(label="PCR", value="—", delta=None)
with k4:
    st.markdown("<div class='card'><div class='kpi'>Volume (est)</div><div class='muted'>—</div></div>", unsafe_allow_html=True)
    st.metric(label="Volume", value="—", delta=None)

st.write("")

# ---------- Top movers ----------
st.markdown("<div class='card'><h3 style='margin:0'>Top Movers</h3></div>", unsafe_allow_html=True)
movers = nse_api.get_top_gainers_losers()
gainers = movers.get("gainers", pd.DataFrame())
losers = movers.get("losers", pd.DataFrame())

c1, c2 = st.columns(2)
with c1:
    st.subheader("Gainers")
    if not gainers.empty:
        st.dataframe(gainers[["symbol","ltp","pChange"]].rename(columns={"symbol":"Symbol","ltp":"LTP","pChange":"Change%"}).head(10))
    else:
        st.info("No gainers data right now.")
with c2:
    st.subheader("Losers")
    if not losers.empty:
        st.dataframe(losers[["symbol","ltp","pChange"]].rename(columns={"symbol":"Symbol","ltp":"LTP","pChange":"Change%"}).head(10))
    else:
        st.info("No losers data right now.")

st.write("")

# ---------- Option Chain teaser ----------
st.markdown("<div class='card'><h3 style='margin:0'>Option Chain — Quick View</h3></div>", unsafe_allow_html=True)
sym = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY"], index=0)
if st.button("Load Option Chain"):
    with st.spinner("Fetching option chain..."):
        try:
            df_oc = nse_api.get_option_chain_index(sym)
            st.dataframe(df_oc.head(40))
            # plot OI grouped (requires utils/charts.py)
            try:
                fig = oi_grouped_chart(df_oc)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass
        except Exception as e:
            st.error("Failed to load option chain: " + str(e))

st.write("")
st.markdown("<div class='muted'>Note: This page uses NSE public endpoints. For production/redistribution you should use licensed data or paid feeds.</div>", unsafe_allow_html=True)
