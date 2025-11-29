import streamlit as st
from utils.tv_optionchain import get_tv_option_chain, parse_option_chain

st.title("Live Option Chain (TradingView Source)")

symbol = st.selectbox("Select Index", ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"])

tv_data, err = get_tv_option_chain(symbol)

if err:
    st.error("Data temporarily unavailable. Try again.")
    st.stop()

calls, puts = parse_option_chain(tv_data)

col1, col2 = st.columns(2)

with col1:
    st.subheader("CALLS (CE)")
    st.dataframe(calls)

with col2:
    st.subheader("PUTS (PE)")
    st.dataframe(puts)
