# app_option_chain.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import math

# Try importing nsepython (primary). If it fails, we'll handle gracefully.
try:
    from nsepython import nse_optionchain_scrapper
    NSEPY_AVAILABLE = True
except Exception:
    NSEPY_AVAILABLE = False

st.set_page_config(page_title="Option Chain - Nifty", layout="wide")
st.title("🔗 Option Chain Viewer — NIFTY / BANKNIFTY")

st.markdown(
    """
    Fetches option chain (via `nsepython`) and shows:
    - Option chain table
    - CE vs PE Open Interest charts
    - Max Pain calculation
    """
)

# ---- Utilities ----
@st.cache_data(ttl=30)
def fetch_option_chain(symbol: str):
    """
    Returns a DataFrame with columns:
    strike, CE_OI, PE_OI, CE_changeOI, PE_changeOI
    """
    if not NSEPY_AVAILABLE:
        raise RuntimeError("nsepython not available in this environment.")

    raw = nse_optionchain_scrapper(symbol)
    if not raw:
        return pd.DataFrame()

    records = raw.get("records", {}).get("data", [])
    rows = []
    for r in records:
        strike = r.get("strikePrice")
        ce = r.get("CE")
        pe = r.get("PE")

        rows.append({
            "strike": strike,
            "CE_OI": int(ce.get("openInterest")) if ce and ce.get("openInterest") is not None else 0,
            "PE_OI": int(pe.get("openInterest")) if pe and pe.get("openInterest") is not None else 0,
            "CE_changeOI": int(ce.get("changeinOpenInterest")) if ce and ce.get("changeinOpenInterest") is not None else 0,
            "PE_changeOI": int(pe.get("changeinOpenInterest")) if pe and pe.get("changeinOpenInterest") is not None else 0,
            "CE_ltp": float(ce.get("lastPrice")) if ce and ce.get("lastPrice") is not None else None,
            "PE_ltp": float(pe.get("lastPrice")) if pe and pe.get("lastPrice") is not None else None,
        })
    df = pd.DataFrame(rows).dropna(subset=["strike"]).sort_values("strike").reset_index(drop=True)
    return df

def calculate_max_pain(df: pd.DataFrame, spot: float = None):
    """
    Max pain calculation:
    For each strike S, sum of value of all option buyer payoff at expiration (CE intrinsic + PE intrinsic).
    The strike with minimum total pain is max pain.
    """
    strikes = df["strike"].unique()
    strikes = sorted(strikes)
    pain_list = []
    # spot is optional; we only use strikes and OI
    for s in strikes:
        total_pain = 0
        for _, row in df.iterrows():
            k = row["strike"]
            ce_oi = row["CE_OI"]
            pe_oi = row["PE_OI"]
            # buyer payoff magnitude for CE at strike k if expiry price = s: max(0, s-k)
            ce_payoff = max(0, s - k)
            pe_payoff = max(0, k - s)
            total_pain += ce_oi * ce_payoff + pe_oi * pe_payoff
        pain_list.append((s, total_pain))
    pain_df = pd.DataFrame(pain_list, columns=["strike", "total_pain"]).sort_values("total_pain")
    if not pain_df.empty:
        max_pain_strike = int(pain_df.iloc[0]["strike"])
    else:
        max_pain_strike = None
    return max_pain_strike, pain_df

# ---- Controls ----
col1, col2 = st.columns([3, 1])
with col1:
    symbol = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY"], index=0)
with col2:
    refresh = st.button("Refresh")

# ---- Fetch data ----
st.info("Fetching option chain... refresh frequency is cached (30s).")
try:
    df = fetch_option_chain(symbol)
except Exception as e:
    st.error(f"Failed to fetch option chain with nsepython: {e}")
    st.stop()

if df is None or df.empty:
    st.warning("No option-chain data available right now. Try Refresh in a few seconds.")
    st.stop()

# ---- Display basic info ----
st.write(f"Data fetched at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
st.write(f"Strike range: {int(df['strike'].min())} — {int(df['strike'].max())}")

# ---- Small helper: show table and filters ----
with st.expander("Option Chain Table (show/hide)"):
    # Add simple filters
    min_strike = int(df['strike'].min())
    max_strike = int(df['strike'].max())
    s1, s2 = st.slider("Strike range", min_value=min_strike, max_value=max_strike, value=(min_strike, max_strike), step=50)
    filtered = df[(df['strike'] >= s1) & (df['strike'] <= s2)].copy()
    st.dataframe(filtered.style.format({
        "strike": "{:.0f}",
        "CE_OI": "{:,.0f}",
        "PE_OI": "{:,.0f}",
        "CE_changeOI": "{:,.0f}",
        "PE_changeOI": "{:,.0f}",
    }), height=420)

# ---- Charts ----
st.subheader("Open Interest by Strike")

fig = go.Figure()
fig.add_trace(go.Bar(x=df['strike'], y=df['CE_OI'], name='CE OI'))
fig.add_trace(go.Bar(x=df['strike'], y=df['PE_OI'], name='PE OI'))
fig.update_layout(barmode='group', xaxis_title='Strike', yaxis_title='Open Interest', title=f'CE vs PE OI ({symbol})')
st.plotly_chart(fig, use_container_width=True)

st.subheader("CE + PE (stacked)")
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=df['strike'], y=df['CE_OI'], name='CE OI'))
fig2.add_trace(go.Bar(x=df['strike'], y=df['PE_OI'], name='PE OI'))
fig2.update_layout(barmode='stack', xaxis_title='Strike', yaxis_title='Open Interest', title=f'Stacked CE+PE OI ({symbol})')
st.plotly_chart(fig2, use_container_width=True)

# ---- Max pain ----
st.subheader("Max Pain (approximate)")
# Try to get spot from the option chain raw structure if possible (we didn't return it); optional: fetch quote
# For simplicity we compute max pain using strikes and OI
max_pain_strike, pain_df = calculate_max_pain(df)
if max_pain_strike is not None:
    st.metric("Max Pain Strike", f"{max_pain_strike}")
    # show min pain plot
    top_pain = pain_df.sort_values("total_pain").head(10)
    fig3 = go.Figure([go.Bar(x=top_pain['strike'], y=top_pain['total_pain'])])
    fig3.update_layout(title="Top 10 Least-Painful Strikes (lower is more pain-minimizing)", xaxis_title="Strike", yaxis_title="Total Pain")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.write("Max Pain could not be computed.")

st.caption("Note: Max Pain calculation here is a simple approximation (uses OI as buyer exposure). Use carefully.")

# ---- Footer / tips ----
st.write("---")
st.write("Tips:")
st.write(
    """
    - NSE can throttle scrapers: don't refresh too frequently.  
    - If nsepython fails, we can add an alternative data fetcher (e.g., a proxy or paid API).  
    - Want additional features? Ask me to add: Max Pain heatmap, PCR, strike-level Greeks, or a Dark theme.
    """
)
