# pages/option_chain.py
import streamlit as st
import pandas as pd
import time
from datetime import datetime
from utils.tv_api import get_tv_optionchain
from utils.oc_parser import parse_tv_rows
from utils.oc_helpers import (
    build_combined_df, compute_pcr, compute_max_pain,
    plot_oi_grouped, plot_oi_stacked, plot_net_heatmap
)

st.set_page_config(page_title="Option Chain — NiftyTrader style", layout="wide")

# --- CSS to approximate NiftyTrader styling ---
st.markdown("""
<style>
body { background-color: #0b1220; color: #dbe9f2; }
.card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius: 10px; padding:10px; margin-bottom:10px; border:1px solid rgba(255,255,255,0.03);}
.table-light th { color: #9aa6b2; font-weight:600; font-size:13px; }
.small-muted { color:#9aa6b2; font-size:12px; }
.kpi { font-size:18px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

st.title("NiftyTrader-style Option Chain (TradingView source)")

# Sidebar controls
with st.sidebar:
    st.markdown("## Controls")
    symbol = st.selectbox("Index", ["NIFTY", "BANKNIFTY", "FINNIFTY"], index=0)
    refresh_sec = st.slider("Auto-refresh (sec)", 5, 60, 12)
    strikes_each_side = st.slider("Strikes each side of ATM", 8, 24, 12)
    if st.button("Force refresh"):
        st.session_state['force_refresh'] = True
        st.rerun()
    st.markdown("---")
    st.markdown("Data source: TradingView scanner (no API key required)")

# Fetch data (with simple throttle)
last_key = f"last_fetch_{symbol}"
now = time.time()
last = st.session_state.get(last_key, 0)
if (now - last) > refresh_sec or st.session_state.get('force_refresh', False):
    st.session_state[last_key] = now
    st.session_state['force_refresh'] = False
    tv_rows = get_tv_optionchain(symbol)
else:
    tv_rows = get_tv_optionchain(symbol)  # still fetch; TradingView is fast/cheap

if not tv_rows:
    st.error("No option chain data available from TradingView. Try again in a few seconds.")
    st.stop()

calls, puts = parse_tv_rows(tv_rows)

# Build combined dataframe
# adapt column names to helper expectations: CE_oi, CE_ltp etc.
def unify_lists(calls, puts):
    # rename keys for helpers
    calls_u = []
    for r in calls:
        calls_u.append({
            "strike": r["strike"],
            "expiry": r["expiry"],
            "CE_ltp": r["ltp"],
            "CE_oi": r["oi"],
            "CE_bid": r["bid"],
            "CE_ask": r["ask"],
            "CE_vol": r["vol"]
        })
    puts_u = []
    for r in puts:
        puts_u.append({
            "strike": r["strike"],
            "expiry": r["expiry"],
            "PE_ltp": r["ltp"],
            "PE_oi": r["oi"],
            "PE_bid": r["bid"],
            "PE_ask": r["ask"],
            "PE_vol": r["vol"]
        })
    # convert into unified dataframe using helpers (they expect raw lists but with CE_/PE_ prefixes)
    # easier: build df by merging
    ce_df = pd.DataFrame(calls_u).set_index("strike")
    pe_df = pd.DataFrame(puts_u).set_index("strike")
    df = pd.concat([ce_df.add_prefix("CE_"), pe_df.add_prefix("PE_")], axis=1, join="outer").reset_index()
    df = df.fillna(0)
    # ensure numeric types
    for col in df.columns:
        if col.startswith("CE_") or col.startswith("PE_"):
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            except Exception:
                pass
    return df

combined = unify_lists(calls, puts)
if combined.empty:
    st.error("Failed to build option chain table.")
    st.stop()

# detect ATM heuristically (strike with min CE_ltp+PE_ltp)
try:
    combined['mid_sum'] = combined.get('CE_ltp', 0) + combined.get('PE_ltp', 0)
    atm = int(combined.loc[combined['mid_sum'].idxmin()]['strike'])
except Exception:
    atm = int(combined['strike'].iloc[len(combined)//2])

# window around ATM
step = int(round(combined['strike'].diff().median() if len(combined)>1 else 50))
left = atm - strikes_each_side*step
right = atm + strikes_each_side*step
filtered = combined[(combined['strike'] >= left) & (combined['strike'] <= right)].copy().reset_index(drop=True)

# compute metrics
filtered['CE_oi'] = filtered['CE_oi'].astype(int)
filtered['PE_oi'] = filtered['PE_oi'].astype(int)
total_ce = int(filtered['CE_oi'].sum())
total_pe = int(filtered['PE_oi'].sum())
pcr = compute_pcr(filtered.rename(columns={"CE_oi":"CE_oi","PE_oi":"PE_oi"}))
max_pain, pain_df = compute_max_pain(filtered.rename(columns={"CE_oi":"CE_oi","PE_oi":"PE_oi"}))

# Top KPI row
k1,k2,k3,k4,k5 = st.columns([1.3,1.0,1.0,1.0,1.0])
k1.metric("Index", symbol)
k2.metric("ATM (approx)", f"{atm}")
k3.metric("Total CE OI", f"{total_ce:,d}")
k4.metric("Total PE OI", f"{total_pe:,d}")
k5.metric("PCR (PE/CE)", f"{pcr:.2f}" if pd.notna(pcr) else "N/A")

st.markdown("---")

# Charts and heatmap
c1,c2 = st.columns([2,1])
with c1:
    st.subheader("Open Interest — Grouped & Stacked")
    st.plotly_chart(plot_oi_grouped(filtered), use_container_width=True)
    st.plotly_chart(plot_oi_stacked(filtered), use_container_width=True)
with c2:
    st.subheader("Net OI & Max Pain")
    st.plotly_chart(plot_net_heatmap(filtered), use_container_width=True)
    st.metric("Max Pain", f"{max_pain if max_pain is not None else '—'}")

st.markdown("---")

# Strike ladder (CE left, Strike center, PE right) — HTML rendering for styling
max_oi = max(filtered['CE_oi'].max(), filtered['PE_oi'].max(), 1)

def bar_html(val, side='ce'):
    if val <= 0:
        return ""
    width = int((val/max_oi)*140)
    color = "#00d38a" if side=='ce' else "#ff6b6b"
    return f"<div style='display:inline-block;height:12px;width:{width}px;background:{color};border-radius:3px;margin-left:6px'></div>"

rows_html = []
for _, r in filtered.iterrows():
    s = int(r['strike'])
    ce_oi = int(r['CE_oi'])
    pe_oi = int(r['PE_oi'])
    ce_ltp = float(r.get('CE_ltp', 0.0))
    pe_ltp = float(r.get('PE_ltp', 0.0))
    bg = ""
    if s == atm:
        bg = "background: rgba(255,255,255,0.03);"
    if s == max_pain:
        bg = "background: rgba(255,200,0,0.06);"
    rows_html.append(f"""
    <tr style="{bg}">
      <td style="text-align:center;width:90px">{s}</td>
      <td style="text-align:right;width:140px">{ce_oi:,d} {bar_html(ce_oi,'ce')}</td>
      <td style="text-align:right;width:80px">{ce_ltp:.2f}</td>
      <td style="text-align:center;width:80px"></td>
      <td style="text-align:right;width:80px">{pe_ltp:.2f}</td>
      <td style="text-align:left;width:140px">{pe_oi:,d} {bar_html(pe_oi,'pe')}</td>
    </tr>
    """)

table_html = f"""
<div class="card">
<table style="width:100%;border-collapse:collapse;color:#dbe9f2;font-size:13px">
  <thead class="table-light" style="color:#9aa6b2">
    <tr>
      <th>Strike</th>
      <th>CE OI</th>
      <th>CE LTP</th>
      <th></th>
      <th>PE LTP</th>
      <th>PE OI</th>
    </tr>
  </thead>
  <tbody>
    {''.join(rows_html)}
  </tbody>
</table>
</div>
"""
st.markdown(table_html, unsafe_allow_html=True)

st.caption(f"Data via TradingView scanner. Last fetch: {datetime.utcfromtimestamp(st.session_state.get(last_key, time.time())).strftime('%Y-%m-%d %H:%M:%S')} UTC. Refresh: {refresh_sec}s")
