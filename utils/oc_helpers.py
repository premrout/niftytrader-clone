# pages/oi_monitor_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import requests
from typing import Tuple, Dict, Any

# Try to reuse utils.nse_api if available (preferred)
USE_UTILS_NSE = False
try:
    from utils.nse_api import get_raw_option_chain, get_index_quote, normalize_option_chain
    USE_UTILS_NSE = True
except Exception:
    USE_UTILS_NSE = False

# ---------- If utils.nse_api not available, provide a local fallback fetcher ----------
# This fallback uses the same NSE endpoints and cookie warm-up logic, with retries.
BASE = "https://www.nseindia.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain"
}

def create_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=8)
    except Exception:
        pass
    return s

def safe_json_request(session: requests.Session, url: str, retries: int = 6, timeout: int = 10):
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=timeout, headers=HEADERS)
            txt = r.text.strip()
            # HTML response likely means blocking - refresh session and retry
            if txt.startswith("<") or r.status_code in (401,403):
                session = create_session()
                time.sleep(1 + attempt * 0.2)
                continue
            return r.json()
        except Exception:
            session = create_session()
            time.sleep(1 + attempt * 0.2)
    raise RuntimeError(f"Failed to fetch JSON from {url}")

def fallback_get_raw_option_chain(symbol: str = "NIFTY") -> list:
    session = create_session()
    url = f"{BASE}/api/option-chain-indices?symbol={symbol}"
    data = safe_json_request(session, url)
    records = data.get("records", {}) if isinstance(data, dict) else {}
    return records.get("data", []) if isinstance(records, dict) else []

def fallback_get_index_quote(symbol: str = "NIFTY") -> Dict[str, Any]:
    session = create_session()
    url = f"{BASE}/api/quote-derivative?symbol={symbol}"
    try:
        data = safe_json_request(session, url)
        # try to extract underlyingValue or sensible price fields
        if isinstance(data, dict) and "underlyingValue" in data:
            return {"lastPrice": data.get("underlyingValue")}
    except Exception:
        pass
    return {}

# ---------- Safe converters (copied from your console script) ----------
def safe_float(x):
    try:
        return float(x or 0.0)
    except Exception:
        try:
            return float(str(x).replace(",", "").strip() or 0.0)
        except Exception:
            return 0.0

def safe_int(x):
    try:
        return int(x or 0)
    except Exception:
        try:
            return int(float(str(x).replace(",", "").strip() or 0))
        except Exception:
            return 0

# ---------- Business logic: build the same DataFrame columns as your console tool ----------
def build_option_chain_df(raw_records: list, selected_expiry: str = None) -> pd.DataFrame:
    """
    Normalize raw NSE `records.data` list into a DataFrame with columns:
    Strike, CE_LTP, PE_LTP, CE_OI, PE_OI, CE_VALUE, PE_VALUE, CE_CHG_VALUE, PE_CHG_VALUE, DIFF_VALUE, DIFF_PERCENT
    Optionally filter by expiry date (string).
    """
    rows = []
    for item in raw_records:
        # filter by expiry if provided
        expiry = item.get("expiryDate") or item.get("expiry") or None
        if selected_expiry and expiry and expiry != selected_expiry:
            continue
        strike = item.get("strikePrice") or item.get("strike")
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
        # safe percent relative to absolute magnitude; if both zero, percent = 0
        diff_percent = (diff_value / (abs(pe_chg_value) if abs(pe_chg_value) != 0 else 1)) * 100 if pe_chg_value != 0 else 0.0

        rows.append({
            "Strike": int(strike) if strike is not None else None,
            "Expiry": expiry,
            "CE_LTP": ce_ltp,
            "PE_LTP": pe_ltp,
            "CE_OI": ce_oi,
            "PE_OI": pe_oi,
            "CE_VALUE": ce_value,
            "PE_VALUE": pe_value,
            "CE_CHG_VALUE": ce_chg_value,
            "PE_CHG_VALUE": pe_chg_value,
            "DIFF_VALUE": diff_value,
            "DIFF_PERCENT": diff_percent
        })
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["Strike"])
    if not df.empty:
        df = df.sort_values("Strike").reset_index(drop=True)
    return df

# ---------- Utilities: pick nearby strikes (same as console) ----------
def pick_nearby_10(df: pd.DataFrame, atm: int) -> pd.DataFrame:
    if df.empty:
        return df
    below = df[df["Strike"] < atm].tail(5)
    at = df[df["Strike"] == atm]
    above = df[df["Strike"] > atm].head(5)
    combined = pd.concat([below, at, above])
    if len(combined) < 10:
        strikes = sorted(df["Strike"].unique())
        if atm in strikes:
            idx = strikes.index(atm)
        else:
            idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - atm))
        start = max(0, idx - 5)
        end = min(len(strikes), idx + 5 + 1)
        sel = strikes[start:end]
        combined = df[df["Strike"].isin(sel)]
    return combined.reset_index(drop=True)

# ---------- Trend logic (same as console) ----------
def trend_suggestion_combined(df: pd.DataFrame) -> Dict[str, Any]:
    total_ce_value = float(df["CE_VALUE"].sum())
    total_pe_value = float(df["PE_VALUE"].sum())
    total_ce_chg = float(df["CE_CHG_VALUE"].sum())
    total_pe_chg = float(df["PE_CHG_VALUE"].sum())
    total_ce_oi = int(df["CE_OI"].sum())
    total_pe_oi = int(df["PE_OI"].sum())

    pcr = (total_pe_oi / total_ce_oi) if total_ce_oi != 0 else 0.0

    if (total_pe_value > total_ce_value) and (total_pe_chg > total_ce_chg):
        value_trend = "Bullish (PE value & change dominate)"
        value_bias = 1
    elif (total_ce_value > total_pe_value) and (total_ce_chg > total_pe_chg):
        value_trend = "Bearish (CE value & change dominate)"
        value_bias = -1
    else:
        value_trend = "Neutral (mixed value signals)"
        value_bias = 0

    if pcr > 1.25:
        pcr_trend = "Bullish (High PCR)"
        pcr_bias = 1
    elif pcr < 0.75:
        pcr_trend = "Bearish (Low PCR)"
        pcr_bias = -1
    else:
        pcr_trend = "Neutral (PCR mid-range)"
        pcr_bias = 0

    combined_score = value_bias + pcr_bias
    if combined_score >= 2:
        final_trend = "📈 Strong Bullish"
    elif combined_score == 1:
        final_trend = "🔼 Bullish Bias"
    elif combined_score == 0:
        final_trend = "⚖ Sideways / Neutral"
    elif combined_score == -1:
        final_trend = "🔽 Bearish Bias"
    else:
        final_trend = "📉 Strong Bearish"

    return {
        "pcr": pcr,
        "final_trend": final_trend,
        "value_trend": value_trend,
        "pcr_trend": pcr_trend,
        "total_ce_value": total_ce_value,
        "total_pe_value": total_pe_value,
        "total_ce_chg": total_ce_chg,
        "total_pe_chg": total_pe_chg,
        "total_ce_oi": total_ce_oi,
        "total_pe_oi": total_pe_oi
    }

# ---------- Simple plotting helpers (plotly) ----------
import plotly.graph_objects as go
def plot_grouped_oi(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Strike'], y=df['CE_OI'], name='CE OI', marker_color='#00d38a'))
    fig.add_trace(go.Bar(x=df['Strike'], y=df['PE_OI'], name='PE OI', marker_color='#ff6b6b'))
    fig.update_layout(barmode='group', template='plotly_dark', height=320, margin=dict(t=20))
    return fig

def plot_diff_value(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Strike'], y=df['DIFF_VALUE'], name='CE_CHG_VALUE - PE_CHG_VALUE', marker_color='#ffaa00'))
    fig.update_layout(template='plotly_dark', height=240, margin=dict(t=20))
    return fig

def plot_net_heatmap(df: pd.DataFrame):
    net = (df['CE_OI'] - df['PE_OI']).tolist()
    fig = go.Figure(data=go.Heatmap(z=[net], x=df['Strike'].tolist(), y=['Net OI'], colorscale='RdBu', zmid=0))
    fig.update_layout(template='plotly_dark', height=140, margin=dict(t=10,b=10))
    return fig

# ---------- Streamlit UI ----------
st.set_page_config(page_title="OI Monitor — Integrated", layout="wide")
st.markdown("<h2 style='text-align:center'>OI Monitor — Integrated with NiftyTrader UI</h2>", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.markdown("## Controls")
    symbol = st.selectbox("Index", ["NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY"], index=0)
    refresh_sec = st.slider("Auto-refresh interval (sec)", min_value=5, max_value=60, value=15, step=1)
    nearby_count = st.slider("Strikes each side of ATM", min_value=3, max_value=12, value=5, step=1)
    st.write("")
    if st.button("Force refresh"):
        st.session_state['force_refresh'] = True
        st.rerun()

# Fetch raw option chain and quote (use utils if present)
last_key = f"oi_last_fetch_{symbol}"
now = time.time()
last = st.session_state.get(last_key, 0.0)
should_fetch = (now - last) > refresh_sec or st.session_state.get('force_refresh', False)

try:
    if USE_UTILS_NSE:
        raw = get_raw_option_chain(symbol)
        quote = get_index_quote(symbol)
    else:
        if should_fetch:
            # fallback: warmup and call
            raw = fallback_get_raw_option_chain(symbol)
            quote = fallback_get_index_quote(symbol)
        else:
            raw = fallback_get_raw_option_chain(symbol)
            quote = fallback_get_index_quote(symbol)
except Exception as e:
    st.error(f"Failed fetching data: {e}")
    st.stop()

if not raw:
    st.error("Option chain not available right now. NSE may be blocking requests or market might be closed.")
    st.stop()

# Optionally present expiry selector if multiple expiries found
expiries = []
try:
    records_meta = raw  # raw is list of rows; expiry values inside each row
    expiries = sorted(list({r.get("expiryDate") or r.get("expiry") for r in raw if r.get("expiryDate") or r.get("expiry")}))
except Exception:
    expiries = []

selected_expiry = None
if expiries:
    selected_expiry = st.selectbox("Expiry (select)", options=expiries, index=0)

# normalize to DataFrame (same columns as console)
df_all = build_option_chain_df(raw, selected_expiry)

if df_all.empty:
    st.error("No strikes found after normalization.")
    st.stop()

# compute ATM using quote if available, else mid-sum heuristic
spot = safe_float(quote.get("lastPrice") if isinstance(quote, dict) else quote.get("underlyingValue", None) if isinstance(quote, dict) else None)
try:
    if spot and spot > 0:
        # round to nearest 50 (NIFTY ticks often 50); if symbol != NIFTY you may want different step
        step = int(round(df_all['Strike'].diff().median() if len(df_all)>1 else 50))
        atm = int(round(spot / step) * step)
    else:
        df_all['mid_sum'] = df_all['CE_LTP'] + df_all['PE_LTP']
        atm = int(df_all.loc[df_all['mid_sum'].idxmin()]['Strike'])
except Exception:
    atm = int(df_all['Strike'].iloc[len(df_all)//2])

# pick nearby strikes based on nearby_count slider
def pick_nearby(df, atm_val, each_side):
    below = df[df["Strike"] < atm_val].tail(each_side)
    at = df[df["Strike"] == atm_val]
    above = df[df["Strike"] > atm_val].head(each_side)
    combined = pd.concat([below, at, above])
    if len(combined) < (each_side*2 + 1):
        strikes = sorted(df["Strike"].unique())
        if atm_val in strikes:
            idx = strikes.index(atm_val)
        else:
            idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - atm_val))
        start = max(0, idx - each_side)
        end = min(len(strikes), idx + each_side + 1)
        sel = strikes[start:end]
        combined = df[df["Strike"].isin(sel)]
    return combined.reset_index(drop=True)

df_near = pick_nearby(df_all, atm, nearby_count)

# add TOTAL row
total_row = {
    "Strike": "TOTAL",
    "Expiry": "",
    "CE_LTP": "",
    "PE_LTP": "",
    "CE_OI": int(df_near["CE_OI"].sum()),
    "PE_OI": int(df_near["PE_OI"].sum()),
    "CE_VALUE": float(df_near["CE_VALUE"].sum()),
    "PE_VALUE": float(df_near["PE_VALUE"].sum()),
    "CE_CHG_VALUE": float(df_near["CE_CHG_VALUE"].sum()),
    "PE_CHG_VALUE": float(df_near["PE_CHG_VALUE"].sum()),
    "DIFF_VALUE": float(df_near["DIFF_VALUE"].sum()),
    "DIFF_PERCENT": float(df_near["DIFF_PERCENT"].mean() if len(df_near)>0 else 0.0)
}
df_display = pd.concat([df_near, pd.DataFrame([total_row])], ignore_index=True)

# compute trend meta
meta = trend_suggestion_combined(df_near)

# --- UI layout ---
st.markdown("### Overview")
col1, col2, col3, col4, col5 = st.columns([1.2,1.0,1.0,1.0,1.0])
col1.metric("Index", symbol)
col2.metric("ATM (approx)", f"{atm}")
col3.metric("PCR (PE/CE)", f"{meta['pcr']:.2f}")
col4.metric("Total CE Value", f"{meta['total_ce_value']:,.0f}")
col5.metric("Total PE Value", f"{meta['total_pe_value']:,.0f}")

st.markdown("**Final Trend:** " + meta['final_trend'])
st.markdown("**Value Trend:** " + meta['value_trend'] + "  |  **PCR Trend:** " + meta['pcr_trend'])

st.markdown("---")
left, right = st.columns([2,1])
with left:
    st.subheader("OI Charts")
    st.plotly_chart(plot_grouped_oi(df_near), use_container_width=True)
    st.plotly_chart(plot_diff_value(df_near), use_container_width=True)
with right:
    st.subheader("Net OI Heatmap & Max Pain")
    st.plotly_chart(plot_net_heatmap(df_near), use_container_width=True)
    # compute max pain quickly
    try:
        # compute max pain in-stream
        strikes = sorted(df_near['Strike'].unique())
        pain_list = []
        for s in strikes:
            total_pain = 0
            for _, row in df_near.iterrows():
                k = int(row['Strike'])
                ce_oi = int(row.get('CE_OI',0))
                pe_oi = int(row.get('PE_OI',0))
                ce_payoff = s - k if s > k else 0
                pe_payoff = k - s if k > s else 0
                total_pain += ce_oi * ce_payoff + pe_oi * pe_payoff
            pain_list.append((s,int(total_pain)))
        max_pain = min(pain_list, key=lambda t: t[1])[0] if pain_list else None
    except Exception:
        max_pain = None
    st.metric("Max Pain (approx)", f"{max_pain if max_pain is not None else '—'}")

st.markdown("---")
st.subheader("Strike Ladder (ATM highlighted)")

# Build HTML table similar to NiftyTrader style
max_oi = max(int(df_near['CE_OI'].max() if not df_near['CE_OI'].empty else 0), int(df_near['PE_OI'].max() if not df_near['PE_OI'].empty else 0), 1)
def bar_html(val, side='ce'):
    if val <= 0:
        return ""
    width = int((val/max_oi)*140)
    color = "#00d38a" if side=='ce' else "#ff6b6b"
    return f"<div style='display:inline-block;height:12px;width:{width}px;background:{color};border-radius:3px;margin-left:6px'></div>"

rows_html = []
for _, r in df_display.iterrows():
    s = r["Strike"]
    is_atm = (s == atm)
    bg = ""
    if s == "TOTAL":
        bg = "background: rgba(255,255,255,0.02); font-weight:700;"
    elif is_atm:
        bg = "background: rgba(255,255,255,0.04);"
    rows_html.append(f"""
    <tr style="{bg}">
      <td style="text-align:center;width:90px">{s}</td>
      <td style="text-align:right;width:120px">{int(r['CE_OI']) if str(s) != 'TOTAL' else r['CE_OI']:,d} {bar_html(int(r['CE_OI']) if str(s) != 'TOTAL' else r['CE_OI'],'ce')}</td>
      <td style="text-align:right;width:90px">{r['CE_LTP'] if str(s) != 'TOTAL' else ''}</td>
      <td style="text-align:center;width:60px"></td>
      <td style="text-align:right;width:90px">{r['PE_LTP'] if str(s) != 'TOTAL' else ''}</td>
      <td style="text-align:left;width:120px">{int(r['PE_OI']) if str(s) != 'TOTAL' else r['PE_OI']:,d} {bar_html(int(r['PE_OI']) if str(s) != 'TOTAL' else r['PE_OI'],'pe')}</td>
      <td style="text-align:right;width:120px">{('%.0f' % r['DIFF_VALUE']) if str(s) != 'TOTAL' else ('%.0f' % r['DIFF_VALUE'])}</td>
    </tr>
    """)

table_html = f"""
<div style="border-radius:8px;padding:8px;background:#0b1220;border:1px solid rgba(255,255,255,0.03)">
<table style="width:100%;border-collapse:collapse;color:#dbe9f2;font-size:13px">
  <thead style="color:#9aa6b2">
    <tr>
      <th>Strike</th><th>CE OI</th><th>CE LTP</th><th></th><th>PE LTP</th><th>PE OI</th><th>DIFF_VAL</th>
    </tr>
  </thead>
  <tbody>
    {''.join(rows_html)}
  </tbody>
</table>
</div>
"""
st.markdown(table_html, unsafe_allow_html=True)

# CSV download
csv = df_display.to_csv(index=False)
st.download_button("Download displayed CSV", csv, file_name=f"{symbol}_oi_monitor_{int(time.time())}.csv", mime="text/csv")

st.caption(f"Data source: NSE public JSON endpoints. Last fetch: {datetime.utcfromtimestamp(st.session_state.get(last_key, time.time())).strftime('%Y-%m-%d %H:%M:%S')} UTC. Auto-refresh: {refresh_sec}s.")

# update last fetch stamp
st.session_state[last_key] = time.time()
st.session_state['force_refresh'] = False
