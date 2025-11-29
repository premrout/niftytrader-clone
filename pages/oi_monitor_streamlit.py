# pages/option_chain_playwright.py
"""
NiftyTrader-style Option Chain page using Playwright to render the JS-driven website.

Place this file under: C:\Users\premr\nifty_app\pages\option_chain_playwright.py

Requires:
  pip install streamlit pandas plotly requests beautifulsoup4 lxml playwright
  python -m playwright install
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
import plotly.graph_objects as go

st.set_page_config(page_title="Option Chain (Playwright)", layout="wide")

# --------------------------- Helpers: parsing & analytics ---------------------------
def compute_pcr(df: pd.DataFrame) -> float:
    ce = int(df['CE_OI'].sum()) if 'CE_OI' in df.columns else 0
    pe = int(df['PE_OI'].sum()) if 'PE_OI' in df.columns else 0
    if ce == 0:
        return float('nan')
    return float(pe) / float(ce)

def compute_max_pain(df: pd.DataFrame) -> Optional[int]:
    if df.empty:
        return None
    strikes = sorted(df['Strike'].unique())
    pain_list = []
    for s in strikes:
        total_pain = 0
        for _, row in df.iterrows():
            k = int(row['Strike'])
            ce_oi = int(row.get('CE_OI', 0))
            pe_oi = int(row.get('PE_OI', 0))
            ce_payoff = s - k if s > k else 0
            pe_payoff = k - s if k > s else 0
            total_pain += ce_oi * ce_payoff + pe_oi * pe_payoff
        pain_list.append({'strike': s, 'total_pain': int(total_pain)})
    pain_df = pd.DataFrame(pain_list).sort_values('total_pain').reset_index(drop=True)
    return int(pain_df.iloc[0]['strike']) if not pain_df.empty else None

def plot_oi_grouped(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Strike'], y=df['CE_OI'], name='CE OI', marker_color='#00d38a'))
    fig.add_trace(go.Bar(x=df['Strike'], y=df['PE_OI'], name='PE OI', marker_color='#ff6b6b'))
    fig.update_layout(barmode='group', template='plotly_dark', height=320, margin=dict(t=30))
    return fig

def plot_oi_stacked(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Strike'], y=df['CE_OI'], name='CE OI', marker_color='#00d38a'))
    fig.add_trace(go.Bar(x=df['Strike'], y=df['PE_OI'], name='PE OI', marker_color='#ff6b6b'))
    fig.update_layout(barmode='stack', template='plotly_dark', height=320, margin=dict(t=30))
    return fig

def plot_net_heatmap(df: pd.DataFrame):
    df2 = df.copy()
    df2['net'] = df2['CE_OI'] - df2['PE_OI']
    z = [df2['net'].tolist()]
    x = df2['Strike'].tolist()
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=['Net OI'], colorscale='RdBu', zmid=0))
    fig.update_layout(template='plotly_dark', height=140, margin=dict(t=10,b=10))
    return fig

# --------------------------- Playwright fetcher (cached) ---------------------------
@st.cache_data(ttl=10)
def fetch_table_with_playwright(symbol_slug: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Use Playwright to load NiftyTrader's JS-driven page and extract the rendered option chain table.
    Returns (df, error_message). If error_message is None, df is valid.
    """
    # Import inside function so app can show friendly message if Playwright not installed
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        return None, f"Playwright not installed or failed to import: {e}. Install with: pip install playwright && python -m playwright install"

    base_url = "https://www.niftytrader.in/nse-option-chain/"
    url = base_url + symbol_slug

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
            context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
            page = context.new_page()
            page.goto(url, timeout=30000)
            # wait for a table element; NiftyTrader often renders table after JS; give some time
            page.wait_for_selector("table", timeout=20000)
            # get the first table's outerHTML
            table_html = page.query_selector("table").inner_html()
            # Close browser
            context.close()
            browser.close()
    except Exception as e:
        return None, f"Playwright fetch error: {e}"

    # parse with pandas.read_html (lxml)
    try:
        # pandas expects a full <table> element; wrap inner_html
        import io
        tbl_html = "<table>" + table_html + "</table>"
        df_list = pd.read_html(tbl_html)
        if not df_list:
            return None, "No tables found by pandas after Playwright fetch."
        df = df_list[0]
    except Exception as e:
        return None, f"Failed parsing table HTML: {e}"

    # Standardize columns: try to detect and rename to consistent names
    df = standardize_niftytrader_table(df)
    if df is None or df.empty:
        return None, "Parsed table appears empty or unexpected format."

    return df, None

def standardize_niftytrader_table(df_raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Convert the scraped DataFrame into canonical columns:
    Strike, CE_LTP, CE_OI, CE_Change_OI, CE_Vol, PE_LTP, PE_OI, PE_Change_OI, PE_Vol
    This function tries common NiftyTrader column headings and heuristically maps them.
    """
    df = df_raw.copy()
    # normalize column names
    cols = [str(c).strip() for c in df.columns]
    normalized = [c.replace("\n", " ").strip() for c in cols]
    df.columns = normalized

    # Heuristic mapping: find strike column (often labelled 'Strike Price' or 'Strike')
    strike_col = None
    for c in df.columns:
        if "Strike" in c or "Strike Price" in c or c.lower().startswith("strike"):
            strike_col = c
            break
    if strike_col is None:
        # try first numeric column
        numeric_cols = [c for c in df.columns if pd.to_numeric(df[c], errors='coerce').notna().sum() > 0]
        if numeric_cols:
            strike_col = numeric_cols[len(numeric_cols)//2]  # guess
    if strike_col is None:
        return None

    # Identify CE/PE group columns by approximate names
    # We'll search for keywords
    ce_ltp = None; ce_oi = None; ce_chg = None; ce_vol = None
    pe_ltp = None; pe_oi = None; pe_chg = None; pe_vol = None

    for c in df.columns:
        low = c.lower()
        if "ce" in low or "call" in low:
            # skip for now - name may contain words; continue scanning
            pass
        # ltp keywords
        if ("ltp" in low or "last" in low or "last price" in low) and ce_ltp is None:
            # heuristically assign by proximity to strike: check relative position
            ce_ltp = c
            continue
        if ("open interest" in low or "oi" in low) and ce_oi is None:
            # assign by proximity first occurrence
            # We will decide CE vs PE later by comparing columns positions relative to strike
            if ce_oi is None:
                ce_oi = c
            continue
        if ("change" in low and "oi" in low) and ce_chg is None:
            ce_chg = c
            continue
        if ("volume" in low or "vol" in low) and ce_vol is None:
            ce_vol = c
            continue

    # fallback heuristics: take columns by index relative to strike
    col_list = list(df.columns)
    strike_idx = col_list.index(strike_col)
    # typical NiftyTrader layout: [CE_OI, CE_Chg, CE_LTP, Strike, PE_LTP, PE_Chg, PE_OI]
    try:
        left = col_list[:strike_idx]
        right = col_list[strike_idx+1:]
        # pick rightmost of left as CE_OI, middle-right leftmost as CE_LTP etc.
        if not ce_oi and left:
            ce_oi = left[0] if len(left) >= 1 else None
        if not ce_ltp and left:
            # maybe last of left
            ce_ltp = left[-1] if len(left) >= 1 else ce_ltp
        if not pe_oi and right:
            pe_oi = right[-1] if len(right) >= 1 else None
        if not pe_ltp and right:
            pe_ltp = right[0] if len(right) >= 1 else None
    except Exception:
        pass

    # Build canonical df with safe conversion
    canonical = pd.DataFrame()
    canonical['Strike'] = pd.to_numeric(df[strike_col], errors='coerce').astype('Int64')
    # helper to extract numeric column if exists
    def pick_numeric(colname):
        if colname and colname in df.columns:
            return pd.to_numeric(df[colname].astype(str).str.replace(",",""), errors='coerce').fillna(0).astype(int)
        else:
            return pd.Series([0]*len(df)).astype(int)

    canonical['CE_OI'] = pick_numeric(ce_oi)
    canonical['PE_OI'] = pick_numeric(pe_oi)
    # try to find LTPs from neighboring columns
    # attempt simple numeric search for columns with floats near option prices
    float_cols = []
    for c in df.columns:
        cnt = pd.to_numeric(df[c].astype(str).str.replace(",",""), errors='coerce').notna().sum()
        if cnt > 0:
            float_cols.append(c)
    # prefer columns that are not the strike or OI
    float_candidates = [c for c in float_cols if c not in (strike_col, ce_oi, pe_oi)]
    ce_ltp_col = None
    pe_ltp_col = None
    if float_candidates:
        # choose candidate closest to strike index on left for CE, right for PE
        try:
            # find indices
            for c in float_candidates:
                if df.columns.get_loc(c) < df.columns.get_loc(strike_col) and ce_ltp_col is None:
                    ce_ltp_col = c
                if df.columns.get_loc(c) > df.columns.get_loc(strike_col) and pe_ltp_col is None:
                    pe_ltp_col = c
        except Exception:
            ce_ltp_col = float_candidates[0] if float_candidates else None
            pe_ltp_col = float_candidates[-1] if float_candidates else None

    def pick_float(colname):
        if colname and colname in df.columns:
            return pd.to_numeric(df[colname].astype(str).str.replace(",",""), errors='coerce').fillna(0.0).astype(float)
        else:
            return pd.Series([0.0]*len(df)).astype(float)

    canonical['CE_LTP'] = pick_float(ce_ltp_col)
    canonical['PE_LTP'] = pick_float(pe_ltp_col)

    # fill missing columns as zeros if required
    for k in ['CE_Change_OI', 'PE_Change_OI', 'CE_Vol', 'PE_Vol']:
        canonical[k] = 0

    # rename columns to expected names used across the app
    # ensure Strike is int
    canonical = canonical.dropna(subset=['Strike'])
    canonical['Strike'] = canonical['Strike'].astype(int)
    # sort and return
    canonical = canonical.sort_values('Strike').reset_index(drop=True)
    return canonical

# --------------------------- UI & main logic ---------------------------

# map user-friendly indices to NiftyTrader slugs
INDEX_SLUGS = {
    "NIFTY": "nifty",
    "BANKNIFTY": "bank-nifty",
    "FINNIFTY": "finnifty",
    "MIDCPNIFTY": "midcap-nifty"  # try a sensible slug; falls back to lowercase symbol if missing
}

st.markdown("<h2 style='text-align:center'>NiftyTrader-style Option Chain (Playwright)</h2>", unsafe_allow_html=True)

# sidebar controls
with st.sidebar:
    st.markdown("## Controls")
    index_choice = st.selectbox("Index", ["NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY"], index=0)
    refresh_sec = st.selectbox("Auto-refresh (sec)", [3,5,8,10,12,15], index=1)
    strikes_each_side = st.slider("Strikes each side of ATM", 3, 12, 5)
    if st.button("Force refresh"):
        # clear cache and rerun
        fetch_table_with_playwright.clear()
        st.experimental_rerun()

# symbol slug
slug = INDEX_SLUGS.get(index_choice, index_choice.lower())

# fetch table (cached)
with st.spinner("Fetching option chain (Playwright)…"):
    df_raw, err = fetch_table_with_playwright(slug)

if err:
    st.error(err)
    st.stop()

if df_raw is None or df_raw.empty:
    st.error("Parsed option chain empty. Playwright fetched table but parsing failed.")
    st.stop()

# compute ATM: try using mid-sum heuristic; NiftyTrader shows spot separately (we could fetch elsewhere)
df_raw['mid_sum'] = df_raw['CE_LTP'] + df_raw['PE_LTP']
try:
    atm = int(df_raw.loc[df_raw['mid_sum'].idxmin()]['Strike'])
except Exception:
    atm = int(df_raw['Strike'].iloc[len(df_raw)//2])

# window around ATM
step = int(round(df_raw['Strike'].diff().median() if len(df_raw)>1 else 50))
left = atm - strikes_each_side * step
right = atm + strikes_each_side * step
filtered = df_raw[(df_raw['Strike'] >= left) & (df_raw['Strike'] <= right)].copy().reset_index(drop=True)

# compute metrics
pcr = compute_pcr(filtered)
max_pain = compute_max_pain(filtered)

# header KPIs
c1,c2,c3,c4 = st.columns([1.2,1.0,1.0,1.0])
c1.metric("Index", index_choice)
c2.metric("ATM (approx)", f"{atm}")
c3.metric("PCR (PE/CE)", f"{pcr:.2f}" if not np.isnan(pcr) else "N/A")
c4.metric("Max Pain", f"{max_pain if max_pain is not None else '—'}")

st.markdown("---")

# charts
left_col, right_col = st.columns([2,1])
with left_col:
    st.subheader("Open Interest — Grouped & Stacked")
    st.plotly_chart(plot_oi_grouped(filtered), use_container_width=True)
    st.plotly_chart(plot_oi_stacked(filtered), use_container_width=True)
with right_col:
    st.subheader("Net OI Heatmap")
    st.plotly_chart(plot_net_heatmap(filtered), use_container_width=True)

st.markdown("---")

# Strike ladder table (HTML)
max_oi = max(int(filtered['CE_OI'].max() if not filtered['CE_OI'].empty else 0),
             int(filtered['PE_OI'].max() if not filtered['PE_OI'].empty else 0), 1)

# improved bar rendering (caps, total composite)
MAX_BAR_PX = 140
TOTAL_BAR_PX = 180
MIN_BAR_PX = 2

def fmt_int_commas(x):
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)

def signed_arrow_html(v):
    try:
        n = float(v)
    except Exception:
        return f"<span>{v}</span>"
    if n > 0:
        return f"<span style='color:#00ff99;font-weight:700;'>{int(n):,} ▲</span>"
    elif n < 0:
        return f"<span style='color:#ff8b8b;font-weight:700;'>{int(n):,} ▼</span>"
    else:
        return f"<span style='color:#9aa6b2;'>{int(n):,} →</span>"

def strike_bar_html(val, max_oi, side='ce', cap_px=MAX_BAR_PX):
    try:
        v = int(val)
    except Exception:
        v = 0
    if v <= 0 or max_oi <= 0:
        return ""
    width = int((v / float(max_oi)) * cap_px)
    if width < MIN_BAR_PX:
        width = MIN_BAR_PX
    if width > cap_px:
        width = cap_px
    color = "#00d38a" if side == 'ce' else "#ff6b6b"
    return f"<div style='display:inline-block;height:12px;width:{width}px;background:{color};border-radius:3px;margin-left:6px'></div>"

def total_composite_bar_html(ce_total, pe_total, cap_px=TOTAL_BAR_PX):
    try:
        ce = float(ce_total)
        pe = float(pe_total)
    except Exception:
        ce = float(0); pe = float(0)
    tot = ce + pe
    if tot <= 0:
        return f"<div style='display:inline-block;height:12px;width:{cap_px}px;background:rgba(255,255,255,0.03);border-radius:3px;margin-left:6px'></div>"
    ce_w = max(int((ce / tot) * cap_px), 1) if ce > 0 else 0
    pe_w = max(int((pe / tot) * cap_px), 1) if pe > 0 else 0
    if ce_w + pe_w > cap_px:
        if ce_w >= pe_w:
            ce_w -= (ce_w + pe_w - cap_px)
        else:
            pe_w -= (ce_w + pe_w - cap_px)
    ce_html = f"<div style='display:inline-block;height:12px;width:{ce_w}px;background:#00d38a;border-radius:3px;'></div>" if ce_w>0 else ""
    pe_html = f"<div style='display:inline-block;height:12px;width:{pe_w}px;background:#ff6b6b;border-radius:3px;'></div>" if pe_w>0 else ""
    return f"<div style='display:inline-flex; align-items:center; border-radius:3px; overflow:hidden; width:{cap_px}px; border:1px solid rgba(255,255,255,0.03); margin-left:6px'>{ce_html}{pe_html}</div>"

rows_html = []
# pick nearby strikes + total row
display_df = filtered.copy()
total_row = {
    "Strike": "TOTAL",
    "CE_OI": int(display_df['CE_OI'].sum()),
    "PE_OI": int(display_df['PE_OI'].sum()),
    "CE_LTP": "",
    "PE_LTP": "",
    "DIFF_VAL": int((display_df['CE_LTP']*display_df['CE_OI']).sum() - (display_df['PE_LTP']*display_df['PE_OI']).sum())
}
display_df = pd.concat([display_df, pd.DataFrame([total_row])], ignore_index=True)

for _, r in display_df.iterrows():
    s = r["Strike"]
    is_total = (str(s).upper() == "TOTAL")
    is_atm = (s == atm)
    ce_oi = r['CE_OI'] if not is_total else r['CE_OI']
    pe_oi = r['PE_OI'] if not is_total else r['PE_OI']
    if is_total:
        ce_bar = total_composite_bar_html(ce_oi, pe_oi, cap_px=TOTAL_BAR_PX)
        pe_bar = ""
        ce_disp = fmt_int_commas(ce_oi)
        pe_disp = fmt_int_commas(pe_oi)
    else:
        ce_bar = strike_bar_html(ce_oi, max_oi, side='ce', cap_px=MAX_BAR_PX)
        pe_bar = strike_bar_html(pe_oi, max_oi, side='pe', cap_px=MAX_BAR_PX)
        ce_disp = fmt_int_commas(ce_oi)
        pe_disp = fmt_int_commas(pe_oi)

    diff_val = int(r.get('DIFF_VAL', r.get('DIFF_VALUE', 0)))
    diff_html = signed_arrow_html(diff_val)

    bg = ""
    if is_total:
        bg = "background: rgba(255,255,255,0.02); font-weight:700;"
    elif is_atm:
        bg = "background: rgba(255,255,255,0.04);"

    ce_ltp = "" if is_total else (f"{r.get('CE_LTP'):.2f}" if r.get('CE_LTP') else "")
    pe_ltp = "" if is_total else (f"{r.get('PE_LTP'):.2f}" if r.get('PE_LTP') else "")

    rows_html.append(f"""
    <tr style="{bg}">
      <td style="text-align:center;width:90px">{s}</td>
      <td style="text-align:right;width:120px">{ce_disp} {ce_bar}</td>
      <td style="text-align:right;width:90px">{ce_ltp}</td>
      <td style="text-align:center;width:60px"></td>
      <td style="text-align:right;width:90px">{pe_ltp}</td>
      <td style="text-align:left;width:120px">{pe_disp} {pe_bar}</td>
      <td style="text-align:right;width:120px">{diff_html}</td>
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
csv = pd.DataFrame(display_df).to_csv(index=False)
st.download_button("Download displayed CSV", csv, file_name=f"{index_choice}_option_chain_{int(time.time())}.csv", mime="text/csv")

st.caption(f"Data source: niftytrader.in (rendered via Playwright). Last fetch: {datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')} UTC. Auto-refresh: {refresh_sec}s.")
