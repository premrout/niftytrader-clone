# pages/option_chain_playwright.py
r"""
NiftyTrader-style Option Chain page (Playwright-backed scraper)

This Streamlit page uses Playwright to render the JavaScript-driven NiftyTrader page
and extract the option-chain table. It provides NiftyTrader-like UI (CE/PE ladder,
bars, PCR, MaxPain, charts), plus an auto-refresh dropdown.

Prerequisites (local):
  python -m pip install -r requirements.txt
  python -m playwright install
"""

from typing import Optional, Tuple, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Option Chain (Playwright)", layout="wide")

# ------------------------------- Helpers & Plots -------------------------------
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

# ------------------------------- Playwright fetcher -------------------------------
@st.cache_data(ttl=8)
def fetch_table_with_playwright(symbol_slug: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Render the NiftyTrader page with Playwright and extract the first <table>.
    Returns (df, error). If error is not None, df will be None.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        return None, ("Playwright not installed or failed to import. "
                      "Install with: pip install playwright && python -m playwright install. "
                      f"Import error: {e}")

    base_url = "https://www.niftytrader.in/nse-option-chain/"
    url = base_url + symbol_slug

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
            context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
            page = context.new_page()
            page.goto(url, timeout=30000)
            # wait up to 20s for a table to appear
            page.wait_for_selector("table", timeout=20000)
            # read the first occurrence
            el = page.query_selector("table")
            if not el:
                context.close()
                browser.close()
                return None, "Page loaded but no <table> element found on target page."
            table_html = el.inner_html()
            context.close()
            browser.close()
    except Exception as e:
        return None, f"Playwright fetch error: {e}"

    # parse with pandas
    try:
        tbl_html = "<table>" + table_html + "</table>"
        df_list = pd.read_html(tbl_html)
        if not df_list:
            return None, "No table parsed by pandas after Playwright render."
        df = df_list[0]
    except Exception as e:
        return None, f"Failed parsing rendered table HTML: {e}"

    # standardize / canonicalize columns
    df = standardize_niftytrader_table(df)
    if df is None or df.empty:
        return None, "Parsed table appears empty or unexpected format (standardization failed)."
    return df, None

def standardize_niftytrader_table(df_raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Heuristically convert scraped table to canonical columns:
    Strike (int), CE_LTP (float), CE_OI (int), PE_LTP (float), PE_OI (int)
    This function is tolerant to column name variations.
    """
    df = df_raw.copy()
    df.columns = [str(c).strip().replace("\n"," ").replace("\r"," ") for c in df.columns]

    # attempt to find strike column
    strike_col = None
    for c in df.columns:
        low = c.lower()
        if "strike" in low:
            strike_col = c
            break
    if strike_col is None:
        # fallback: choose the column with most numeric values but not all equal
        num_counts = {c: pd.to_numeric(df[c].astype(str).str.replace(",",""), errors='coerce').notna().sum() for c in df.columns}
        if num_counts:
            strike_col = max(num_counts, key=num_counts.get)

    if strike_col is None:
        return None

    # build canonical DF
    canonical = pd.DataFrame()
    canonical['Strike'] = pd.to_numeric(df[strike_col].astype(str).str.replace(",",""), errors='coerce')
    canonical = canonical.dropna(subset=['Strike'])
    canonical['Strike'] = canonical['Strike'].astype(int)

    # find numeric columns around strike and map heuristically
    cols = list(df.columns)
    idx = cols.index(strike_col)
    left = cols[:idx]
    right = cols[idx+1:]

    # helper to pick nearest numeric col on the left/right
    def pick_numeric_from(cols_list, prefer_float=False):
        for c in cols_list[::-1] if not prefer_float else cols_list[::-1]:
            s = df[c].astype(str).str.replace(",","")
            # choose if has some numeric entries
            if pd.to_numeric(s, errors='coerce').notna().sum() > 0:
                return c
        return None

    ce_oi_col = pick_numeric_from(left)
    pe_oi_col = pick_numeric_from(right)
    ce_ltp_col = pick_numeric_from(left, prefer_float=True)
    pe_ltp_col = pick_numeric_from(right, prefer_float=True)

    # safe extractors
    def safe_int_series(col):
        if col and col in df.columns:
            return pd.to_numeric(df[col].astype(str).str.replace(",",""), errors='coerce').fillna(0).astype(int)
        else:
            return pd.Series([0]*len(canonical), dtype=int)

    def safe_float_series(col):
        if col and col in df.columns:
            return pd.to_numeric(df[col].astype(str).str.replace(",",""), errors='coerce').fillna(0.0).astype(float)
        else:
            return pd.Series([0.0]*len(canonical), dtype=float)

    canonical['CE_OI'] = safe_int_series(ce_oi_col).values
    canonical['PE_OI'] = safe_int_series(pe_oi_col).values
    canonical['CE_LTP'] = safe_float_series(ce_ltp_col).values
    canonical['PE_LTP'] = safe_float_series(pe_ltp_col).values

    # ensure ordering and return
    canonical = canonical[['Strike','CE_LTP','CE_OI','PE_LTP','PE_OI']]
    canonical = canonical.sort_values('Strike').reset_index(drop=True)
    return canonical

# ------------------------------- UI / Page --------------------------------
st.markdown("<h2 style='text-align:center'>NiftyTrader-style Option Chain (Playwright)</h2>", unsafe_allow_html=True)

# controls in sidebar
with st.sidebar:
    st.markdown("## Controls")
    index_choice = st.selectbox("Index", ["NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY"], index=0)
    refresh_sec = st.selectbox("Auto-refresh (sec)", [3,5,8,10,12,15], index=1)
    strikes_each_side = st.slider("Strikes each side of ATM", 3, 12, 5)
    if st.button("Force refresh"):
        fetch_table_with_playwright.clear()
        st.rerun()

# mapping slugs used by niftytrader
INDEX_SLUGS = {
    "NIFTY": "nifty",
    "BANKNIFTY": "bank-nifty",
    "FINNIFTY": "finnifty",
    "MIDCPNIFTY": "midcap-nifty"
}
slug = INDEX_SLUGS.get(index_choice, index_choice.lower())

# fetch table (cached)
with st.spinner("Rendering page & extracting option chain (this may take a few seconds)..."):
    df_raw, error = fetch_table_with_playwright(slug)

if error:
    st.error(error)
    st.stop()
if df_raw is None or df_raw.empty:
    st.error("No option chain parsed. The site structure may have changed or Playwright fetch failed.")
    st.stop()

# ATM heuristic: strike with minimum CE_LTP+PE_LTP
df_raw['mid_sum'] = df_raw['CE_LTP'] + df_raw['PE_LTP']
try:
    atm = int(df_raw.loc[df_raw['mid_sum'].idxmin()]['Strike'])
except Exception:
    atm = int(df_raw['Strike'].iloc[len(df_raw)//2])

# pick window around ATM
step = int(round(df_raw['Strike'].diff().median() if len(df_raw)>1 else 50))
left = atm - strikes_each_side * step
right = atm + strikes_each_side * step
filtered = df_raw[(df_raw['Strike'] >= left) & (df_raw['Strike'] <= right)].copy().reset_index(drop=True)

# compute metrics
pcr = compute_pcr(filtered)
max_pain = compute_max_pain(filtered)

# KPIs
c1,c2,c3,c4 = st.columns([1.2,1.0,1.0,1.0])
c1.metric("Index", index_choice)
c2.metric("ATM (approx)", f"{atm}")
c3.metric("PCR (PE/CE)", f"{pcr:.2f}" if not np.isnan(pcr) else "N/A")
c4.metric("Max Pain", f"{max_pain if max_pain is not None else '—'}")

st.markdown("---")

left_col, right_col = st.columns([2,1])
with left_col:
    st.subheader("Open Interest — Grouped & Stacked")
    st.plotly_chart(plot_oi_grouped(filtered), use_container_width=True)
    st.plotly_chart(plot_oi_stacked(filtered), use_container_width=True)
with right_col:
    st.subheader("Net OI Heatmap")
    st.plotly_chart(plot_net_heatmap(filtered), use_container_width=True)

st.markdown("---")

# Build strike ladder table HTML
max_oi = max(
    int(filtered['CE_OI'].max() if not filtered['CE_OI'].empty else 0),
    int(filtered['PE_OI'].max() if not filtered['PE_OI'].empty else 0),
    1
)

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
        ce = 0.0; pe = 0.0
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
