# pages/option_chain_no_playwright.py
"""
Streamlit Option Chain (no Playwright).
Shows nearest expiry, configurable strikes around ATM (default 5 each side),
OI-change chart, PCR & PCR-change, RSI (synthetic from option prices),
and a momentum suggestion that uses RSI + OI change.

Auto-refreshes data every 15 seconds by updating the main placeholder only.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Option Chain (NSE API)", layout="wide")

# ---------- Config ----------
REFRESH_SECONDS = 15  # data refresh interval
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com"
}

# ------------------ Utilities ------------------
def safe_session_get(session: requests.Session, url: str, headers=None, timeout=8, retries=3):
    headers = headers or HEADERS
    for i in range(retries):
        try:
            r = session.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
            # transient server error -> retry
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Failed to GET {url} after {retries} attempts")

@st.cache_data(ttl=8, show_spinner=False)
def fetch_nse_chain(symbol: str = "NIFTY"):
    """
    Fetch JSON option-chain from NSE for an index symbol (NIFTY, BANKNIFTY, ...).
    Returns parsed JSON dict. Cached for a short TTL.
    """
    base = "https://www.nseindia.com"
    url = f"{base}/api/option-chain-indices?symbol={symbol}"
    s = requests.Session()
    # initial GET to obtain cookies / session headers (NSE often needs this)
    try:
        s.get(base, headers=HEADERS, timeout=5)
    except Exception:
        # ignore initial fail, try direct
        pass
    r = safe_session_get(s, url, headers=HEADERS, timeout=10, retries=4)
    return r.json()

def parse_chain(json_data):
    """
    Parse NSE JSON into a DataFrame with required fields (nearest expiry only).
    """
    rows = []
    records = json_data.get("records", {}) if json_data else {}
    data_list = records.get("data", []) if records else []
    expiry_dates = records.get("expiryDates", []) if records else []
    nearest_expiry = expiry_dates[0] if expiry_dates else None

    for item in data_list:
        # filter for nearest expiry only
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
    # coerce types
    df["Strike"] = pd.to_numeric(df["Strike"], errors="coerce")
    for col in ["CE_LTP", "PE_LTP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in ["CE_OI", "PE_OI", "CE_CHG_OI", "PE_CHG_OI"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["Strike"]).sort_values("Strike").reset_index(drop=True)
    return df

# ------------------ RSI ------------------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI using Wilder smoothing (EMA-like). Returns series aligned with input.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ------------------ UI controls ------------------
st.title("📊 Option Chain — NSE (No Playwright)")
st.caption(f"Nearest expiry. Data refresh every {REFRESH_SECONDS}s (data-only refresh).")

with st.sidebar:
    st.markdown("## Controls")
    index_choice = st.selectbox("Index", ["NIFTY", "BANKNIFTY", "FINNIFTY"], index=0)
    strikes_each_side = st.slider("Strikes each side of ATM", 3, 12, 5)
    if st.button("Force refresh (clear cache)"):
        fetch_nse_chain.clear()
        st.experimental_rerun()

# ------------------ placeholder loop (data-only refresh) ------------------
main_placeholder = st.empty()

# We'll run a loop that updates the placeholder with current data every REFRESH_SECONDS.
# The loop will stop if user interacts (navigates away) or if Streamlit session ends.
# Note: long-running loops are OK for this pattern because the UI is updated in the placeholder.
run_count = 0
while True:
    run_count += 1
    with main_placeholder.container():
        # Fetch + parse
        try:
            raw = fetch_nse_chain(index_choice)
        except Exception as e:
            st.error(f"Failed to fetch data from NSE: {e}")
            break

        df_all = parse_chain(raw)
        if df_all.empty:
            st.error("No option chain data parsed. NSE structure may have changed or request blocked.")
            break

        expiry_dates = raw.get("records", {}).get("expiryDates", [])
        nearest_expiry = expiry_dates[0] if expiry_dates else None
        st.markdown(f"**Expiry used:** {nearest_expiry if nearest_expiry else 'Not available'}")

        # show spot (if present)
        spot = raw.get("records", {}).get("underlyingValue")
        spot_val = None
        if spot is not None:
            try:
                spot_val = float(spot)
            except Exception:
                spot_val = None

        if spot_val is not None:
            st.metric(label=f"{index_choice} Spot", value=f"{spot_val:,.2f}")
        else:
            st.markdown("**Spot price not available in payload**")

        # Determine ATM
        if spot_val is not None:
            strikes = df_all["Strike"].unique()
            atm = int(min(strikes, key=lambda s: abs(s - spot_val)))
        else:
            df_all["mid_sum"] = df_all["CE_LTP"] + df_all["PE_LTP"]
            idx = df_all["mid_sum"].idxmin() if not df_all["mid_sum"].isna().all() else None
            if idx is not None:
                atm = int(df_all.loc[idx, "Strike"])
            else:
                atm = int(df_all["Strike"].iloc[len(df_all)//2])

        # compute step and window
        step = int(round(df_all["Strike"].diff().median())) if len(df_all) > 1 else 50
        left_bound = atm - strikes_each_side * step
        right_bound = atm + strikes_each_side * step
        df_window = df_all[(df_all["Strike"] >= left_bound) & (df_all["Strike"] <= right_bound)].copy().reset_index(drop=True)

        # Table with ATM highlighted
        st.subheader(f"Nearby Option Strikes around ATM {atm} (±{strikes_each_side} strikes)")
        def highlight_atm(row):
            return ['background-color: #1f2a30; color: #ffd700' if row['Strike'] == atm else '' for _ in row]
        styled = (df_window.style
                  .apply(highlight_atm, axis=1)
                  .format({
                      "Strike": "{:,.0f}",
                      "CE_OI": "{:,}",
                      "PE_OI": "{:,}",
                      "CE_LTP": "{:,.2f}",
                      "PE_LTP": "{:,.2f}",
                      "CE_CHG_OI": "{:,}",
                      "PE_CHG_OI": "{:,}"
                  }))
        st.dataframe(styled, use_container_width=True)

        # OI change chart
        st.subheader("CE & PE ΔOI — Today (Nearby strikes)")
        def plot_oi_change(df: pd.DataFrame):
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df["Strike"], y=df["CE_CHG_OI"], name="CE ΔOI"))
            fig.add_trace(go.Bar(x=df["Strike"], y=df["PE_CHG_OI"], name="PE ΔOI"))
            fig.update_layout(barmode="group", template="plotly_dark", height=360,
                              xaxis_title="Strike", yaxis_title="Change in OI")
            return fig
        st.plotly_chart(plot_oi_change(df_window), use_container_width=True)

        # PCR (window) & line
        st.subheader("PCR (PE_OI / CE_OI) — Window")
        pcr_total_ce = df_window["CE_OI"].sum()
        pcr_total_pe = df_window["PE_OI"].sum()
        pcr = (pcr_total_pe / pcr_total_ce) if pcr_total_ce else 0
        pcr_change = (df_window["PE_CHG_OI"].sum() - df_window["CE_CHG_OI"].sum()) / max(pcr_total_ce, 1)

        st.markdown(f"- **PCR (window):** {pcr:.3f}")
        st.markdown(f"- **PCR change (today, window):** {pcr_change:.3f}")

        # PCR line
        pcr_line = df_window["PE_OI"] / df_window["CE_OI"].replace(0, np.nan)
        fig_pcr = go.Figure()
        fig_pcr.add_trace(go.Scatter(x=df_window["Strike"], y=pcr_line, mode="lines+markers", name="PCR"))
        fig_pcr.update_layout(template="plotly_dark", height=300, yaxis_title="PCR", xaxis_title="Strike")
        st.plotly_chart(fig_pcr, use_container_width=True)

        # RSI (synthetic)
        st.subheader("RSI (synthetic) — derived from option prices")
        df_window["SYM_PRICE"] = (df_window["CE_LTP"] + df_window["PE_LTP"]) / 2.0
        rsi_series = compute_rsi(df_window["SYM_PRICE"], period=14)
        df_window["RSI"] = rsi_series

        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=df_window["Strike"], y=df_window["SYM_PRICE"], mode="lines+markers", name="Synthetic Price"))
        fig_price.update_layout(template="plotly_dark", height=300, yaxis_title="Synthetic Price", xaxis_title="Strike")

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df_window["Strike"], y=df_window["RSI"], mode="lines+markers", name="RSI"))
        fig_rsi.update_layout(template="plotly_dark", height=260, yaxis_title="RSI", yaxis=dict(range=[0,100]), xaxis_title="Strike")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(fig_price, use_container_width=True)
        with col2:
            st.plotly_chart(fig_rsi, use_container_width=True)

        # Momentum suggestion
        st.subheader("Momentum suggestion (RSI + OI-change)")
        latest_rsi_val = float(df_window["RSI"].dropna().iloc[-1]) if not df_window["RSI"].dropna().empty else None
        total_ce_chg = int(df_window["CE_CHG_OI"].sum())
        total_pe_chg = int(df_window["PE_CHG_OI"].sum())

        # RSI bias
        rsi_bias = 0
        if latest_rsi_val is not None:
            if latest_rsi_val > 70:
                rsi_bias = -1
            elif latest_rsi_val < 30:
                rsi_bias = 1
            elif latest_rsi_val > 55:
                rsi_bias = -0.5
            elif latest_rsi_val < 45:
                rsi_bias = 0.5

        # OI bias: CE accumulation => bullish
        oi_bias = 0
        if (total_ce_chg - total_pe_chg) > max(200, 0.05 * (abs(total_ce_chg) + abs(total_pe_chg) + 1)):
            oi_bias = 1
        elif (total_pe_chg - total_ce_chg) > max(200, 0.05 * (abs(total_ce_chg) + abs(total_pe_chg) + 1)):
            oi_bias = -1

        combined_score = rsi_bias + oi_bias
        if combined_score >= 1.5:
            suggestion = "📈 Strong Bullish"
        elif combined_score >= 0.5:
            suggestion = "🔼 Mild Bullish"
        elif combined_score <= -1.5:
            suggestion = "📉 Strong Bearish"
        elif combined_score <= -0.5:
            suggestion = "🔽 Mild Bearish"
        else:
            suggestion = "⚖ Neutral / Sideways"

        st.markdown(f"- **Latest RSI (synthetic):** {latest_rsi_val:.2f}" if latest_rsi_val is not None else "- **Latest RSI (synthetic):** N/A")
        st.markdown(f"- **Total CE ΔOI (window):** {total_ce_chg:,}")
        st.markdown(f"- **Total PE ΔOI (window):** {total_pe_chg:,}")
        st.success(f"Momentum suggestion: **{suggestion}**")

        # CSV Download & footer
        st.markdown("---")
        download_df = df_window.copy()
        download_df["fetched_at_utc"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        csv = download_df.to_csv(index=False)
        st.download_button("Download displayed CSV", csv, file_name=f"{index_choice}_option_chain_{int(time.time())}.csv", mime="text/csv")

        st.caption(f"Data source: NSE (nearest expiry). Last fetch: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC. Data refresh: {REFRESH_SECONDS}s")

    # Wait before next data update (only updates the placeholder)
    time.sleep(REFRESH_SECONDS)

    # If the page was unloaded (user navigated away) the loop will be stopped by Streamlit.
    # Also break after many iterations as a safety (optional)
    if run_count >= 1000000:
        break
