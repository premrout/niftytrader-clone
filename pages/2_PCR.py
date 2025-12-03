# app.py
"""
Streamlit app: Collect live intraday from NSE each day and cache it.
- Fetches current day's intraday candles from NSE chart API (the same JSON used by the NSE site).
- Saves per-day CSVs to .nse_cache/ (nifty_YYYY-MM-DD.csv).
- Shows the last N cached days (price + volume) clipped to 09:15-11:30 IST.
- Computes trend-strength per day and a simple entry/exit-by-volume heuristic.
- Auto-refresh hint: the app requests a refresh at next 09:00 and 17:00 IST if you keep it open.
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import datetime
import pytz
import time
import math
import matplotlib.pyplot as plt
from typing import Optional

st.set_page_config(page_title="NIFTY Live Collector + Cache", layout="wide")
IST = pytz.timezone("Asia/Kolkata")
CACHE_DIR = ".nse_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# -----------------------
# NSE session (polite)
# -----------------------
def nse_session():
    s = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive",
    }
    s.headers.update(headers)
    try:
        # warm up to get cookies
        s.get("https://www.nseindia.com", timeout=5)
    except Exception:
        # ignore; the next request may still succeed
        pass
    return s

# -----------------------
# Fetch today's intraday chart JSON from NSE
# -----------------------
def fetch_today_intraday_from_nse(retries: int = 3, backoff_s: float = 0.8) -> Optional[pd.DataFrame]:
    """
    Returns DataFrame with columns ['datetime' (IST), 'price', 'volume' (if present)]
    or None on failure.
    """
    url = "https://www.nseindia.com/api/chart-databyindex?index=EQUITY%7CNIFTY%2050&preopen=0"
    for attempt in range(retries):
        try:
            s = nse_session()
            r = s.get(url, timeout=8)
            r.raise_for_status()
            j = r.json()
            # Different NSE JSON shapes exist. Handle common forms.
            # Case A: j['grapthData'] = list of [timestamp(ms), price]
            if "grapthData" in j and isinstance(j["grapthData"], list) and len(j["grapthData"])>0:
                gd = j["grapthData"]
                # element could be list or dict
                if isinstance(gd[0], (list, tuple)) and len(gd[0]) >= 2:
                    df = pd.DataFrame(gd)
                    # normalize columns
                    if df.shape[1] >= 2:
                        df = df.iloc[:, :2]
                        df.columns = ["timestamp", "price"]
                    # convert to datetime IST
                    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
                    df = df[["datetime", "price"]]
                    # volume may be at j['volume'] parallel list
                    if "volume" in j and isinstance(j["volume"], list) and len(j["volume"]) == len(df):
                        df["volume"] = j["volume"]
                    else:
                        df["volume"] = np.nan
                    return df
                elif isinstance(gd[0], dict):
                    rows = []
                    for rec in gd:
                        # rec may hold x/time and y/price and v/volume
                        ts = rec.get("x") or rec.get("timestamp") or rec.get("time")
                        price = rec.get("y") or rec.get("price") or rec.get("close")
                        vol = rec.get("v") or rec.get("volume")
                        if ts is None or price is None:
                            continue
                        try:
                            ts_int = int(ts)
                            dt = pd.to_datetime(ts_int, unit="ms").tz_localize("UTC").tz_convert("Asia/Kolkata")
                        except Exception:
                            try:
                                dt = pd.to_datetime(ts).tz_localize("UTC").tz_convert("Asia/Kolkata")
                            except Exception:
                                continue
                        rows.append({"datetime": dt, "price": price, "volume": vol})
                    if rows:
                        df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
                        if "volume" not in df.columns:
                            df["volume"] = np.nan
                        return df
            # fallback: some responses include 'data' or other shapes - attempt generic parsing:
            # Try to find lists of equal length for timestamps & prices
            cand_lists = []
            def collect_lists(obj):
                if isinstance(obj, dict):
                    for v in obj.values():
                        collect_lists(v)
                elif isinstance(obj, list):
                    if len(obj)>0 and (isinstance(obj[0], (int, float, str))):
                        cand_lists.append(obj)
                    else:
                        for item in obj:
                            collect_lists(item)
            collect_lists(j)
            # try to pair two lists of equal length: timestamps & prices
            for a in cand_lists:
                for b in cand_lists:
                    if len(a)==len(b) and a is not b:
                        # guess that one is timestamps (large ints) and other is price (floats)
                        # pick pair where one has large values >1e11 (ms epoch)
                        if any(isinstance(x,int) and x>1e11 for x in a):
                            try:
                                df = pd.DataFrame({"timestamp": a, "price": b})
                                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
                                df = df[["datetime","price"]]
                                df["volume"] = np.nan
                                return df
                            except Exception:
                                pass
            # not parsed
            return None
        except Exception:
            time.sleep(backoff_s * (attempt+1))
            continue
    return None

# -----------------------
# Cache helpers
# -----------------------
def cache_path_for_date(d: datetime.date) -> str:
    return os.path.join(CACHE_DIR, f"nifty_{d.isoformat()}.csv")

def save_day_df(d: datetime.date, df: pd.DataFrame):
    if df is None or df.empty:
        return False
    path = cache_path_for_date(d)
    # convert datetime to ISO without tz for portability
    df2 = df.copy()
    if "datetime" in df2.columns:
        df2["datetime"] = pd.to_datetime(df2["datetime"]).dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m-%d %H:%M:%S")
    df2.to_csv(path, index=False)
    return True

def load_cached_dates() -> list:
    files = [f for f in os.listdir(CACHE_DIR) if f.startswith("nifty_") and f.endswith(".csv")]
    dates = []
    for f in files:
        try:
            iso = f[len("nifty_"):-4]
            d = datetime.datetime.fromisoformat(iso).date()
            dates.append(d)
        except Exception:
            continue
    return sorted(dates)

def load_day_df(d: datetime.date) -> Optional[pd.DataFrame]:
    path = cache_path_for_date(d)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=["datetime"])
        # mark tz as IST
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None).dt.tz_localize("Asia/Kolkata")
        return df
    except Exception:
        return None

# -----------------------
# Trend strength (compact)
# -----------------------
def compute_vwap(df: pd.DataFrame) -> pd.Series:
    if "volume" in df.columns and df["volume"].notna().any():
        pv = (df["price"] * df["volume"]).fillna(0)
        cum_pv = pv.cumsum()
        cum_vol = df["volume"].fillna(0).cumsum()
        v = pd.Series(np.nan, index=df.index)
        mask = cum_vol > 0
        v.loc[mask] = cum_pv.loc[mask] / cum_vol.loc[mask]
        return v
    else:
        return df["price"].expanding().mean()

def compute_trend_strength_from_clip(clip: pd.DataFrame) -> dict:
    # returns {'score':float, 'class':str, 'components': {...}}
    if clip is None or clip.empty:
        return {}
    df = clip.copy().reset_index(drop=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    if df.empty:
        return {}
    first = float(df["price"].iloc[0]); last = float(df["price"].iloc[-1])
    prange = max(1e-6, df["price"].max() - df["price"].min())
    ema9 = df["price"].ewm(span=9, adjust=False).mean()
    ema26 = df["price"].ewm(span=26, adjust=False).mean()
    ema9_s = (ema9.iloc[-1] - ema9.iloc[0]) / prange
    ema26_s = (ema26.iloc[-1] - ema26.iloc[0]) / prange
    vwap = compute_vwap(df); vwap_last = float(vwap.iloc[-1]) if not vwap.isna().all() else np.nan
    vwap_gap_pct = (last - vwap_last) / vwap_last if (not np.isnan(vwap_last) and vwap_last!=0) else (last-first)/max(1e-6, first)
    momentum_pct = (last - first)/max(1e-6, first)
    # vol confirmation
    if "volume" in df.columns and df["volume"].notna().any():
        avgv = float(df["volume"].mean()); medv = float(np.nanmedian(df["volume"].values))
        vol_conf = (avgv / (medv + 1e-9)) if medv>0 else 1.0
        vol_conf = min(vol_conf, 5.0)
    else:
        vol_conf = 1.0
    # volatility penalty
    avg_move = float(np.abs(df["price"].diff().fillna(0)).mean())
    vol_norm = avg_move / max(1e-6, prange)
    vol_penalty = float(np.clip(np.tanh(6*vol_norm), 0, 1))
    # scoring transforms
    def slope_to_score(s): return 1.0/(1.0+np.exp(-4*s))
    ema_score = 0.65*slope_to_score(ema9_s) + 0.35*slope_to_score(ema26_s)
    vwap_score = 0.5 + 0.5*np.tanh(6*vwap_gap_pct)
    momentum_score = 0.5 + 0.5*np.tanh(4*momentum_pct)
    vol_score = 1.0 - 1.0/(1.0 + (vol_conf-1.0)) if vol_conf>=1 else vol_conf/2.0 + 0.25
    vol_score = float(np.clip(vol_score,0,1))
    base = 0.38*ema_score + 0.28*vwap_score + 0.20*momentum_score + 0.14*vol_score
    score = base * (1.0 - 0.6*vol_penalty)
    score_0_100 = float(np.clip(score*100.0, 0.0, 100.0))
    if score_0_100 >= 75:
        cl = "Strong Bullish"
    elif score_0_100 >= 60:
        cl = "Mild Bullish"
    elif score_0_100 >= 45:
        cl = "Neutral"
    elif score_0_100 >= 30:
        cl = "Mild Bearish"
    else:
        cl = "Strong Bearish"
    return {
        "score": round(score_0_100,1),
        "class": cl,
        "components": {
            "ema9_s": ema9_s, "ema26_s": ema26_s,
            "vwap_gap_pct": vwap_gap_pct, "momentum_pct": momentum_pct,
            "vol_conf": vol_conf, "vol_norm": vol_norm
        }
    }

# -----------------------
# Volume profile & entry/exit detection
# -----------------------
def compute_avg_volume_by_min(frames: list) -> pd.Series:
    rows = []
    for df in frames:
        if df is None or df.empty: continue
        sub = df.copy()
        # minute key in IST HH:MM
        sub["min_key"] = pd.to_datetime(sub["datetime"]).dt.strftime("%H:%M")
        if "volume" not in sub.columns:
            continue
        rows.append(sub[["min_key","volume"]])
    if not rows:
        return pd.Series(dtype=float)
    big = pd.concat(rows, ignore_index=True)
    avg = big.groupby("min_key")["volume"].mean().sort_index()
    return avg

def detect_entry_exit(avg_series: pd.Series):
    if avg_series.empty: return {}
    thr = np.percentile(avg_series.values, 90)
    high_mins = avg_series[avg_series>=thr].index.tolist()
    entry = high_mins[0] if high_mins else avg_series.idxmax()
    peak = avg_series.max()
    taper = 0.4*peak
    mins = list(avg_series.index)
    try:
        pi = mins.index(entry)
    except ValueError:
        pi = mins.index(avg_series.idxmax())
    exit_min = None
    for i in range(pi+1, len(mins)):
        if avg_series.iloc[i] <= taper:
            exit_min = mins[i]; break
    if exit_min is None:
        exit_min = mins[-1]
    return {"entry": entry, "exit": exit_min, "peak": int(peak)}

# -----------------------
# Auto-refresh scheduling helper (client-side)
# -----------------------
def seconds_until_next_target(hours=(9,17)):
    now = datetime.datetime.now(IST)
    today = now.date()
    candidates = []
    for h in hours:
        target = IST.localize(datetime.datetime.combine(today, datetime.time(h,0,0)))
        if target <= now:
            target = target + datetime.timedelta(days=1)
        candidates.append(target)
    return int((min(candidates) - now).total_seconds())

# -----------------------
# App UI & main flow
# -----------------------
st.title("NIFTY — Live collector & cache (NSE) — builds history over time")

left, right = st.columns([1,3])
with left:
    n_days = st.number_input("Days to show (cached)", min_value=3, max_value=60, value=30, step=1)
    refresh_now = st.button("Fetch / Update today's intraday now")
    force_overwrite = st.checkbox("Force overwrite today's cache (if exists)", value=False)
    show_next = st.checkbox("Show next client auto-refresh", value=True)
with right:
    st.markdown("This app fetches today's intraday from NSE and saves it to `.nse_cache/` as `nifty_YYYY-MM-DD.csv`. "
                "Open this app daily or keep it open (it will auto-refresh at 09:00 and 17:00 IST while open).")

# client-side auto-refresh: use a small autorefresh (requires streamlit-autorefresh)
try:
    from streamlit_autorefresh import st_autorefresh
    sec_to_next = seconds_until_next_target((9,17))
    interval = max(5, min(sec_to_next, 24*3600))
    st_autorefresh(interval=interval*1000, key="nse_autorefresh")
except Exception:
    # if not installed, ignore
    pass

if show_next:
    sec_next = seconds_until_next_target((9,17))
    t_next = datetime.datetime.now(IST) + datetime.timedelta(seconds=sec_next)
    st.write("Next scheduled refresh (approx):", t_next.strftime("%Y-%m-%d %H:%M:%S %Z"))

# Step 1: load cached dates
cached_dates = load_cached_dates()
st.write(f"Cached days available: {len(cached_dates)} (latest: {cached_dates[-1] if cached_dates else 'none'})")

# Step 2: fetch today's intraday if requested or if today's cache missing
today = datetime.date.today()
today_path = cache_path_for_date(today)
need_fetch = refresh_now or (today not in cached_dates) or force_overwrite

if need_fetch:
    st.info("Fetching today's intraday from NSE...")
    df_today = fetch_today_intraday_from_nse()
    if df_today is None or df_today.empty:
        st.error("Could not fetch intraday from NSE. Possible reasons: temporary block, network issue, or API shape changed. Try again in a minute.")
    else:
        saved = save_day_df(today, df_today)
        if saved:
            st.success(f"Today's intraday saved to cache ({cache_path_for_date(today)})")
        else:
            st.warning("Fetched today's data but could not save to cache.")
    # reload cached list
    cached_dates = load_cached_dates()

# Step 3: prepare list of dates to display: intersection of cached_dates and last n trading days
def last_n_weekdays(n):
    days = []
    cur = datetime.date.today()
    while len(days) < n:
        if cur.weekday() < 5:
            days.append(cur)
        cur = cur - datetime.timedelta(days=1)
    return list(reversed(days))

candidate = last_n_weekdays(max(n_days, len(cached_dates)))
# We'll display up to n_days of cached entries (most recent first)
dates_to_display = [d for d in reversed(candidate) if d in cached_dates][:n_days]
dates_to_display = list(reversed(dates_to_display))  # oldest->newest for consistent grid

# Step 4: load frames and compute analysis
frames = []
for d in dates_to_display:
    df = load_day_df(d)
    frames.append((d, df))

# compute avg volume across available frames (09:15-11:30)
clipped_frames = []
for (d, df) in frames:
    if df is None: continue
    mask = (pd.to_datetime(df["datetime"]).dt.time >= datetime.time(9,15)) & (pd.to_datetime(df["datetime"]).dt.time <= datetime.time(11,30))
    clipped = df.loc[mask].copy()
    clipped_frames.append(clipped)

avg_vol_series = compute_avg_volume_by_min(clipped_frames)
entry_exit = detect_entry_exit(avg_vol_series)

# Display analysis
st.header("Volume analysis (09:15 → 11:30 IST) across cached days")
if avg_vol_series.empty:
    st.info("Not enough cached minute-volume data yet. Keep the app running daily to build history.")
else:
    st.metric("Peak avg volume (minute)", f"{entry_exit.get('peak')}")
    st.write("Suggested entry minute:", entry_exit.get("entry"), "Suggested exit minute:", entry_exit.get("exit"))
    st.dataframe(avg_vol_series.reset_index().rename(columns={"min_key":"Minute","volume":"AvgVolume"}).head(50))

# Plot grid with trend strength badges
st.header("Mini-charts (09:15→11:30) + Trend strength")
cols_per_row = 5
rows = max(1, math.ceil(len(frames)/cols_per_row))
for r in range(rows):
    cols = st.columns(cols_per_row)
    for c in range(cols_per_row):
        idx = r*cols_per_row + c
        if idx >= len(frames):
            continue
        d, df = frames[idx]
        with cols[c]:
            st.subheader(d.strftime("%Y-%m-%d"))
            if df is None or df.empty:
                st.info("No cached intraday for this day")
                continue
            # clip
            df_clip = df.loc[(pd.to_datetime(df["datetime"]).dt.time >= datetime.time(9,15)) & (pd.to_datetime(df["datetime"]).dt.time <= datetime.time(11,30))].copy()
            if df_clip.empty:
                st.info("No data in 09:15–11:30")
                continue
            ts = compute_trend_strength_from_clip(df_clip)
            # badge
            score = ts.get("score", None)
            cls = ts.get("class", "N/A")
            color = "#999"
            if cls == "Strong Bullish": color="#1f8b4c"
            elif cls == "Mild Bullish": color="#65c466"
            elif cls == "Neutral": color="#bfbf00"
            elif cls == "Mild Bearish": color="#ff8a5b"
            elif cls == "Strong Bearish": color="#d9534f"
            leftc, rightc = st.columns([1,4])
            with leftc:
                st.markdown(f"<div style='background:{color};color:white;padding:6px;border-radius:6px;text-align:center'><strong>{score if score is not None else 'NA'}</strong></div>", unsafe_allow_html=True)
            with rightc:
                st.write(cls)
                comps = ts.get("components", {})
                if comps:
                    st.caption(f"ema9_s={comps.get('ema9_s'):.4f} vwap_gap={comps.get('vwap_gap_pct'):.4f} vol_conf={comps.get('vol_conf'):.2f}")
            # plot
            fig, (ax0, ax1) = plt.subplots(2,1, figsize=(3.2,2.4), gridspec_kw={'height_ratios':[2,0.8]})
            x = pd.to_datetime(df_clip["datetime"])
            xs = x.dt.strftime("%H:%M")
            ax0.plot(xs, df_clip["price"], linewidth=0.9)
            try:
                v = compute_vwap(df_clip)
                ax0.plot(xs, v, linewidth=0.8, linestyle="--")
            except Exception:
                pass
            ax0.set_xticks([xs.iloc[0], xs.iloc[len(xs)//2], xs.iloc[-1]])
            ax0.tick_params(axis="x", rotation=45, labelsize=7)
            ax0.set_ylabel("Price", fontsize=8)
            if "volume" in df_clip.columns and df_clip["volume"].notna().any():
                ax1.bar(xs, df_clip["volume"].fillna(0), width=0.6)
            ax1.set_xticks([xs.iloc[0], xs.iloc[len(xs)//2], xs.iloc[-1]])
            ax1.tick_params(axis="x", rotation=45, labelsize=7)
            ax1.set_ylabel("Vol", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

st.write("---")
st.markdown("""
**Persistence recommendation:**  
- For reliable long-term storage you should periodically copy `.nse_cache/` to durable storage: S3, Google Drive, or commit to a private GitHub repo (for small history).  
- If you want I can add optional S3 sync code (requires AWS creds) or Google Drive upload.
""")
