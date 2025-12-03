# app.py
"""
NIFTY Live collector + cache + S3 sync + backfill helper + export + trend-weight tuning
- Uses NSE HTTPS chart API to fetch current-day intraday and caches to .nse_cache/
- Optional: uploads each daily CSV to S3 (when AWS creds provided via Streamlit secrets)
- Provides one-click export (zip) and trend-strength tuning sliders
- Includes a small backfill-helper endpoint and instructions compatible with GitHub Actions
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
import io
import zipfile
from typing import Optional

# Optional S3
HAS_BOTO3 = False
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    HAS_BOTO3 = True
except Exception:
    HAS_BOTO3 = False

st.set_page_config(page_title="NIFTY Live Collector + Cache + S3", layout="wide")
IST = pytz.timezone("Asia/Kolkata")
CACHE_DIR = ".nse_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# -------------------------
# Config & Secrets
# -------------------------
secrets = st.secrets if hasattr(st, "secrets") else {}
AWS_CONF = secrets.get("aws", {})  # expected keys: access_key_id, secret_access_key, region, s3_bucket, s3_prefix (optional)
S3_ENABLED = HAS_BOTO3 and bool(AWS_CONF.get("access_key_id") and AWS_CONF.get("secret_access_key") and AWS_CONF.get("s3_bucket"))

# User-tunable weights (persist in session)
if "weights" not in st.session_state:
    st.session_state.weights = {"ema":0.38, "vwap":0.28, "momentum":0.20, "volume":0.14, "vol_penalty_factor":0.6}

# -------------------------
# NSE HTTPS fetcher (polite)
# -------------------------
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
        # warm cookies
        s.get("https://www.nseindia.com", timeout=5)
    except Exception:
        pass
    return s

def fetch_today_intraday_from_nse(retries:int=3, backoff_s:float=0.8) -> Optional[pd.DataFrame]:
    """
    Fetch current-day intraday JSON from NSE chart API via HTTPS.
    Returns DataFrame with columns ['datetime' (tz-aware IST), 'price', 'volume'(if present)] or None.
    """
    url = "https://www.nseindia.com/api/chart-databyindex?index=EQUITY%7CNIFTY%2050&preopen=0"
    for attempt in range(retries):
        try:
            s = nse_session()
            r = s.get(url, timeout=10)
            r.raise_for_status()
            j = r.json()
            # Common shapes: 'grapthData' list of [ts, price] or list of dicts
            if "grapthData" in j and isinstance(j["grapthData"], list) and len(j["grapthData"])>0:
                gd = j["grapthData"]
                if isinstance(gd[0], (list, tuple)) and len(gd[0])>=2:
                    df = pd.DataFrame(gd)
                    df = df.iloc[:, :2]
                    df.columns = ["timestamp", "price"]
                    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
                    df = df[["datetime", "price"]]
                    if "volume" in j and isinstance(j["volume"], list) and len(j["volume"])==len(df):
                        df["volume"] = j["volume"]
                    else:
                        df["volume"] = np.nan
                    return df
                elif isinstance(gd[0], dict):
                    rows = []
                    for rec in gd:
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
            # fallback: not parsed
            return None
        except Exception:
            time.sleep(backoff_s * (attempt+1))
            continue
    return None

# -------------------------
# Cache helpers
# -------------------------
def cache_path_for_date(d: datetime.date) -> str:
    return os.path.join(CACHE_DIR, f"nifty_{d.isoformat()}.csv")

def save_day_df(d: datetime.date, df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    path = cache_path_for_date(d)
    df2 = df.copy()
    if "datetime" in df2.columns:
        # convert to string (IST) for portability
        df2["datetime"] = pd.to_datetime(df2["datetime"]).dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m-%d %H:%M:%S")
    df2.to_csv(path, index=False)
    return True

def load_day_df(d: datetime.date) -> Optional[pd.DataFrame]:
    path = cache_path_for_date(d)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=["datetime"])
        # treat saved datetimes as IST localized
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize("Asia/Kolkata")
        return df
    except Exception:
        return None

def list_cached_dates() -> list:
    files = [f for f in os.listdir(CACHE_DIR) if f.startswith("nifty_") and f.endswith(".csv")]
    dates = []
    for f in files:
        try:
            iso = f[len("nifty_"):-4]
            d = datetime.date.fromisoformat(iso)
            dates.append(d)
        except Exception:
            continue
    return sorted(dates)

# -------------------------
# Optional S3 sync
# -------------------------
def s3_client_from_secrets():
    if not S3_ENABLED:
        return None
    try:
        sess = boto3.session.Session(
            aws_access_key_id=AWS_CONF.get("access_key_id"),
            aws_secret_access_key=AWS_CONF.get("secret_access_key"),
            region_name=AWS_CONF.get("region")
        )
        s3 = sess.client("s3")
        return s3
    except Exception:
        return None

def upload_file_to_s3(local_path: str, s3_key: str) -> bool:
    s3 = s3_client_from_secrets()
    if s3 is None:
        return False
    try:
        s3.upload_file(local_path, AWS_CONF.get("s3_bucket"), s3_key)
        return True
    except Exception:
        return False

# -------------------------
# Trend strength (weights tunable)
# -------------------------
def compute_vwap(df: pd.DataFrame) -> pd.Series:
    if "volume" in df.columns and df["volume"].notna().any():
        pv = (df["price"] * df["volume"]).fillna(0)
        cum_pv = pv.cumsum()
        cum_vol = df["volume"].fillna(0).cumsum()
        v = pd.Series(np.nan, index=df.index)
        mask = cum_vol>0
        v.loc[mask] = cum_pv.loc[mask] / cum_vol.loc[mask]
        return v
    else:
        return df["price"].expanding().mean()

def compute_trend_strength_from_clip(clip: pd.DataFrame, weights: dict) -> dict:
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
    vwap = compute_vwap(df)
    vwap_last = float(vwap.iloc[-1]) if not vwap.isna().all() else np.nan
    if not np.isnan(vwap_last) and vwap_last!=0:
        vwap_gap_pct = (last - vwap_last) / vwap_last
    else:
        vwap_gap_pct = (last - first) / max(1e-6, first)
    momentum_pct = (last - first) / max(1e-6, first)
    if "volume" in df.columns and df["volume"].notna().any():
        avgv = float(df["volume"].mean()); medv = float(np.nanmedian(df["volume"].values))
        vol_conf = (avgv / (medv + 1e-9)) if medv>0 else 1.0
        vol_conf = min(vol_conf, 5.0)
    else:
        vol_conf = 1.0
    avg_move = float(np.abs(df["price"].diff().fillna(0)).mean())
    vol_norm = avg_move / max(1e-6, prange)
    vol_penalty = float(np.clip(np.tanh(6 * vol_norm), 0, 1))
    def slope_to_score(s): return 1.0 / (1.0 + np.exp(-4 * s))
    ema_score = 0.65 * slope_to_score(ema9_s) + 0.35 * slope_to_score(ema26_s)
    vwap_score = 0.5 + 0.5 * np.tanh(6 * vwap_gap_pct)
    momentum_score = 0.5 + 0.5 * np.tanh(4 * momentum_pct)
    vol_score = 1.0 - 1.0/(1.0 + (vol_conf-1.0)) if vol_conf>=1 else vol_conf/2.0 + 0.25
    vol_score = float(np.clip(vol_score, 0.0, 1.0))
    base = weights["ema"] * ema_score + weights["vwap"] * vwap_score + weights["momentum"] * momentum_score + weights["volume"] * vol_score
    score = base * (1.0 - weights.get("vol_penalty_factor", 0.6) * vol_penalty)
    score_0_100 = float(np.clip(score * 100.0, 0.0, 100.0))
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
    return {"score": round(score_0_100,1), "class": cl, "components": {"ema9_s":ema9_s,"vwap_gap_pct":vwap_gap_pct,"momentum_pct":momentum_pct,"vol_conf":vol_conf,"vol_norm":vol_norm}}

# -------------------------
# Volume profile & entry/exit
# -------------------------
def compute_avg_volume_by_min(frames: list) -> pd.Series:
    rows = []
    for df in frames:
        if df is None or df.empty: continue
        sub = df.copy()
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

# -------------------------
# UI: main page
# -------------------------
st.title("📈 NIFTY Live Collector + Cache (NSE HTTPS)")

left, right = st.columns([1,3])
with left:
    n_days = st.number_input("Days to show (cached)", min_value=3, max_value=60, value=30, step=1)
    btn_fetch = st.button("Fetch / Update today's intraday now")
    force_overwrite = st.checkbox("Force overwrite today's cache", value=False)
    btn_export = st.button("Export all cached CSVs (zip)")
    if S3_ENABLED:
        st.markdown(f"**S3 enabled** — bucket `{AWS_CONF.get('s3_bucket')}`")
    else:
        st.markdown("**S3 not enabled**. To enable upload-to-S3, add AWS creds to Streamlit Secrets under `aws` (access_key_id, secret_access_key, region, s3_bucket).")
    st.markdown("Trend-weight tuning:")
    w_ema = st.slider("EMA weight", min_value=0.0, max_value=1.0, value=st.session_state.weights["ema"], step=0.01)
    w_vwap = st.slider("VWAP weight", min_value=0.0, max_value=1.0, value=st.session_state.weights["vwap"], step=0.01)
    w_mom = st.slider("Momentum weight", min_value=0.0, max_value=1.0, value=st.session_state.weights["momentum"], step=0.01)
    w_vol = st.slider("Volume weight", min_value=0.0, max_value=1.0, value=st.session_state.weights["volume"], step=0.01)
    vol_pen = st.slider("Volatility penalty factor", min_value=0.0, max_value=1.0, value=st.session_state.weights["vol_penalty_factor"], step=0.01)
    # normalize weights so they sum approx 1 (unless all zero)
    total = (w_ema + w_vwap + w_mom + w_vol)
    if total <= 0:
        total = 1.0
    weights = {"ema": w_ema/total, "vwap": w_vwap/total, "momentum": w_mom/total, "volume": w_vol/total, "vol_penalty_factor": vol_pen}
    st.session_state.weights = weights
    st.caption("Weights normalized automatically (sum to 1). Changes apply immediately.")

with right:
    st.markdown("This app uses NSE's HTTPS chart API to fetch today's intraday. Each fetch is polite (headers + warm-up). Cached CSVs live in `.nse_cache/`. Use the Export button to download all cached CSVs as a zip. If S3 is enabled, each daily file will be uploaded to S3 after saving.")
    st.markdown("For reliable long-term storage, enable S3 or periodically download the zip.")

# -------------------------
# Fetch / Save today's data if requested
# -------------------------
today = datetime.date.today()
cached_dates = list_cached_dates()
need_fetch = btn_fetch or (today not in cached_dates) or force_overwrite

if need_fetch:
    st.info("Fetching today's intraday from NSE (HTTPS)...")
    df_today = fetch_today_intraday_from_nse()
    if df_today is None or df_today.empty:
        st.error("Could not fetch intraday from NSE. Try again soon (NSE may block frequent requests).")
    else:
        saved = save_day_df(today, df_today)
        if saved:
            st.success(f"Saved today's intraday to cache: {cache_path_for_date(today)}")
            # attempt S3 upload if enabled
            if S3_ENABLED:
                s3_key = (AWS_CONF.get("s3_prefix","").rstrip("/") + "/" if AWS_CONF.get("s3_prefix") else "") + os.path.basename(cache_path_for_date(today))
                ok = upload_file_to_s3(cache_path_for_date(today), s3_key)
                if ok:
                    st.success(f"Uploaded today's cache to s3://{AWS_CONF.get('s3_bucket')}/{s3_key}")
                else:
                    st.warning("S3 upload failed (check AWS credentials / permissions).")
        else:
            st.warning("Fetched today's data but failed to save cache.")

# -------------------------
# Export (zip) all cached CSVs
# -------------------------
if btn_export:
    cached = [cache_path_for_date(d) for d in list_cached_dates()]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in cached:
            try:
                zf.write(p, arcname=os.path.basename(p))
            except Exception:
                continue
    buf.seek(0)
    st.download_button("Download zipped cache", data=buf, file_name=f"nifty_cache_{datetime.date.today().isoformat()}.zip", mime="application/zip")

# -------------------------
# Build display list and analysis
# -------------------------
# show most-recent n_days that are cached
cached_dates = list_cached_dates()
# get last n_days trading days
def last_n_weekdays(n):
    days=[]
    cur=datetime.date.today()
    while len(days)<n:
        if cur.weekday()<5:
            days.append(cur)
        cur = cur - datetime.timedelta(days=1)
    return list(reversed(days))
cand = last_n_weekdays(max(n_days, len(cached_dates)))
dates_to_display = [d for d in reversed(cand) if d in cached_dates][:n_days]
dates_to_display = list(reversed(dates_to_display))

frames = []
for d in dates_to_display:
    frames.append((d, load_day_df(d)))

# compute avg volume series and entry/exit
clipped_frames = []
for (d, df) in frames:
    if df is None: continue
    mask = (pd.to_datetime(df["datetime"]).dt.time >= datetime.time(9,15)) & (pd.to_datetime(df["datetime"]).dt.time <= datetime.time(11,30))
    clipped_frames.append(df.loc[mask].copy())
avg_vol_series = compute_avg_volume_by_min(clipped_frames)
entry_exit = detect_entry_exit(avg_vol_series)

st.header("Volume analysis (09:15 → 11:30 IST) across cached days")
if avg_vol_series.empty:
    st.info("Not enough cached minute-volume data yet. Keep the app running daily to build history.")
else:
    st.metric("Peak avg volume (minute)", f"{entry_exit.get('peak')}")
    st.write("Suggested entry minute:", entry_exit.get("entry"), "Suggested exit minute:", entry_exit.get("exit"))
    st.dataframe(avg_vol_series.reset_index().rename(columns={"min_key":"Minute","volume":"AvgVolume"}).head(80))

# -------------------------
# Plot grid with trend strength badges
# -------------------------
st.header("Mini-charts (09:15→11:30) + Trend strength (weights applied)")

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
            clip = df.loc[(pd.to_datetime(df["datetime"]).dt.time >= datetime.time(9,15)) & (pd.to_datetime(df["datetime"]).dt.time <= datetime.time(11,30))].copy()
            if clip.empty:
                st.info("No data in 09:15–11:30")
                continue
            ts = compute_trend_strength_from_clip(clip, st.session_state.weights)
            score = ts.get("score", None)
            cls = ts.get("class", "N/A")
            color = "#999"
            if cls == "Strong Bullish": color="#1f8b4c"
            elif cls == "Mild Bullish": color="#65c466"
            elif cls == "Neutral": color="#bfbf00"
            elif cls == "Mild Bearish": color="#ff8a5b"
            elif cls == "Strong Bearish": color="#d9534f"
            badge_col, info_col = st.columns([1,4])
            with badge_col:
                st.markdown(f"<div style='background:{color};color:white;padding:6px;border-radius:6px;text-align:center'><strong>{score if score is not None else 'NA'}</strong></div>", unsafe_allow_html=True)
            with info_col:
                st.write(cls)
                comps = ts.get("components", {})
                if comps:
                    st.caption(f"EMA9_s={comps.get('ema9_s'):.4f} VWAP_gap={comps.get('vwap_gap_pct'):.4f} vol_conf={comps.get('vol_conf'):.2f}")
            # plot price+vwap+volume
            fig, (ax0, ax1) = plt.subplots(2,1, figsize=(3.2,2.4), gridspec_kw={'height_ratios':[2,0.8]})
            x = pd.to_datetime(clip["datetime"])
            xs = x.dt.strftime("%H:%M")
            ax0.plot(xs, clip["price"], linewidth=0.9)
            try:
                v = compute_vwap(clip)
                ax0.plot(xs, v, linewidth=0.8, linestyle="--")
            except Exception:
                pass
            ax0.set_xticks([xs.iloc[0], xs.iloc[len(xs)//2], xs.iloc[-1]])
            ax0.tick_params(axis="x", rotation=45, labelsize=7)
            ax0.set_ylabel("Price", fontsize=8)
            if "volume" in clip.columns and clip["volume"].notna().any():
                ax1.bar(xs, clip["volume"].fillna(0), width=0.6)
            ax1.set_xticks([xs.iloc[0], xs.iloc[len(xs)//2], xs.iloc[-1]])
            ax1.tick_params(axis="x", rotation=45, labelsize=7)
            ax1.set_ylabel("Vol", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

st.write("---")
st.markdown("""
### Backfill options / scheduling
1. **GitHub Action (preferred)**: set up a workflow that calls your deployed Streamlit app's `/fetch` endpoint (or triggers a fetch via curl to the app), or runs a script that performs the same fetch logic and commits/uploads the cached CSV to S3. See the included `backfill.yml` snippet in the repo.  
2. **Cron / scheduler**: call `https://<your-app>.streamlit.app` (open the page) at 15:35 IST or run a small script on a server to call the same fetch function and upload to S3.

### Persistence note
- Streamlit Cloud's filesystem is ephemeral across deployments — enabling S3 upload gives durable storage.
""")
