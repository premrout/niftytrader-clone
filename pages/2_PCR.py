# app.py
"""
Streamlit app: Fetch daily report file(s) from NSE All Reports page (https HTTPS).
- Finds file links on https://www.nseindia.com/all-reports
- Downloads today's report (or latest if today's not present)
- Caches each downloaded file to .nse_reports/
- Optionally auto-refreshes once a day at a chosen time (client-side)
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
import datetime
import time
import pytz
import pathlib
import re
import io
import pandas as pd

# --- Config
st.set_page_config(page_title="NSE — Daily report fetcher (All Reports)", layout="wide")
CACHE_DIR = ".nse_reports"
os.makedirs(CACHE_DIR, exist_ok=True)
IST = pytz.timezone("Asia/Kolkata")
ALL_REPORTS_URL = "https://www.nseindia.com/all-reports"
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]

# Helper: polite session
def make_nse_session(user_agent=None):
    s = requests.Session()
    ua = user_agent or USER_AGENTS[0]
    headers = {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive",
    }
    s.headers.update(headers)
    try:
        # warm up cookies
        s.get("https://www.nseindia.com", timeout=6)
    except Exception:
        # warming may fail sometimes; it's okay
        pass
    return s

# Robust fetch of the All Reports page + parse links
def fetch_all_reports_page(retries=3, backoff=1.0):
    for attempt in range(retries):
        try:
            s = make_nse_session()
            r = s.get(ALL_REPORTS_URL, timeout=12)
            # if not HTML or status != 200, record debug and return None
            if r.status_code != 200 or not r.headers.get("Content-Type","").lower().startswith("text/html"):
                # save debug snippet
                debug_path = os.path.join(CACHE_DIR, "nse_debug.txt")
                with open(debug_path, "a", encoding="utf-8") as fh:
                    fh.write(f"{datetime.datetime.now().isoformat()} | status={r.status_code} | content_type={r.headers.get('Content-Type')}\n")
                    fh.write(r.text[:2000].replace("\n"," ") + "\n\n")
                return None
            return r.text
        except Exception as e:
            time.sleep(backoff * (2 ** attempt))
            continue
    return None

# Extract candidate download links from page HTML
def extract_report_links(html):
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.find_all("a", href=True)
    links = []
    for a in anchors:
        href = a["href"].strip()
        text = (a.get_text() or "").strip()
        # canonicalize relative links
        if href.startswith("//"):
            href = "https:" + href
        elif href.startswith("/"):
            href = "https://www.nseindia.com" + href
        # accept only file-like links or ones pointing to reports/archives
        if re.search(r"\.(csv|zip|pdf|xls|xlsx|gz)$", href, re.IGNORECASE) or any(k in href.lower() for k in ["/daily", "/reports", "/archives", "/downloads", "bhavcopy", "daily-report", "market-pulse", "marketpulse"]):
            links.append({"href": href, "text": text})
    # deduplicate preserving order
    seen = set()
    uniq = []
    for l in links:
        if l["href"] not in seen:
            uniq.append(l); seen.add(l["href"])
    return uniq

# Choose best links for a given date
def choose_links_for_date(links, date_obj):
    """Return links whose href or text contains the date in common formats, else return latest few"""
    d_iso = date_obj.isoformat()  # YYYY-MM-DD
    d_ddmmyyyy = date_obj.strftime("%d%m%Y")  # 01012025
    d_dd_mm_yyyy = date_obj.strftime("%d-%m-%Y")
    d_dd_mmm_yyyy = date_obj.strftime("%d-%b-%Y")  # 01-Jan-2025 (Jan capitalized)
    candidates = []
    for l in links:
        h = l["href"].lower()
        t = (l["text"] or "").lower()
        if d_iso in h or d_iso in t or d_ddmmyyyy in h or d_ddmmyyyy in t or d_dd_mm_yyyy in h or d_dd_mm_yyyy in t or d_dd_mmm_yyyy.lower() in h or d_dd_mmm_yyyy.lower() in t:
            candidates.append(l)
    if candidates:
        return candidates
    # fallback: prefer links with "daily" or "bhavcopy" or today's month+year
    month_year = date_obj.strftime("%b").lower() + str(date_obj.year)
    fallback = [l for l in links if any(k in (l["href"]+l["text"]).lower() for k in ["daily", "daily-report", "bhavcopy", "market-pulse", month_year])]
    if fallback:
        return fallback[:6]
    # final fallback: return first few file-like links
    file_links = [l for l in links if re.search(r"\.(csv|zip|pdf|xls|xlsx|gz)$", l["href"], re.IGNORECASE)]
    return file_links[:6]

# Download a link and save to cache directory
def download_and_cache(link_href, date_obj):
    try:
        s = make_nse_session()
        r = s.get(link_href, timeout=30, stream=True)
        # if we get HTML here, likely a block page; save debug and return None
        ctype = r.headers.get("Content-Type","")
        if r.status_code != 200 or "text/html" in ctype.lower():
            debug_path = os.path.join(CACHE_DIR, "nse_debug.txt")
            with open(debug_path, "a", encoding="utf-8") as fh:
                fh.write(f"{datetime.datetime.now().isoformat()} | download_status={r.status_code} | url={link_href} | content_type={ctype}\n")
                fh.write(r.text[:2000].replace("\n"," ") + "\n\n")
            return None
        # determine filename
        # try Content-Disposition first
        cd = r.headers.get("Content-Disposition")
        if cd:
            m = re.search(r'filename="?([^"]+)"?', cd)
            filename = m.group(1) if m else None
        else:
            filename = None
        if not filename:
            filename = os.path.basename(link_href.split("?")[0])
            if not filename:
                filename = f"report_{date_obj.isoformat()}"
        # create safe local path
        local_name = f"nse_{date_obj.isoformat()}_{filename}"
        local_path = os.path.join(CACHE_DIR, local_name)
        # stream write
        with open(local_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
        return local_path
    except Exception as e:
        # log debug
        debug_path = os.path.join(CACHE_DIR, "nse_debug.txt")
        with open(debug_path, "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.datetime.now().isoformat()} | download_exception for {link_href} : {repr(e)}\n")
        return None

# Utility: show CSV preview if small
def preview_csv(path, max_rows=10):
    try:
        df = pd.read_csv(path, nrows=max_rows)
        return df
    except Exception:
        return None

# ----------------------
# UI
# ----------------------
st.title("NSE — Daily report fetch & cache (All Reports)")

left, right = st.columns([1, 2])
with left:
    # choose AM or PM daily fetch time
    st.markdown("**Schedule daily fetch**")
    fetch_hour = st.selectbox("Fetch time (IST) — choose one", ["09:00 (morning)", "17:00 (evening)", "Custom..."])
    custom_time = None
    if fetch_hour == "Custom...":
        custom_time = st.time_input("Custom fetch time (IST)", value=datetime.time(9,0))
    quick_fetch = st.button("Force fetch now")
    show_debug = st.checkbox("Show debug log (.nse_reports/nse_debug.txt)", value=False)
with right:
    st.markdown("This tool fetches report files published on NSE's All Reports page and caches one file per download in `.nse_reports/`. It tries to pick files for today's date and will fall back to latest available.")

# compute scheduled time
if fetch_hour == "09:00 (morning)":
    scheduled_time = datetime.time(9, 0)
elif fetch_hour == "17:00 (evening)":
    scheduled_time = datetime.time(17, 0)
else:
    scheduled_time = custom_time

# client side auto refresh (run only while page open)
def seconds_until_next_run(target_time: datetime.time):
    now = datetime.datetime.now(IST)
    today_target = datetime.datetime.combine(now.date(), target_time)
    today_target = IST.localize(today_target.replace(tzinfo=None))
    if today_target <= now:
        today_target = today_target + datetime.timedelta(days=1)
    return int((today_target - now).total_seconds())

# attempt scheduled autorefresh using streamlit-autorefresh (if available)
try:
    from streamlit_autorefresh import st_autorefresh
    sec = seconds_until_next_run(scheduled_time)
    # only schedule up to 24h
    if sec > 0:
        st_autorefresh(interval=sec*1000, key="nse_daily_refresh")
except Exception:
    pass

# Show cached files summary
cached_files = sorted([f for f in os.listdir(CACHE_DIR) if f.startswith("nse_")], reverse=True)
st.subheader("Cached files")
if cached_files:
    for f in cached_files[:50]:
        p = os.path.join(CACHE_DIR, f)
        st.write(f"- {f} — {os.path.getsize(p)//1024} KB — ", end="")
        st.download_button("Download", data=open(p,"rb").read(), file_name=f)
else:
    st.info("No cached files yet. Press 'Force fetch now' or wait for scheduled run.")

# Trigger fetch if scheduled time reached OR force clicked
def should_fetch_now(scheduled_time):
    # fetch if forced; else fetch only if no cache for today and current time >= scheduled_time (IST)
    now = datetime.datetime.now(IST)
    today = now.date()
    # check if we already have cached file for today
    have_today = any(today.isoformat() in fname for fname in cached_files)
    if quick_fetch:
        return True
    if not have_today:
        # if current IST time >= scheduled_time
        if now.time() >= scheduled_time:
            return True
    return False

if should_fetch_now(scheduled_time):
    st.info("Fetching All Reports page and trying to download today's files...")
    html = fetch_all_reports_page()
    if html is None:
        st.error("Could not fetch All Reports page (NSE may block). Check .nse_reports/nse_debug.txt for details.")
    else:
        links = extract_report_links(html)
        if not links:
            st.warning("No candidate report links found on the All Reports page.")
        else:
            st.write(f"Found {len(links)} candidate links (showing top 12):")
            for idx, l in enumerate(links[:12]):
                st.write(f"{idx+1}. {l['href']} — {l['text']}")
            today = datetime.date.today()
            chosen = choose_links_for_date(links, today)
            st.write(f"Selected {len(chosen)} link(s) for {today.isoformat()}:")
            downloaded = []
            for l in chosen:
                st.write("->", l["href"])
                local = download_and_cache(l["href"], today)
                if local:
                    st.success(f"Saved to {local}")
                    downloaded.append(local)
                else:
                    st.error(f"Failed to download {l['href']} — saved debug info to .nse_reports/nse_debug.txt")
            if downloaded:
                st.success(f"Downloaded {len(downloaded)} file(s). They are in the `.nse_reports/` folder.")
                # preview CSVs
                st.subheader("CSV Preview (if any of the downloads are CSV)")
                for p in downloaded:
                    if p.lower().endswith(".csv"):
                        df_preview = preview_csv(p, max_rows=8)
                        if df_preview is not None:
                            st.write("Preview of", os.path.basename(p))
                            st.dataframe(df_preview)
                        else:
                            st.write("Downloaded CSV but could not preview:", os.path.basename(p))

# show debug log if requested
if show_debug:
    dbg_path = os.path.join(CACHE_DIR, "nse_debug.txt")
    if os.path.exists(dbg_path):
        st.subheader("Debug log (tail)")
        txt = open(dbg_path, "r", encoding="utf-8").read()[-4000:]
        st.text_area("nse_debug.txt (tail)", txt, height=300)
    else:
        st.info("No debug log available.")

st.markdown("---")
st.markdown("**Notes & next steps**  \n- If NSE blocks this app on Streamlit Cloud, consider running it locally or enabling a GitHub Action to fetch the file at market close and push to a storage bucket.  \n- If you'd like, I can add an automatic S3 upload for each saved file, a backfill cron (GitHub Action), or a small UI that lists which specific report types you want (Bhavcopy / Market Pulse / etc.).")
