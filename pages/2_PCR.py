import streamlit as st
import requests
import json
from datetime import datetime

st.title("NSE Intraday Debug Tool (Exact Response Needed)")
st.write("This tool fetches NSE intraday JSON/HTML so we can fix your main script.")

URL = "https://www.nseindia.com/api/chart-databyindex?index=EQUITY%7CNIFTY%2050&preopen=0"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                  " AppleWebKit/537.36 (KHTML, like Gecko)"
                  " Chrome/125.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Referer": "https://www.nseindia.com",
    "Connection": "keep-alive"
}

def warmup_session(sess):
    try:
        sess.get("https://www.nseindia.com", timeout=8)
    except:
        pass

def fetch_nse():
    sess = requests.Session()
    sess.headers.update(HEADERS)

    warmup_session(sess)

    try:
        r = sess.get(URL, timeout=15)
        return r.status_code, r.headers.get("Content-Type", ""), r.text
    except Exception as e:
        return None, None, f"Exception: {e}"

st.subheader("Fetching live…")

status, ctype, body = fetch_nse()

st.code(f"""
Status: {status}
Content-Type: {ctype}
""")

# Show first 1500 chars
preview = body[:1500]
st.text_area("Response Preview (first 1500 chars)", preview, height=400)

# Attempt JSON parsing
try:
    j = json.loads(body)
    st.success("VALID JSON received from NSE")
    st.json(j)
except Exception as e:
    st.error(f"NOT JSON → {e}")
    if "<html" in body.lower():
        st.warning("This looks like an HTML block page (403 / captcha / rate-limit).")

# Save debug for analysis
DEBUG_FILE = "nse_debug_output.txt"
with open(DEBUG_FILE, "w", encoding="utf-8") as f:
    f.write(f"TIME: {datetime.now()}\nSTATUS: {status}\nTYPE: {ctype}\n\nBODY:\n{body}")

st.success("Saved full response to nse_debug_output.txt. Download and paste here.")
st.download_button("Download Debug File", body, file_name="nse_debug_output.txt")
