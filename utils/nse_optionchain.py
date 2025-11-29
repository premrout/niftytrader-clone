import requests
import time

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain"
}

BASE_URL = "https://www.nseindia.com/api/option-chain-indices?symbol="


session = requests.Session()
session.headers.update(NSE_HEADERS)

def refresh_cookies():
    # Get fresh cookies
    session.get("https://www.nseindia.com", timeout=10)
    time.sleep(1)


def get_option_chain(symbol):
    """
    Fetch option chain data from NSE.
    Handles cookies, 403 errors, missing fields.
    """

    url = BASE_URL + symbol.upper()

    try:
        res = session.get(url, timeout=10)

        # If NSE blocks → refresh cookies
        if res.status_code in (401, 403):
            refresh_cookies()
            res = session.get(url, timeout=10)

        data = res.json()

        if "records" not in data:
            return None, "NSE returned no data"

        oc = data["records"].get("data", [])

        if not oc:
            return None, "Empty option chain"

        return oc, None

    except Exception as e:
        return None, str(e)
