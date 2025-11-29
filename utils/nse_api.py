# utils/nse_api.py
import requests, time, random
from cachetools import TTLCache, cached
from typing import Tuple, Dict, Any, List

BASE = "https://www.nseindia.com"
# polite headers - mimic a modern browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Referer": "https://www.nseindia.com"
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)
# caches
INDEX_CACHE = TTLCache(maxsize=20, ttl=30)    # 30s
OC_CACHE = TTLCache(maxsize=40, ttl=12)       # 12s (option chain)
QUOTE_CACHE = TTLCache(maxsize=400, ttl=5)    # 5s for quick quotes

def _ensure_session():
    """
    Hit NSE homepage once to establish cookies. Idempotent.
    """
    try:
        if not SESSION.cookies.get("nse_cookie_init"):
            SESSION.get(BASE, timeout=10)
            SESSION.cookies.set("nse_cookie_init", "1")
            # small sleep to let cookies set
            time.sleep(0.05)
    except Exception:
        # swallow; actual requests will retry
        pass

def _get_json(path: str, params: dict = None, tries: int = 3, backoff: float = 0.6):
    _ensure_session()
    url = f"{BASE}{path}"
    attempt = 0
    while attempt < tries:
        try:
            r = SESSION.get(url, params=params, timeout=12)
            if r.status_code == 200:
                # Some endpoints occasionally return text/html on block -> try again
                return r.json()
            # on other codes, backoff and retry
            attempt += 1
            time.sleep(backoff * (2 ** attempt) + random.random() * 0.3)
        except ValueError:
            # JSON parse error (HTML returned), retry
            attempt += 1
            time.sleep(backoff * (2 ** attempt) + random.random() * 0.3)
        except requests.RequestException:
            attempt += 1
            time.sleep(backoff * (2 ** attempt) + random.random() * 0.3)
    raise ConnectionError(f"Failed to GET {url} after {tries} tries")

# -------------------- Public API --------------------

@cached(INDEX_CACHE)
def get_all_indices() -> List[Dict[str, Any]]:
    """
    Returns a list of index dicts from NSE /api/allIndices
    """
    data = _get_json("/api/allIndices")
    items = data.get("data") if isinstance(data, dict) else []
    return items or []

@cached(OC_CACHE)
def get_option_chain_index(symbol: str = "NIFTY") -> List[Dict[str, Any]]:
    """
    Returns the option-chain 'records.data' list (list of dicts).
    Normalized minimal fields are returned by consumer code.
    """
    data = _get_json("/api/option-chain-indices", params={"symbol": symbol})
    records = data.get("records", {}) if isinstance(data, dict) else {}
    raw = records.get("data", []) if isinstance(records, dict) else []
    return raw or []

@cached(QUOTE_CACHE)
def get_quote(symbol: str) -> Dict[str, Any]:
    """
    Get latest quote for an equity or index. Tries equity endpoint then index endpoint.
    Returns normalized dict with keys like lastPrice, pChange, change.
    """
    # try equity quote first
    try:
        q = _get_json("/api/quote-equity", params={"symbol": symbol})
        # normalize
        price_info = q.get("priceInfo") if isinstance(q, dict) else None
        if price_info:
            return {
                "lastPrice": price_info.get("lastPrice") or price_info.get("lastTradedPrice"),
                "change": price_info.get("change"),
                "pChange": price_info.get("pChange"),
                "timestamp": price_info.get("lastUpdateTime")
            }
    except Exception:
        pass

    # try index quote
    try:
        q = _get_json("/api/quote-index", params={"index": symbol})
        if isinstance(q, dict) and "data" in q:
            # some index endpoints return data list
            d = q.get("data")
            if isinstance(d, list) and len(d) > 0:
                entry = d[0]
                return {
                    "lastPrice": entry.get("last"),
                    "change": entry.get("change"),
                    "pChange": entry.get("pChange"),
                    "timestamp": entry.get("time")
                }
        # fallback: return raw
        return q if isinstance(q, dict) else {}
    except Exception:
        return {}

# small helper: convert raw records into normalized table rows
def normalize_option_chain_records(raw_records: List[Dict[str, Any]]) -> list:
    rows = []
    for r in raw_records:
        strike = r.get("strikePrice") or r.get("strike")
        ce = r.get("CE") or {}
        pe = r.get("PE") or {}
        rows.append({
            "strike": int(strike) if strike is not None else None,
            "CE_OI": int(ce.get("openInterest") or 0),
            "PE_OI": int(pe.get("openInterest") or 0),
            "CE_changeOI": int(ce.get("changeinOpenInterest") or 0),
            "PE_changeOI": int(pe.get("changeinOpenInterest") or 0),
            "CE_ltp": float(ce.get("lastPrice") or 0.0),
            "PE_ltp": float(pe.get("lastPrice") or 0.0),
        })
    # sort by strike
    rows = [r for r in rows if r["strike"] is not None]
    rows.sort(key=lambda x: x["strike"])
    return rows
