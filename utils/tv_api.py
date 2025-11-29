# utils/tv_api.py
import requests
from typing import List, Dict, Any

TV_SCAN_URL = "https://scanner.tradingview.com/india/scan"

def get_tv_optionchain(symbol: str = "NIFTY") -> List[Dict[str, Any]]:
    """
    Query TradingView scanner for a ticker's option data.
    Returns list of data rows (TradingView format) or empty list on error.
    """
    payload = {
        "symbols": {
            "tickers": [f"NSE:{symbol.upper()}"],
            "query": {"types": []}
        },
        "columns": [
            "name",            # 0
            "close",           # 1 -> last price
            "open_interest",   # 2
            "bid",             # 3
            "ask",             # 4
            "expiration",      # 5
            "strike_price",    # 6
            "option_type",     # 7 -> "call" or "put"
            "volume"           # 8
        ]
    }
    try:
        r = requests.post(TV_SCAN_URL, json=payload, timeout=10)
        r.raise_for_status()
        j = r.json()
        return j.get("data", []) or []
    except Exception as e:
        # Keep it quiet for UI — return empty list on failure
        # Caller should show a friendly message
        print("TV API error:", e)
        return []
