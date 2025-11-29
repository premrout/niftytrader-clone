import requests

TV_URL = "https://scanner.tradingview.com/india/scan"

def get_tv_option_chain(symbol="NIFTY"):
    payload = {
        "symbols": {
            "tickers": [f"NSE:{symbol}"],
            "query": {"types": []}
        },
        "columns": [
            "name",
            "close",
            "open_interest",
            "bid",
            "ask",
            "expiration",
            "strike_price",
            "option_type",
            "volume"
        ]
    }
    
    try:
        r = requests.post(TV_URL, json=payload, timeout=10)
        data = r.json().get("data", [])
        return data, None
    except Exception as e:
        return None, str(e)


def parse_option_chain(tv_data):
    calls = []
    puts = []

    for item in tv_data:
        d = item["d"]
        oc = {
            "strikePrice": d[6],
            "expiryDate": d[5],
            "openInterest": d[2],
            "changeinOpenInterest": 0,  # TV doesn't give directly
            "lastPrice": d[1],
            "bidprice": d[3],
            "askPrice": d[4],
            "volume": d[8]
        }

        if d[7] == "call":
            calls.append(oc)
        else:
            puts.append(oc)

    return sorted(calls, key=lambda x: x["strikePrice"]), sorted(puts, key=lambda x: x["strikePrice"])
