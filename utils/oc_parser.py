# utils/oc_parser.py
from typing import List, Dict, Tuple

def parse_tv_rows(tv_rows: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Convert TradingView 'data' rows into sorted CE and PE lists.
    Each returned row has keys:
      - strike (int)
      - expiry (string or timestamp)
      - ltp (float)
      - oi (int)
      - bid (float)
      - ask (float)
      - vol (int)
      - side ('CE' or 'PE')
    """
    calls = []
    puts = []
    for item in tv_rows:
        # TradingView rows format example: item["d"] list aligns with columns in tv_api
        d = item.get("d", [])
        if not d or len(d) < 9:
            continue
        try:
            strike = int(d[6]) if d[6] is not None else None
        except Exception:
            # sometimes float-like
            try:
                strike = int(float(d[6]))
            except Exception:
                strike = None
        if strike is None:
            continue
        row = {
            "strike": strike,
            "expiry": d[5],
            "ltp": float(d[1]) if d[1] is not None else 0.0,
            "oi": int(d[2]) if d[2] is not None else 0,
            "bid": float(d[3]) if d[3] is not None else 0.0,
            "ask": float(d[4]) if d[4] is not None else 0.0,
            "vol": int(d[8]) if d[8] is not None else 0,
        }
        side = d[7].lower() if isinstance(d[7], str) else ""
        if "call" in side or side == "c" or side == "ce":
            row["side"] = "CE"
            calls.append(row)
        else:
            row["side"] = "PE"
            puts.append(row)
    calls = sorted(calls, key=lambda x: x["strike"])
    puts = sorted(puts, key=lambda x: x["strike"])
    return calls, puts
