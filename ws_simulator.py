# ws_simulator.py
# Simple market data simulator that mimics a live websocket feed.
# It updates instrument prices periodically and stores them in LIVE_TICKS.

import time
import threading
import random
from collections import defaultdict
from typing import Dict

LIVE_TICKS: Dict[str, dict] = {}
LIVE_TICKS_LOCK = threading.Lock()
_RUNNING = False

# default symbols and base prices
SYMBOL_BASE = {
    "NIFTY": 21600.0,
    "BANKNIFTY": 48300.0,
    "RELIANCE": 2500.0,
    "INFY": 1500.0,
    "TCS": 3300.0
}

def _init_ticks():
    now = time.time()
    with LIVE_TICKS_LOCK:
        for s, base in SYMBOL_BASE.items():
            LIVE_TICKS[s] = {
                "symbol": s,
                "lastPrice": round(base, 2),
                "change": 0.0,
                "pChange": 0.0,
                "timestamp": now
            }

def _simulate_step():
    """Random walk step for each symbol."""
    with LIVE_TICKS_LOCK:
        for s, tick in LIVE_TICKS.items():
            base = tick["lastPrice"]
            # random pct move between -0.25% and +0.25%
            pct = random.uniform(-0.0025, 0.0025)
            new_price = base * (1 + pct)
            change = new_price - base
            pchange = (change / base) * 100 if base != 0 else 0
            tick["lastPrice"] = round(new_price, 2)
            tick["change"] = round(change, 2)
            tick["pChange"] = round(pchange, 2)
            tick["timestamp"] = time.time()

def simulator_loop(interval: float = 1.0):
    global _RUNNING
    if _RUNNING:
        return
    _RUNNING = True
    _init_ticks()
    try:
        while _RUNNING:
            _simulate_step()
            time.sleep(interval)
    finally:
        _RUNNING = False

def start_simulator_in_thread(interval: float = 1.0):
    """Start simulator as a daemon background thread (idempotent)."""
    t = threading.Thread(target=simulator_loop, kwargs={"interval": interval}, daemon=True)
    t.start()
    return t

if __name__ == "__main__":
    print("Starting simulator in foreground (Ctrl-C to stop)...")
    _init_ticks()
    try:
        simulator_loop(interval=1.0)
    except KeyboardInterrupt:
        print("Simulator stopped.")
