"""
Standalone probe: can we reach Binance market data via data-api.binance.vision?
Does NOT modify live_trading_loop.py. One small request only (limit=10).
"""

from __future__ import annotations

import json

import requests

URL = "https://data-api.binance.vision/api/v3/klines"
PARAMS = {"symbol": "BTCUSDT", "interval": "1h", "limit": 10}


def main() -> int:
    print(f"GET {URL}")
    print(f"params={PARAMS}")
    try:
        response = requests.get(URL, params=PARAMS, timeout=30)
    except requests.RequestException as exc:
        print(f"REQUEST ERROR: {exc}")
        return 1

    print(f"status_code={response.status_code}")
    print(f"content-type={response.headers.get('content-type')}")
    text = response.text[:500]
    print(f"body_preview={text!r}")

    if response.status_code != 200:
        return 1

    try:
        rows = response.json()
    except json.JSONDecodeError as exc:
        print(f"JSON ERROR: {exc}")
        return 1

    print(f"row_count={len(rows) if isinstance(rows, list) else type(rows).__name__}")
    if isinstance(rows, list) and rows:
        print(f"first_row={rows[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
