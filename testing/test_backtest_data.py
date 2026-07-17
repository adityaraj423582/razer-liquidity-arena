"""
Pull public Binance USDT-M perpetual 1h candles for backtesting.

No authentication / no LTP credentials — Binance public REST only.
Paginates past the 1500-candle per-request limit for longer histories.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import csv
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "LINKUSDT",
)
INTERVAL = "1h"
INTERVAL_MS = 60 * 60 * 1000
DAYS = 180
MAX_PER_REQUEST = 1500  # Binance hard limit
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def _get_klines_page(symbol: str, start_time_ms: int, end_time_ms: int) -> list[list[Any]]:
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "startTime": start_time_ms,
        "endTime": end_time_ms,
        "limit": MAX_PER_REQUEST,
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"{symbol}: connection/DNS error - {exc}") from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(f"{symbol}: request timed out - {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"{symbol}: request failed - {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"{symbol}: unexpected HTTP {response.status_code}: {response.text[:200]}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"{symbol}: response is not valid JSON") from exc

    if not isinstance(payload, list):
        raise RuntimeError(
            f"{symbol}: expected a list of candles, got {type(payload).__name__}"
        )
    return payload


def fetch_klines(symbol: str, days: int = DAYS) -> list[list[Any]]:
    """Paginate 1h klines from (now - days) to now using startTime."""
    end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    start_ms = int(
        (datetime.now(tz=timezone.utc) - timedelta(days=days)).timestamp() * 1000
    )

    all_rows: list[list[Any]] = []
    cursor = start_ms
    pages = 0

    while cursor < end_ms:
        page = _get_klines_page(symbol, cursor, end_ms)
        pages += 1
        if not page:
            break

        all_rows.extend(page)
        last_open_ms = int(page[-1][0])
        next_cursor = last_open_ms + INTERVAL_MS
        if next_cursor <= cursor:
            raise RuntimeError(f"{symbol}: pagination cursor did not advance")
        cursor = next_cursor

        # Full page likely means more data; short page means we reached the end
        if len(page) < MAX_PER_REQUEST:
            break

        # Be polite to the public endpoint
        time.sleep(0.1)

    # Deduplicate by open time (safety if windows overlap)
    by_open: dict[int, list[Any]] = {}
    for row in all_rows:
        by_open[int(row[0])] = row
    ordered = [by_open[k] for k in sorted(by_open)]
    print(f"  ({symbol}: fetched {len(ordered)} candles across {pages} request(s))")
    return ordered


def parse_candles(raw: list[list[Any]]) -> list[dict[str, Any]]:
    candles: list[dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            raise RuntimeError(f"Malformed candle row: {row!r}")
        open_time_ms = int(row[0])
        candles.append(
            {
                "timestamp": datetime.fromtimestamp(
                    open_time_ms / 1000, tz=timezone.utc
                ).isoformat(),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
        )
    return candles


def save_csv(symbol: str, candles: list[dict[str, Any]]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{symbol}_{INTERVAL}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "open", "high", "low", "close", "volume"],
        )
        writer.writeheader()
        writer.writerows(candles)
    return path


def print_summary(symbol: str, candles: list[dict[str, Any]], path: Path) -> None:
    closes = [c["close"] for c in candles]
    print(f"{symbol}:")
    print(f"  candles: {len(candles)}")
    print(f"  date range: {candles[0]['timestamp']} -> {candles[-1]['timestamp']}")
    print(f"  close min: {min(closes)}")
    print(f"  close max: {max(closes)}")
    print(f"  saved: {path}")


def main() -> int:
    expected = DAYS * 24
    print("Credentials used: none (Binance public REST)")
    print(f"Endpoint: {BASE_URL}")
    print(
        f"interval={INTERVAL} days={DAYS} (~{expected} hourly candles), "
        f"paginate with startTime, max {MAX_PER_REQUEST}/request\n"
    )

    for symbol in SYMBOLS:
        try:
            raw = fetch_klines(symbol)
            candles = parse_candles(raw)
            if not candles:
                print(f"{symbol}: STOP - no candles returned")
                return 1
            path = save_csv(symbol, candles)
            print_summary(symbol, candles, path)
            print()
        except Exception as exc:
            print(f"ERROR: {exc}")
            return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
