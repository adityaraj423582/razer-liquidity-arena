"""
Pull public Binance USDT-M perpetual funding-rate history.

No authentication — Binance public REST only.
Funding prints about every 8 hours (~540 records / 180 days).
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
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
DAYS = 180
MAX_PER_REQUEST = 1000
DATA_DIR = Path(__file__).resolve().parent / "data"


def _get_page(symbol: str, start_time_ms: int, end_time_ms: int) -> list[dict[str, Any]]:
    params = {
        "symbol": symbol,
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
            f"{symbol}: expected a list, got {type(payload).__name__}"
        )
    return payload


def fetch_funding(symbol: str, days: int = DAYS) -> list[dict[str, Any]]:
    end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    start_ms = int(
        (datetime.now(tz=timezone.utc) - timedelta(days=days)).timestamp() * 1000
    )

    rows: list[dict[str, Any]] = []
    cursor = start_ms
    pages = 0

    while cursor < end_ms:
        page = _get_page(symbol, cursor, end_ms)
        pages += 1
        if not page:
            break

        rows.extend(page)
        last_ts = int(page[-1]["fundingTime"])
        next_cursor = last_ts + 1
        if next_cursor <= cursor:
            raise RuntimeError(f"{symbol}: pagination cursor did not advance")
        cursor = next_cursor

        if len(page) < MAX_PER_REQUEST:
            break
        time.sleep(0.1)

    # Deduplicate by fundingTime
    by_ts: dict[int, dict[str, Any]] = {}
    for row in rows:
        by_ts[int(row["fundingTime"])] = row
    ordered = [by_ts[k] for k in sorted(by_ts)]
    print(f"  ({symbol}: fetched {len(ordered)} funding records across {pages} request(s))")
    return ordered


def parse_records(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in raw:
        ts_ms = int(row["fundingTime"])
        records.append(
            {
                "timestamp": datetime.fromtimestamp(
                    ts_ms / 1000, tz=timezone.utc
                ).isoformat(),
                "fundingRate": float(row["fundingRate"]),
            }
        )
    return records


def save_csv(symbol: str, records: list[dict[str, Any]]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{symbol}_funding.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "fundingRate"])
        writer.writeheader()
        writer.writerows(records)
    return path


def print_summary(symbol: str, records: list[dict[str, Any]], path: Path) -> None:
    rates = [r["fundingRate"] for r in records]
    avg = sum(rates) / len(rates)
    print(f"{symbol}:")
    print(f"  records: {len(records)}")
    print(f"  date range: {records[0]['timestamp']} -> {records[-1]['timestamp']}")
    print(f"  fundingRate min: {min(rates)}")
    print(f"  fundingRate max: {max(rates)}")
    print(f"  fundingRate avg: {avg}")
    print(f"  saved: {path}")


def main() -> int:
    expected = DAYS * 3  # ~every 8 hours
    print("Credentials used: none (Binance public REST)")
    print(f"Endpoint: {BASE_URL}")
    print(
        f"days={DAYS} (~{expected} funding prints/symbol), "
        f"paginate with startTime, max {MAX_PER_REQUEST}/request\n"
    )

    for symbol in SYMBOLS:
        try:
            raw = fetch_funding(symbol)
            records = parse_records(raw)
            if not records:
                print(f"{symbol}: STOP - no funding records returned")
                return 1
            path = save_csv(symbol, records)
            print_summary(symbol, records, path)
            print()
        except Exception as exc:
            print(f"ERROR: {exc}")
            return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
