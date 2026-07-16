"""
Safety-first order lifecycle smoke test (Liquidity Arena 2026 Track A).

1) Read symbol min size + price tick
2) One-shot BBO bid
3) Place LIMIT BUY 20% below bid at min qty (should not fill)
4) Cancel the order

Stops immediately on any step failure. Does not auto-close positions.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import hmac
import json
import os
import sys
import time
import uuid
from decimal import Decimal, ROUND_DOWN
from typing import Any, Mapping
from urllib.parse import urljoin

import requests
import websockets
from dotenv import load_dotenv
from websockets.exceptions import WebSocketException

SYM = "BINANCE_PERP_BTC_USDT"
WS_URL = "wss://mds.ltp-contest.com/marketdata/v2/public"
SUCCESS_CODES = {200000, "200000", 0, "0"}


def sign_request(secret_key: str, params: Mapping[str, str], nonce: str) -> str:
    sorted_items = sorted(params.items(), key=lambda item: item[0])
    param_string = "&".join(f"{key}={value}" for key, value in sorted_items)
    message = f"{param_string}&{nonce}"
    return hmac.new(
        secret_key.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def load_credentials() -> tuple[str, str, str]:
    load_dotenv()
    access_key = os.getenv("LTP_ACCESS_KEY", "").strip()
    secret_key = os.getenv("LTP_SECRET_KEY", "").strip()
    api_host = os.getenv("LTP_API_HOST", "").strip().rstrip("/")
    missing = [
        name
        for name, value in (
            ("LTP_ACCESS_KEY", access_key),
            ("LTP_SECRET_KEY", secret_key),
            ("LTP_API_HOST", api_host),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required .env values: {', '.join(missing)}")
    return access_key, secret_key, api_host


def api_request(
    method: str,
    path: str,
    access_key: str,
    secret_key: str,
    api_host: str,
    params: dict[str, str] | None = None,
) -> dict[str, Any]:
    params = dict(params or {})
    nonce = str(int(time.time()))
    signature = sign_request(secret_key, params, nonce)
    headers = {
        "Content-Type": "application/json",
        "nonce": nonce,
        "signature": signature,
        "X-MBX-APIKEY": access_key,
    }
    url = urljoin(f"{api_host}/", path.lstrip("/"))

    try:
        if method == "GET":
            response = requests.get(url, params=params, headers=headers, timeout=30)
        elif method == "POST":
            response = requests.post(
                url, data=json.dumps(params), headers=headers, timeout=30
            )
        elif method == "DELETE":
            response = requests.delete(
                url, data=json.dumps(params), headers=headers, timeout=30
            )
        else:
            raise RuntimeError(f"Unsupported HTTP method: {method}")
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"Connection error calling {method} {path}: {exc}") from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(f"Timeout calling {method} {path}: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Request error calling {method} {path}: {exc}") from exc

    if response.status_code in (401, 403):
        raise RuntimeError(
            f"Auth error on {method} {path}: HTTP {response.status_code}"
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"Unexpected HTTP {response.status_code} on {method} {path}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Non-JSON response on {method} {path}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected JSON shape on {method} {path}")
    return payload


def _decode_frame(raw: str | bytes) -> str:
    if isinstance(raw, str):
        return raw
    if len(raw) >= 2 and raw[0] == 0x1F and raw[1] == 0x8B:
        return gzip.decompress(raw).decode("utf-8")
    return raw.decode("utf-8")


async def fetch_bid_once(timeout_s: float = 15.0) -> Decimal:
    subscribe = {
        "event": "subscribe",
        "arg": [{"channel": "BBO", "sym": SYM}],
    }
    deadline = asyncio.get_running_loop().time() + timeout_s
    try:
        async with websockets.connect(WS_URL, open_timeout=10, close_timeout=5) as ws:
            await ws.send(json.dumps(subscribe))
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise RuntimeError("Timed out waiting for BBO bid")
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                text = _decode_frame(raw)
                if text == "ping":
                    await ws.send("pong")
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                arg = payload.get("arg")
                if not isinstance(arg, dict) or arg.get("channel") != "BBO":
                    continue
                data = payload.get("data")
                if isinstance(data, list) and data:
                    data = data[0]
                if not isinstance(data, dict):
                    continue
                bid = data.get("bid")
                if bid is None or str(bid).strip() == "":
                    continue
                return Decimal(str(bid))
    except (OSError, WebSocketException, asyncio.TimeoutError) as exc:
        raise RuntimeError(f"Market-data connection failure: {exc}") from exc


def round_down_to_tick(price: Decimal, tick_size: Decimal) -> Decimal:
    if tick_size <= 0:
        raise RuntimeError(f"Invalid tickSize: {tick_size}")
    steps = (price / tick_size).to_integral_value(rounding=ROUND_DOWN)
    return steps * tick_size


def format_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def extract_sym_info(payload: dict[str, Any]) -> tuple[str, str]:
    data = payload.get("data")
    info: dict[str, Any] | None = None
    if isinstance(data, dict):
        if SYM in data and isinstance(data[SYM], dict):
            info = data[SYM]
        elif data.get("sym") == SYM:
            info = data
        else:
            # Some responses nest under a list
            for value in data.values():
                if isinstance(value, dict) and value.get("sym") == SYM:
                    info = value
                    break
    if info is None:
        raise RuntimeError("Symbol info response did not include BINANCE_PERP_BTC_USDT")

    min_size = info.get("minSize")
    tick_size = info.get("tickSize")
    if min_size is None or str(min_size).strip() == "":
        raise RuntimeError("Symbol info missing minSize (minimum order quantity)")
    if tick_size is None or str(tick_size).strip() == "":
        raise RuntimeError("Symbol info missing tickSize (price step size)")
    return str(min_size), str(tick_size)


def looks_filled(order_payload: dict[str, Any]) -> bool:
    data = order_payload.get("data")
    if not isinstance(data, dict):
        return False
    state = str(data.get("orderState", "")).upper()
    if state in {"FILLED", "PARTIALLY_FILLED", "PARTIAL"}:
        return True
    for key in ("executedQty", "cumExecQty", "filledQty", "lastExecutedQty"):
        raw = data.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            if Decimal(str(raw)) > 0:
                return True
        except Exception:
            continue
    return False


def main() -> int:
    print("=== Step 1: symbol info (read-only) ===")
    try:
        access_key, secret_key, api_host = load_credentials()
        sym_payload = api_request(
            "GET",
            "api/v1/trading/sym/info",
            access_key,
            secret_key,
            api_host,
            {"sym": SYM},
        )
        if sym_payload.get("code") not in SUCCESS_CODES:
            print(
                f"Symbol info failed: code={sym_payload.get('code')} "
                f"message={sym_payload.get('message')}"
            )
            return 1
        min_qty, tick_size = extract_sym_info(sym_payload)
        print(f"min order quantity: {min_qty}")
        print(f"price step size: {tick_size}")
    except Exception as exc:
        print(f"STOP at symbol info: {exc}")
        return 1

    print("\n=== Step 2: one-shot BBO bid ===")
    try:
        bid = asyncio.run(fetch_bid_once())
        print(f"bid: {format_decimal(bid)}")
    except Exception as exc:
        print(f"STOP at market data: {exc}")
        return 1

    print("\n=== Step 3: calculate safe limit price ===")
    try:
        raw_limit = bid * Decimal("0.80")
        limit_price = round_down_to_tick(raw_limit, Decimal(tick_size))
        if limit_price <= 0:
            raise RuntimeError(f"Computed non-positive limitPrice: {limit_price}")
        if limit_price >= bid:
            raise RuntimeError(
                f"Safety check failed: limitPrice {limit_price} is not below bid {bid}"
            )
        limit_price_str = format_decimal(limit_price)
        print(f"limitPrice (20% below bid, tick-rounded down): {limit_price_str}")
        print(f"orderQty (min): {min_qty}")
    except Exception as exc:
        print(f"STOP at price calculation: {exc}")
        return 1

    print("\n=== Step 4: place LIMIT BUY (should not fill) ===")
    client_order_id = f"razertest{int(time.time())}{uuid.uuid4().hex[:6]}"
    # clientOrderId may only contain a-z and 0-9
    client_order_id = "".join(ch for ch in client_order_id if ch.isalnum()).lower()
    order_id: str | None = None
    try:
        place_params = {
            "sym": SYM,
            "side": "BUY",
            "orderType": "LIMIT",
            "orderQty": str(min_qty),
            "limitPrice": limit_price_str,
            "timeInForce": "GTC",
            "clientOrderId": client_order_id,
        }
        place_payload = api_request(
            "POST",
            "api/v1/trading/order",
            access_key,
            secret_key,
            api_host,
            place_params,
        )
        code = place_payload.get("code")
        message = place_payload.get("message")
        data = place_payload.get("data") if isinstance(place_payload.get("data"), dict) else {}
        order_id = data.get("orderId") if isinstance(data, dict) else None
        print(f"code: {code}")
        print(f"message: {message}")
        print(f"orderId: {order_id}")

        if code not in SUCCESS_CODES:
            print("STOP: place order did not succeed; skip cancel.")
            return 1
        if not order_id:
            print("STOP: place succeeded but no orderId returned; cannot cancel safely.")
            return 1

        # Safety: query order once for fill indication before waiting
        query_payload = api_request(
            "GET",
            "api/v1/trading/order",
            access_key,
            secret_key,
            api_host,
            {"orderId": str(order_id)},
        )
        if looks_filled(query_payload):
            state = (query_payload.get("data") or {}).get("orderState")
            print(
                "CRITICAL: order appears FILLED or PARTIALLY_FILLED "
                f"(orderState={state}). Stopping. Not auto-closing any position."
            )
            return 1
    except Exception as exc:
        print(f"STOP at place order: {exc}")
        return 1

    print("\n=== Step 5: wait 3 seconds ===")
    time.sleep(3)

    print("\n=== Step 6: cancel order ===")
    try:
        # Re-check fill status before cancel
        query_payload = api_request(
            "GET",
            "api/v1/trading/order",
            access_key,
            secret_key,
            api_host,
            {"orderId": str(order_id)},
        )
        if looks_filled(query_payload):
            state = (query_payload.get("data") or {}).get("orderState")
            print(
                "CRITICAL: order appears FILLED or PARTIALLY_FILLED before cancel "
                f"(orderState={state}). Stopping. Not auto-closing any position."
            )
            return 1

        cancel_payload = api_request(
            "DELETE",
            "api/v1/trading/order",
            access_key,
            secret_key,
            api_host,
            {"orderId": str(order_id)},
        )
        print(f"code: {cancel_payload.get('code')}")
        print(f"message: {cancel_payload.get('message')}")
        cancel_data = cancel_payload.get("data")
        if isinstance(cancel_data, dict) and cancel_data.get("action"):
            print(f"action: {cancel_data.get('action')}")
        if cancel_payload.get("code") not in SUCCESS_CODES:
            print("WARNING: cancel response was not success — check order manually.")
            return 1
    except Exception as exc:
        print(f"STOP at cancel order: {exc}")
        return 1

    print("\n=== Step 7: confirm final order state ===")
    try:
        time.sleep(2)
        final_payload = api_request(
            "GET",
            "api/v1/trading/order",
            access_key,
            secret_key,
            api_host,
            {"orderId": str(order_id)},
        )
        if final_payload.get("code") not in SUCCESS_CODES:
            print(
                f"STOP at final order query: code={final_payload.get('code')} "
                f"message={final_payload.get('message')}"
            )
            return 1
        final_data = final_payload.get("data")
        if not isinstance(final_data, dict):
            print("STOP at final order query: missing data object")
            return 1
        # LTP uses orderState; fall back to common aliases if present
        status_field = None
        status_value = None
        for key in ("orderState", "orderStatus", "status"):
            if key in final_data and final_data[key] is not None:
                status_field = key
                status_value = final_data[key]
                break
        if status_field is None:
            print(
                "STOP at final order query: no orderState/orderStatus/status field found"
            )
            return 1
        print(f"{status_field}: {status_value}")
        if str(status_value).upper() not in {"CANCELED", "CANCELLED"}:
            print(
                f"WARNING: expected CANCELED/CANCELLED, got {status_value!r} — check manually."
            )
            return 1
    except Exception as exc:
        print(f"STOP at final order query: {exc}")
        return 1

    print("\nLifecycle complete: placed far-below-market LIMIT BUY and confirmed cancel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
