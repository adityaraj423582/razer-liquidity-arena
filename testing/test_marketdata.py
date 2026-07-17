"""
Read-only public market-data smoke test (Liquidity Arena 2026 Track A).

Connects to the contest MDS WebSocket, subscribes to BTC ticker + BBO,
prints a few key fields for up to 15 seconds. No authentication / no API keys.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import asyncio
import gzip
import json
import sys
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed, InvalidURI, WebSocketException

WS_URL = "wss://mds.ltp-contest.com/marketdata/v2/public"
SUBSCRIBE = {
    "event": "subscribe",
    "arg": [
        {"channel": "TICKER", "sym": "BINANCE_PERP_BTC_USDT"},
        {"channel": "BBO", "sym": "BINANCE_PERP_BTC_USDT"},
    ],
}
LISTEN_SECONDS = 15
MAX_PRINTS = 12


def _decode_frame(raw: str | bytes) -> str:
    """Decode a WebSocket frame; LTP MDS may send gzip-compressed binary frames."""
    if isinstance(raw, str):
        return raw
    # gzip magic header 0x1f 0x8b
    if len(raw) >= 2 and raw[0] == 0x1F and raw[1] == 0x8B:
        try:
            return gzip.decompress(raw).decode("utf-8")
        except OSError as exc:
            raise ValueError(f"gzip decompress failed — {exc}") from exc
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("received non-UTF-8 binary frame (not gzip)") from exc


def _first(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _arg_meta(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    arg = payload.get("arg")
    if isinstance(arg, dict):
        channel = arg.get("channel")
        sym = arg.get("sym")
        return (
            str(channel) if channel is not None else None,
            str(sym) if sym is not None else None,
        )
    return None, None


def _data_object(payload: dict[str, Any]) -> dict[str, Any] | None:
    data = payload.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    if isinstance(data, dict):
        return data
    return None


def format_message(payload: dict[str, Any]) -> str | None:
    """Return a one-line summary, or None to skip printing (e.g. subscribe acks)."""
    channel, sym = _arg_meta(payload)

    # Subscribe / error acks — keep quiet unless something failed
    event = payload.get("event")
    if event in ("subscribe", "unsubscribe", "error") or (
        "data" not in payload and ("code" in payload or "msg" in payload)
    ):
        code = payload.get("code")
        if event == "subscribe" and code in (0, "0", None):
            return None
        msg = _first(payload, "msg", "message")
        parts = [f"event={event}" if event is not None else "ack"]
        if code is not None:
            parts.append(f"code={code}")
        if msg is not None:
            parts.append(f"msg={msg}")
        return " | ".join(parts)

    data = _data_object(payload)
    if not isinstance(data, dict):
        return None

    channel_u = (channel or "").upper()

    if channel_u == "TICKER":
        return f"sym={sym} | last={data.get('last')} | chg={data.get('chg')}"

    if channel_u == "BBO":
        return (
            f"sym={sym} | bid={data.get('bid')} | ask={data.get('ask')} | "
            f"bidqty={data.get('bidqty')} | askqty={data.get('askqty')}"
        )

    return None


async def run() -> int:
    print("Credentials used: none (public market-data WebSocket)")
    print(f"Connecting to {WS_URL}")
    print(f"Subscribe: {json.dumps(SUBSCRIBE)}")
    print(f"Listening up to {LISTEN_SECONDS}s...\n")
    printed = 0
    try:
        async with websockets.connect(WS_URL, open_timeout=10, close_timeout=5) as ws:
            await ws.send(json.dumps(SUBSCRIBE))
            deadline = asyncio.get_running_loop().time() + LISTEN_SECONDS

            try:
                while printed < MAX_PRINTS:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        print(f"Reached {LISTEN_SECONDS}s listen window")
                        break
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    except asyncio.TimeoutError:
                        print(f"Reached {LISTEN_SECONDS}s listen window")
                        break

                    try:
                        text = _decode_frame(raw)
                    except ValueError as exc:
                        print(f"Malformed message: {exc}")
                        continue

                    if text == "ping":
                        await ws.send("pong")
                        continue

                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        if text.lower() in ("ping", "pong"):
                            continue
                        print(f"Malformed message: not valid JSON ({text[:80]!r})")
                        continue

                    if not isinstance(payload, dict):
                        print(f"Malformed message: expected JSON object, got {type(payload).__name__}")
                        continue

                    line = format_message(payload)
                    if line is None:
                        continue
                    print(line)
                    printed += 1

            finally:
                await ws.close()
                print("\nConnection closed cleanly.")

    except InvalidURI as exc:
        print(f"Connection failure: invalid WebSocket URI — {exc}")
        return 1
    except asyncio.TimeoutError:
        print("Connection failure: timed out opening WebSocket")
        return 1
    except ConnectionClosed as exc:
        print(f"Connection closed unexpectedly: code={exc.code} reason={exc.reason!r}")
        return 1
    except OSError as exc:
        print(f"Connection failure: network/DNS error — {exc}")
        return 1
    except WebSocketException as exc:
        print(f"Connection failure: WebSocket error — {exc}")
        return 1
    except Exception as exc:
        print(f"Unexpected error: {type(exc).__name__}: {exc}")
        return 1

    if printed == 0:
        print("No market-data messages received.")
        return 1
    return 0


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
