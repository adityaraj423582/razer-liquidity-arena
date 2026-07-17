# Live loop: cascade+funding opportunistic longs on LTP (RAZERDEMO).
# Continuous unattended hourly loop by default (matches 1h candles). Pass --once for a single supervised iteration.
# Optional LIVE_LOOP_INTERVAL_S env override is for local stress tests only; production default remains 3600.

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

import requests
import websockets

from ai_agent import get_regime_assessment
from backtesting.backtest_funding_signal import FundingAsOf, FundingPoint
from strategy import (
    CIRCUIT_BREAKER,
    RISK_FRACTION,
    STOP_LOSS,
    TAKE_PROFIT,
    Candle,
    calculate_position_size,
    check_circuit_breaker,
    check_entry_signal,
)
from testing.test_marketdata import _decode_frame
from testing.test_order_lifecycle import api_request, load_credentials

BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
WS_URL = "wss://mds.ltp-contest.com/marketdata/v2/public"
SYMBOL_MAP = {
    "BTCUSDT": "BINANCE_PERP_BTC_USDT",
    "ETHUSDT": "BINANCE_PERP_ETH_USDT",
}
# Hourly: fetch_recent_candles uses interval="1h"; sleep matches that bar cadence.
LOOP_INTERVAL_S = int(os.environ.get("LIVE_LOOP_INTERVAL_S", "3600"))
HEARTBEAT_PATH = Path("heartbeat.txt")
# 30 days of hourly bars: enough for the AI regime vol percentile; signal uses the tail
HISTORY_BARS = 720
FUNDING_HISTORY_DAYS = 30
LEVERAGE = "2"
TERMINAL_ORDER_STATES = {"CANCELLED", "CANCELED", "REJECTED", "EXPIRED"}

logger = logging.getLogger("live_loop")


def write_heartbeat() -> None:
    """Overwrite heartbeat.txt with the current UTC timestamp (liveness signal)."""
    stamp = datetime.now(tz=timezone.utc).isoformat()
    HEARTBEAT_PATH.write_text(stamp + "\n", encoding="utf-8")


@dataclass
class SymInfo:
    lot_size: Decimal
    tick_size: Decimal
    min_size: Decimal


@dataclass
class Context:
    access_key: str
    secret_key: str
    api_host: str
    sym_info: dict[str, SymInfo]
    # binance_sym -> orderId of the entry we placed (blocks re-entry until terminal)
    open_positions: dict[str, str] = field(default_factory=dict)


def round_down(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def fmt_dec(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def fetch_recent_candles(binance_sym: str) -> list[Candle]:
    params = {"symbol": binance_sym, "interval": "1h", "limit": HISTORY_BARS}
    response = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
    response.raise_for_status()
    rows = response.json()
    candles = []
    for row in rows:
        candles.append(
            Candle(
                ts=datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
        )
    return candles


def fetch_recent_funding(binance_sym: str) -> FundingAsOf:
    start_ms = int(
        (datetime.now(tz=timezone.utc) - timedelta(days=FUNDING_HISTORY_DAYS)).timestamp() * 1000
    )
    params = {"symbol": binance_sym, "startTime": start_ms, "limit": 1000}
    response = requests.get(BINANCE_FUNDING_URL, params=params, timeout=30)
    response.raise_for_status()
    points = [
        FundingPoint(
            ts=datetime.fromtimestamp(int(row["fundingTime"]) / 1000, tz=timezone.utc),
            rate=float(row["fundingRate"]),
        )
        for row in response.json()
    ]
    points.sort(key=lambda p: p.ts)
    return FundingAsOf(points)


async def fetch_bbo(ltp_sym: str, timeout_s: float = 15.0) -> tuple[Decimal, Decimal]:
    subscribe = {"event": "subscribe", "arg": [{"channel": "BBO", "sym": ltp_sym}]}
    deadline = asyncio.get_running_loop().time() + timeout_s
    async with websockets.connect(WS_URL, open_timeout=10, close_timeout=5) as ws:
        await ws.send(json.dumps(subscribe))
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise RuntimeError(f"{ltp_sym}: timed out waiting for BBO")
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            text = _decode_frame(raw)
            if text == "ping":
                await ws.send("pong")
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            arg = payload.get("arg") if isinstance(payload, dict) else None
            if not isinstance(arg, dict) or arg.get("channel") != "BBO":
                continue
            data = payload.get("data")
            if isinstance(data, list) and data:
                data = data[0]
            if not isinstance(data, dict) or data.get("bid") is None:
                continue
            return Decimal(str(data["bid"])), Decimal(str(data["ask"]))


def fetch_sym_info(ctx_keys: tuple[str, str, str], ltp_sym: str) -> SymInfo:
    access_key, secret_key, api_host = ctx_keys
    payload = api_request(
        "GET", "api/v1/trading/sym/info", access_key, secret_key, api_host, {"sym": ltp_sym}
    )
    data = payload.get("data") or {}
    info = data.get(ltp_sym) if isinstance(data, dict) else None
    if not isinstance(info, dict) and isinstance(data, dict) and data.get("sym") == ltp_sym:
        info = data
    if not isinstance(info, dict):
        raise RuntimeError(f"sym/info missing entry for {ltp_sym}")
    return SymInfo(
        lot_size=Decimal(str(info["lotSize"])),
        tick_size=Decimal(str(info["tickSize"])),
        min_size=Decimal(str(info["minSize"])),
    )


def fetch_equity(ctx: Context) -> float | None:
    payload = api_request(
        "GET", "api/v1/trading/account", ctx.access_key, ctx.secret_key, ctx.api_host, {}
    )
    if payload.get("code") not in (200000, "200000"):
        logger.error(f"Account query failed: code={payload.get('code')} message={payload.get('message')}")
        return None
    data = payload.get("data")
    if not isinstance(data, list):
        logger.error(f"Account response data is {type(data).__name__}, expected list; treating breaker as ACTIVE")
        return None
    binance_entry = next(
        (e for e in data if isinstance(e, dict) and e.get("exchangeType") == "BINANCE"), None
    )
    if binance_entry is None:
        logger.error("No BINANCE entry in account data; treating breaker as ACTIVE")
        return None
    try:
        return float(binance_entry["equity"])
    except (KeyError, TypeError, ValueError) as exc:
        logger.error(f"Could not parse BINANCE equity field: {exc}; treating breaker as ACTIVE")
        return None


def reconcile_positions(ctx: Context) -> None:
    for binance_sym, order_id in list(ctx.open_positions.items()):
        payload = api_request(
            "GET", "api/v1/trading/order", ctx.access_key, ctx.secret_key, ctx.api_host,
            {"orderId": order_id},
        )
        data = payload.get("data") or {}
        state = str(data.get("orderState", "")).upper() if isinstance(data, dict) else ""
        if state in TERMINAL_ORDER_STATES:
            logger.info(f"{binance_sym}: entry order {order_id} is {state}; symbol unblocked")
            del ctx.open_positions[binance_sym]
        elif state == "FILLED":
            logger.info(
                f"{binance_sym}: entry order {order_id} FILLED; position managed by attached tp/sl "
                "(manual reconciliation until a positions endpoint is wired)"
            )
        elif state:
            logger.info(f"{binance_sym}: entry order {order_id} state={state}; symbol still blocked")


def place_entry_order(ctx: Context, binance_sym: str, equity: float) -> None:
    ltp_sym = SYMBOL_MAP[binance_sym]
    info = ctx.sym_info[ltp_sym]
    bid, _ask = asyncio.run(fetch_bbo(ltp_sym))

    notional, qty = calculate_position_size(equity, float(bid))
    qty_dec = round_down(Decimal(str(qty)), info.lot_size)
    if qty_dec < info.min_size:
        logger.info(f"{binance_sym}: sized qty {fmt_dec(qty_dec)} below minSize {fmt_dec(info.min_size)}; skipping")
        return

    limit_price = round_down(bid, info.tick_size)
    tp_trigger = round_down(limit_price * Decimal(str(1 + TAKE_PROFIT)), info.tick_size)
    sl_trigger = round_down(limit_price * Decimal(str(1 - STOP_LOSS)), info.tick_size)
    client_order_id = f"razerlive{int(time.time())}{uuid.uuid4().hex[:6]}".lower()

    params = {
        "sym": ltp_sym,
        "side": "BUY",
        "orderType": "LIMIT",
        "orderQty": fmt_dec(qty_dec),
        "limitPrice": fmt_dec(limit_price),
        "timeInForce": "GTC",
        "clientOrderId": client_order_id,
        "tpTriggerPrice": fmt_dec(tp_trigger),
        "tpPrice": "0",
        "slTriggerPrice": fmt_dec(sl_trigger),
        "slPrice": "0",
    }
    payload = api_request(
        "POST", "api/v1/trading/order", ctx.access_key, ctx.secret_key, ctx.api_host, params
    )
    code = payload.get("code")
    data = payload.get("data") or {}
    order_id = data.get("orderId") if isinstance(data, dict) else None
    if code in (200000, "200000") and order_id:
        ctx.open_positions[binance_sym] = str(order_id)
        logger.info(
            f"{binance_sym}: PLACED LIMIT BUY qty={fmt_dec(qty_dec)} @ {fmt_dec(limit_price)} "
            f"tp={fmt_dec(tp_trigger)} sl={fmt_dec(sl_trigger)} orderId={order_id} "
            f"(risk={notional * STOP_LOSS:.2f} USDT = {RISK_FRACTION:.0%} of equity)"
        )
    else:
        logger.error(f"{binance_sym}: order rejected: code={code} message={payload.get('message')}")


def run_iteration(ctx: Context) -> None:
    now = datetime.now(tz=timezone.utc).isoformat()
    logger.info(f"--- iteration start {now} ---")
    try:
        # (a) Circuit breaker first, every loop, no exceptions
        equity = fetch_equity(ctx)
        breaker_on = True if equity is None else check_circuit_breaker(equity)
        equity_text = f"{equity:.2f}" if equity is not None else "UNKNOWN"
        logger.info(
            f"equity={equity_text} USDT | breaker={'ACTIVE' if breaker_on else 'clear'} "
            f"(floor {CIRCUIT_BREAKER:.0f})"
        )

        if breaker_on:
            logger.info("CIRCUIT BREAKER ACTIVE - no new entries")
        else:
            # (b) AI regime gate then signal check, once per symbol per iteration
            for binance_sym in SYMBOL_MAP:
                candles = fetch_recent_candles(binance_sym)
                funding = fetch_recent_funding(binance_sym)
                assessment = get_regime_assessment(binance_sym, candles, funding)
                logger.info(
                    f"{binance_sym}: AI regime={assessment['decision']} ({assessment['reason']})"
                )
                if assessment["decision"] == "PAUSE":
                    logger.info(f"{binance_sym}: AI Agent PAUSE - skipping this symbol this iteration")
                    continue
                signal = check_entry_signal(candles, len(candles) - 1, funding)
                logger.info(f"{binance_sym}: signal={'YES' if signal else 'no'}")
                if not signal:
                    continue
                # (c) Only if flat on that symbol
                if binance_sym in ctx.open_positions:
                    logger.info(f"{binance_sym}: signal present but position/order already open; skipping")
                    continue
                place_entry_order(ctx, binance_sym, equity)

        # (d) Reconcile tracked orders
        reconcile_positions(ctx)
    finally:
        logger.info("--- iteration end ---")


def set_leverage(ctx_keys: tuple[str, str, str]) -> None:
    access_key, secret_key, api_host = ctx_keys
    for ltp_sym in SYMBOL_MAP.values():
        try:
            payload = api_request(
                "POST",
                "api/v1/trading/position/leverage",
                access_key,
                secret_key,
                api_host,
                {"sym": ltp_sym, "leverage": LEVERAGE},
            )
            code = payload.get("code")
            message = payload.get("message")
            if code in (200000, "200000"):
                logger.info(f"leverage set request accepted for {ltp_sym}: {LEVERAGE}x (code={code} message={message})")
            else:
                logger.warning(
                    f"leverage set for {ltp_sym} returned code={code} message={message} - "
                    "continuing (may already be set from a prior run)"
                )
        except Exception as exc:
            logger.warning(f"leverage set for {ltp_sym} failed: {exc} - continuing startup")


def startup(ctx_keys: tuple[str, str, str]) -> Context:
    access_key, secret_key, api_host = ctx_keys
    set_leverage(ctx_keys)
    sym_info = {
        SYMBOL_MAP[b]: fetch_sym_info(ctx_keys, SYMBOL_MAP[b]) for b in SYMBOL_MAP
    }
    ctx = Context(access_key=access_key, secret_key=secret_key, api_host=api_host, sym_info=sym_info)

    for binance_sym, ltp_sym in SYMBOL_MAP.items():
        bid, ask = asyncio.run(fetch_bbo(ltp_sym))
        logger.info(f"startup BBO {binance_sym} ({ltp_sym}): bid={fmt_dec(bid)} ask={fmt_dec(ask)}")

    equity = fetch_equity(ctx)
    equity_text = f"{equity:.2f}" if equity is not None else "UNKNOWN"
    logger.info(f"startup equity: {equity_text} USDT")
    return ctx


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LTP live loop (default: continuous hourly; RAZERDEMO only)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="run a single supervised iteration then exit (legacy mode)",
    )
    # Keep --loop as a no-op alias so older launch commands still work.
    parser.add_argument(
        "--loop",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("live_trading.log", encoding="utf-8")],
    )

    try:
        ctx_keys = load_credentials()
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1

    try:
        ctx = startup(ctx_keys)
    except Exception as exc:
        logger.error(f"Startup failed: {exc}")
        return 1

    if args.once:
        logger.info("running single supervised iteration (--once)")
    else:
        logger.info(
            f"entering continuous loop (interval={LOOP_INTERVAL_S}s, hourly candle cadence); "
            "Ctrl+C to stop"
        )

    try:
        while True:
            try:
                run_iteration(ctx)
            except Exception:
                logger.error(
                    "Iteration failed with unhandled exception; "
                    "will wait for next interval and continue\n%s",
                    traceback.format_exc(),
                )
            finally:
                # Heartbeat after success or caught failure — never block the loop.
                try:
                    write_heartbeat()
                except Exception:
                    logger.warning(
                        "Failed to write heartbeat.txt; continuing\n%s",
                        traceback.format_exc(),
                    )

            if args.once:
                break
            logger.info(f"sleeping {LOOP_INTERVAL_S}s until next iteration (Ctrl+C to stop)")
            time.sleep(LOOP_INTERVAL_S)
    except KeyboardInterrupt:
        logger.info("Ctrl+C received - shutting down cleanly (no open WebSocket connections held)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
