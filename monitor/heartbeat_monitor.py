"""
External heartbeat / event monitor for live_trading_loop.py.

Runs independently (not inside the trading loop). Polls heartbeat.txt and
live_trading.log, and alerts via Telegram on staleness, near-floor equity,
and new order placements.

Usage:
  python monitor/heartbeat_monitor.py          # continuous (default)
  python monitor/heartbeat_monitor.py --once   # startup ping + one check, then exit
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitor.telegram_alert import send_telegram_alert  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

HEARTBEAT_PATH = PROJECT_ROOT / "heartbeat.txt"
# Equity and order placement are logged by live_trading_loop.py here
# (not in ai_decisions.log / audit jsonl).
LIVE_LOG_PATH = PROJECT_ROOT / "live_trading.log"

CHECK_INTERVAL_S = int(os.environ.get("MONITOR_CHECK_INTERVAL_S", str(15 * 60)))
STALE_THRESHOLD_S = int(os.environ.get("MONITOR_STALE_THRESHOLD_S", str(90 * 60)))
EQUITY_NEAR_FLOOR = float(os.environ.get("MONITOR_EQUITY_NEAR_FLOOR", "950"))

EQUITY_RE = re.compile(
    r"equity=(?P<equity>\d+(?:\.\d+)?|UNKNOWN)\s+USDT",
    re.IGNORECASE,
)
# Matches live_trading_loop.place_entry_order success log line
ORDER_PLACED_RE = re.compile(
    r"(?P<symbol>[A-Z0-9]+):\s+PLACED\s+LIMIT\s+(?P<side>BUY|SELL)\s+(?P<details>.+)",
)

logger = logging.getLogger("heartbeat_monitor")


def _parse_heartbeat_ts(raw: str) -> datetime | None:
    text = raw.strip()
    if not text:
        return None
    # ISO-8601 from live_trading_loop.write_heartbeat (may end with +00:00 / Z)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        ts = datetime.fromisoformat(text)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _read_heartbeat() -> datetime | None:
    if not HEARTBEAT_PATH.exists():
        return None
    try:
        return _parse_heartbeat_ts(HEARTBEAT_PATH.read_text(encoding="utf-8"))
    except OSError as exc:
        logger.warning("Could not read %s: %s", HEARTBEAT_PATH, exc)
        return None


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _read_new_lines(path: Path, offset: int) -> tuple[list[str], int]:
    """
    Read new complete lines from ``path`` starting at byte ``offset``.
    Returns (lines, new_offset). Handles truncation/rotation by resetting to 0.
    """
    if not path.exists():
        return [], 0
    size = _file_size(path)
    if size < offset:
        # Log rotated or truncated
        offset = 0
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            chunk = handle.read()
            new_offset = handle.tell()
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return [], offset

    if not chunk:
        return [], new_offset

    text = chunk.decode("utf-8", errors="replace")
    # Keep a partial trailing line for next read by rolling offset back
    if not text.endswith("\n") and "\n" in text:
        last_nl = text.rfind("\n")
        complete = text[: last_nl + 1]
        incomplete_bytes = len(text[last_nl + 1 :].encode("utf-8"))
        new_offset -= incomplete_bytes
        text = complete
    elif not text.endswith("\n"):
        # Entire chunk is a partial line — wait for more
        return [], offset

    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines, new_offset


class MonitorState:
    def __init__(self) -> None:
        self.stale_alerted = False
        self.equity_near_floor_alerted = False
        # Start at EOF so we only alert on events after monitor boot
        self.log_offset = _file_size(LIVE_LOG_PATH)


def check_heartbeat(state: MonitorState) -> None:
    ts = _read_heartbeat()
    now = datetime.now(tz=timezone.utc)

    if ts is None:
        age_desc = "missing or unreadable"
        is_stale = True
        stamp = "unknown"
    else:
        age_s = (now - ts).total_seconds()
        is_stale = age_s > STALE_THRESHOLD_S
        stamp = ts.isoformat()
        age_desc = f"{age_s / 60:.1f} min old"

    if is_stale:
        if not state.stale_alerted:
            ok = send_telegram_alert(
                f"⚠️ Heartbeat stale — last update was {stamp}, "
                f"loop may have crashed or stalled."
            )
            logger.warning(
                "Heartbeat STALE (%s); alert_sent=%s", age_desc, ok
            )
            state.stale_alerted = True
        else:
            logger.info("Heartbeat still stale (%s); suppressing repeat alert", age_desc)
    else:
        if state.stale_alerted:
            ok = send_telegram_alert("✅ Heartbeat recovered")
            logger.info("Heartbeat recovered (%s); recovery_alert_sent=%s", age_desc, ok)
        else:
            logger.info("Heartbeat fresh (%s)", age_desc)
        state.stale_alerted = False


def check_live_log(state: MonitorState) -> None:
    lines, state.log_offset = _read_new_lines(LIVE_LOG_PATH, state.log_offset)
    if not lines:
        return

    for line in lines:
        # Strip typical logging prefix: "2026-... INFO live_loop ..."
        # Match on message body anywhere in the line.
        equity_m = EQUITY_RE.search(line)
        if equity_m:
            raw = equity_m.group("equity")
            if raw.upper() != "UNKNOWN":
                try:
                    equity = float(raw)
                except ValueError:
                    equity = None
                if equity is not None:
                    if equity <= EQUITY_NEAR_FLOOR:
                        if not state.equity_near_floor_alerted:
                            ok = send_telegram_alert(
                                f"🚨 Equity near circuit breaker floor: {equity:.2f} USDT."
                            )
                            logger.warning(
                                "Equity near floor: %.2f; alert_sent=%s", equity, ok
                            )
                            state.equity_near_floor_alerted = True
                        else:
                            logger.info(
                                "Equity still near floor (%.2f); suppressing repeat",
                                equity,
                            )
                    else:
                        if state.equity_near_floor_alerted:
                            logger.info(
                                "Equity recovered above %.0f (now %.2f)",
                                EQUITY_NEAR_FLOOR,
                                equity,
                            )
                        state.equity_near_floor_alerted = False

        order_m = ORDER_PLACED_RE.search(line)
        if order_m:
            symbol = order_m.group("symbol")
            side = order_m.group("side")
            details = order_m.group("details").strip()
            ok = send_telegram_alert(
                f"📈 Trade executed: {symbol} {side} {details}"
            )
            logger.info(
                "Order placement detected %s %s; alert_sent=%s", symbol, side, ok
            )


def run_check(state: MonitorState) -> None:
    check_heartbeat(state)
    check_live_log(state)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="External Telegram monitor for live_trading_loop heartbeat/events"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Send startup ping, run one check cycle, then exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                PROJECT_ROOT / "heartbeat_monitor.log", encoding="utf-8"
            ),
        ],
    )

    logger.info(
        "Starting heartbeat monitor (check_interval=%ss stale_threshold=%ss "
        "equity_near_floor=%.0f) root=%s",
        CHECK_INTERVAL_S,
        STALE_THRESHOLD_S,
        EQUITY_NEAR_FLOOR,
        PROJECT_ROOT,
    )

    started = send_telegram_alert("✅ Heartbeat monitor started")
    if started:
        logger.info("Startup Telegram message sent")
    else:
        logger.warning(
            "Startup Telegram message FAILED — check TELEGRAM_BOT_TOKEN / "
            "TELEGRAM_CHAT_ID in .env"
        )

    state = MonitorState()

    if args.once:
        try:
            run_check(state)
        except Exception:
            logger.error(
                "Check failed with unhandled exception\n%s",
                traceback.format_exc(),
            )
            return 1
        logger.info("--once complete; exiting")
        return 0 if started else 1

    while True:
        try:
            run_check(state)
        except Exception:
            logger.error(
                "Check failed with unhandled exception; "
                "will wait for next interval and continue\n%s",
                traceback.format_exc(),
            )
        logger.info("sleeping %ss until next check (Ctrl+C to stop)", CHECK_INTERVAL_S)
        try:
            time.sleep(CHECK_INTERVAL_S)
        except KeyboardInterrupt:
            logger.info("Ctrl+C received - shutting down monitor cleanly")
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
