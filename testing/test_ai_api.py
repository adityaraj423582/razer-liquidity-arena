"""
Standalone helpers for the LTP-provided AI API.

Uses LTP_AI_API_KEY only (never trading credentials).

Usage:
    # Spends one tiny request — connectivity only:
    python testing/test_ai_api.py
    python testing/test_ai_api.py --connectivity

    # ZERO API calls — dump exact BTCUSDT prompt inputs + prompt text:
    python testing/test_ai_api.py --dump-prompt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path

from dotenv import load_dotenv

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Explicitly load project-root .env (do not rely on terminal/VS Code env injection).
load_dotenv(_ROOT / ".env")

from ai_agent import (  # noqa: E402
    KLINE_TAIL_BARS,
    build_regime_prompt,
    connectivity_test,
)
from live_trading_loop import fetch_recent_candles, fetch_recent_funding  # noqa: E402

PROMPT_DUMP_PATH = _ROOT / "debug_prompt_btcusdt.txt"


def _fmt_float(value: float) -> str:
    """Full Python float repr — no display rounding / truncation."""
    return repr(value)


def dump_btcusdt_prompt() -> int:
    """
    Fetch the same recent_klines / recent_funding the live path uses for BTCUSDT,
    print the exact last-24 candles and full funding series, build the prompt via
    build_regime_prompt() (no API call), save + print the complete prompt text.
    """
    symbol = "BTCUSDT"
    print(f"Fetching recent 1h klines + funding for {symbol} (same path as live loop)...")
    recent_klines = fetch_recent_candles(symbol)
    recent_funding = fetch_recent_funding(symbol)
    print(
        f"recent_klines length = {len(recent_klines)} "
        f"(full series passed into backend; prompt uses last {KLINE_TAIL_BARS})"
    )
    print(f"recent_funding.points length = {len(recent_funding.points)}")
    print("NO API CALL — building prompt only via build_regime_prompt().\n")

    tail = recent_klines[-KLINE_TAIL_BARS:]
    print("=" * 72)
    print(f"LAST {len(tail)} HOURLY CANDLES (exact values from recent_klines tail)")
    print("=" * 72)
    print("index_in_series | timestamp | open | high | low | close | volume")
    base_idx = len(recent_klines) - len(tail)
    for offset, c in enumerate(tail):
        idx = base_idx + offset
        print(
            f"{idx} | {c.ts.isoformat()} | "
            f"{_fmt_float(c.open)} | {_fmt_float(c.high)} | {_fmt_float(c.low)} | "
            f"{_fmt_float(c.close)} | {_fmt_float(c.volume)}"
        )

    print("\n" + "=" * 72)
    print("FULL recent_funding.points (exact values passed to backend)")
    print("=" * 72)
    print("i | timestamp | fundingRate")
    for i, p in enumerate(recent_funding.points):
        print(f"{i} | {p.ts.isoformat()} | {_fmt_float(p.rate)}")

    prompt = build_regime_prompt(symbol, recent_klines, recent_funding)
    PROMPT_DUMP_PATH.write_text(prompt, encoding="utf-8")
    print("\n" + "=" * 72)
    print(f"EXACT FINAL PROMPT STRING (also saved to {PROMPT_DUMP_PATH.name})")
    print("=" * 72)
    print(prompt)
    print("=" * 72)
    print(f"\nSaved {len(prompt)} chars to {PROMPT_DUMP_PATH.resolve()}")
    print("Stopped before any API send. Quota used: 0.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LTP AI API test helpers")
    parser.add_argument(
        "--connectivity",
        action="store_true",
        help="send one minimal real API request (uses quota)",
    )
    parser.add_argument(
        "--dump-prompt",
        action="store_true",
        help="print/save exact BTCUSDT regime prompt inputs; NO API call",
    )
    args = parser.parse_args()

    if args.dump_prompt:
        return dump_btcusdt_prompt()
    # Default / --connectivity: one minimal real request (prior behavior)
    return connectivity_test()


if __name__ == "__main__":
    raise SystemExit(main())
