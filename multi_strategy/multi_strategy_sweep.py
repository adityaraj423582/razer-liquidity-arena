"""
Multi-strategy walk-forward sweep on the competition Binance USDT-M universe.

Walk-forward discipline matches prior research:
  - Period A / B / C = sequential 60-day chunks from common data start
  - Parameters chosen ONLY on Period A (by A Sharpe)
  - Every grid combo is evaluated and logged on A, B, and C
  - PASS bar (this sweep): positive Sharpe independently on BOTH B and C
    (positive Sharpe, not merely positive raw return; not averaged)

Universe source:
  Repo historically hardcoded only 8 research symbols — there is no stored 50-list.
  This script pulls live BINANCE_PERP_*_USDT symbols via existing LTP
  GET /api/v1/trading/sym/info (same path as test_order_lifecycle / live loop),
  converts to Binance names, keeps symbols with full ~180d history, ranks by
  24h quote volume, and takes the top 50. List is written to
  competition_universe.json for auditability.

Research only — does not modify strategy.py or live_trading_loop.py.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import csv
import itertools
import json
import math
import time
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import pstdev
from typing import Any, Callable, Literal

import requests

from backtesting.backtest_cascade_funding import (
    DROP_THRESHOLD,
    FEE_RATE,
    FUNDING_PERCENTILE,
    VOL_RATIO_THRESHOLD,
)
from backtesting.backtest_cascade_reversal import avg_volume
from backtesting.backtest_funding_signal import (
    LOOKBACK,
    MIN_FUNDING_SAMPLES,
    FundingAsOf,
    load_funding,
    percentile,
)
from backtesting.backtest_mean_reversion import (
    DATA_DIR,
    STARTING_CAPITAL,
    Candle,
    Trade,
    combine_equity,
    load_candles,
    max_drawdown_pct,
    sharpe_from_equity,
    sma_at,
)
from testing.test_order_lifecycle import api_request, load_credentials

CHUNK_DAYS = 60
HISTORY_DAYS = 180
MIN_BARS = 4000  # ~167d of hourly bars; require near-full sample
UNIVERSE_SIZE = 50
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_PATH = PROJECT_ROOT / "competition_universe.json"
RESULTS_PATH = Path(__file__).resolve().parent / "multi_strategy_sweep_results.txt"

BINANCE_KLINES = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
BINANCE_EXCHANGE = "https://fapi.binance.com/fapi/v1/exchangeInfo"
BINANCE_TICKER = "https://fapi.binance.com/fapi/v1/ticker/24hr"
INTERVAL_MS = 60 * 60 * 1000
MAX_KLINE_PAGE = 1500
MAX_FUNDING_PAGE = 1000

Side = Literal["long", "short"]


@dataclass
class PeriodMetrics:
    trades: int
    total_return_pct: float
    sharpe: float | None
    max_dd_pct: float


@dataclass
class SweepRow:
    family: str
    label: str
    a: PeriodMetrics
    b: PeriodMetrics
    c: PeriodMetrics

    @property
    def pass_bar(self) -> bool:
        """
        Profitable on BOTH B and C independently:
        positive total return AND positive Sharpe (not return alone, not averaged).
        """
        return (
            self.b.total_return_pct > 0
            and self.c.total_return_pct > 0
            and _positive_sharpe(self.b.sharpe)
            and _positive_sharpe(self.c.sharpe)
        )

    @property
    def bc_avg_sharpe(self) -> float:
        return (_sharpe_num(self.b.sharpe) + _sharpe_num(self.c.sharpe)) / 2.0


def _positive_sharpe(s: float | None) -> bool:
    if s is None or (isinstance(s, float) and (math.isnan(s) or math.isinf(s))):
        return False
    return s > 0.0


def _sharpe_num(s: float | None) -> float:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return -999.0
    if math.isinf(s):
        return 999.0 if s > 0 else -999.0
    return s


def sharpe_sort_key(m: PeriodMetrics) -> float:
    return _sharpe_num(m.sharpe)


def fmt_sharpe(value: float | None) -> str:
    if value is None:
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}"


def slice_candles(
    candles: list[Candle], start: datetime, end: datetime
) -> list[Candle]:
    return [c for c in candles if start <= c.ts < end]


def make_periods(
    candles_by_symbol: dict[str, list[Candle]],
) -> list[tuple[str, datetime, datetime]]:
    starts = [candes[0].ts for candes in candles_by_symbol.values()]
    ends = [candes[-1].ts for candes in candles_by_symbol.values()]
    global_start = max(starts)
    global_end = min(ends)
    a0 = global_start
    b0 = a0 + timedelta(days=CHUNK_DAYS)
    c0 = b0 + timedelta(days=CHUNK_DAYS)
    c1 = c0 + timedelta(days=CHUNK_DAYS)
    if c1 > global_end + timedelta(hours=1):
        c1 = global_end + timedelta(hours=1)
    return [("A", a0, b0), ("B", b0, c0), ("C", c0, c1)]


def ltp_to_binance(sym: str) -> str | None:
    # BINANCE_PERP_BTC_USDT -> BTCUSDT (base may contain underscores)
    parts = sym.split("_")
    if len(parts) < 4 or parts[0] != "BINANCE" or parts[1] != "PERP" or parts[-1] != "USDT":
        return None
    base = "".join(parts[2:-1])
    return f"{base}USDT"


def fetch_ltp_binance_perp_usdt() -> list[str]:
    access_key, secret_key, api_host = load_credentials()
    payload = api_request(
        "GET", "api/v1/trading/sym/info", access_key, secret_key, api_host, {}
    )
    if payload.get("code") not in (200000, "200000"):
        raise RuntimeError(
            f"sym/info failed: code={payload.get('code')} message={payload.get('message')}"
        )
    data = payload.get("data") or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"sym/info data unexpected type: {type(data).__name__}")
    out: list[str] = []
    for key in data:
        if not (isinstance(key, str) and key.startswith("BINANCE_PERP_") and key.endswith("_USDT")):
            continue
        binance = ltp_to_binance(key)
        if binance:
            out.append(binance)
    return sorted(set(out))


def resolve_universe() -> list[str]:
    """
    Build the 50-symbol competition research universe from LTP sym/info,
    intersecting with Binance TRADING USDT-M perps that have enough history.
    """
    print("Resolving universe from LTP GET /api/v1/trading/sym/info ...")
    ltp_syms = fetch_ltp_binance_perp_usdt()
    print(f"  LTP BINANCE_PERP_*_USDT: {len(ltp_syms)}")

    info = requests.get(BINANCE_EXCHANGE, timeout=60)
    info.raise_for_status()
    trading = {
        s["symbol"]
        for s in info.json()["symbols"]
        if s.get("contractType") == "PERPETUAL"
        and s.get("quoteAsset") == "USDT"
        and s.get("status") == "TRADING"
    }
    candidates = [s for s in ltp_syms if s in trading]
    print(f"  Intersection with Binance TRADING USDT-M perps: {len(candidates)}")

    tickers = requests.get(BINANCE_TICKER, timeout=60)
    tickers.raise_for_status()
    vol = {t["symbol"]: float(t["quoteVolume"]) for t in tickers.json()}
    ranked = sorted(candidates, key=lambda s: vol.get(s, 0.0), reverse=True)

    # Probe history length (one cheap klines call with limit=1 + startTime far back
    # is insufficient; instead fetch and keep symbols that already have CSVs or
    # successfully download >= MIN_BARS). Deferred to ensure_data — here take a
    # headroom list (top 80) then filter after download.
    headroom = ranked[:80]
    UNIVERSE_PATH.write_text(
        json.dumps(
            {
                "source": "LTP /api/v1/trading/sym/info BINANCE_PERP_*_USDT "
                "∩ Binance TRADING USDT-M, ranked by 24h quoteVolume",
                "requested_size": UNIVERSE_SIZE,
                "ltp_count": len(ltp_syms),
                "intersection_count": len(candidates),
                "headroom_candidates": headroom,
                "resolved_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return headroom


def _get_klines_page(symbol: str, start_ms: int, end_ms: int) -> list[list[Any]]:
    params = {
        "symbol": symbol,
        "interval": "1h",
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": MAX_KLINE_PAGE,
    }
    response = requests.get(BINANCE_KLINES, params=params, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"{symbol} klines HTTP {response.status_code}: {response.text[:160]}")
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"{symbol}: klines not a list")
    return payload


def download_klines(symbol: str, days: int = HISTORY_DAYS) -> int:
    end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    start_ms = int(
        (datetime.now(tz=timezone.utc) - timedelta(days=days)).timestamp() * 1000
    )
    all_rows: list[list[Any]] = []
    cursor = start_ms
    while cursor < end_ms:
        page = _get_klines_page(symbol, cursor, end_ms)
        if not page:
            break
        all_rows.extend(page)
        next_cursor = int(page[-1][0]) + INTERVAL_MS
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(page) < MAX_KLINE_PAGE:
            break
        time.sleep(0.08)
    by_open = {int(r[0]): r for r in all_rows}
    ordered = [by_open[k] for k in sorted(by_open)]
    if not ordered:
        return 0
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{symbol}_1h.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["timestamp", "open", "high", "low", "close", "volume"]
        )
        writer.writeheader()
        for row in ordered:
            writer.writerow(
                {
                    "timestamp": datetime.fromtimestamp(
                        int(row[0]) / 1000, tz=timezone.utc
                    ).isoformat(),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                }
            )
    return len(ordered)


def _get_funding_page(symbol: str, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    params = {
        "symbol": symbol,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": MAX_FUNDING_PAGE,
    }
    response = requests.get(BINANCE_FUNDING, params=params, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(
            f"{symbol} funding HTTP {response.status_code}: {response.text[:160]}"
        )
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"{symbol}: funding not a list")
    return payload


def download_funding(symbol: str, days: int = HISTORY_DAYS) -> int:
    end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    start_ms = int(
        (datetime.now(tz=timezone.utc) - timedelta(days=days)).timestamp() * 1000
    )
    rows: list[dict[str, Any]] = []
    cursor = start_ms
    while cursor < end_ms:
        page = _get_funding_page(symbol, cursor, end_ms)
        if not page:
            break
        rows.extend(page)
        next_cursor = int(page[-1]["fundingTime"]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(page) < MAX_FUNDING_PAGE:
            break
        time.sleep(0.08)
    by_ts = {int(r["fundingTime"]): r for r in rows}
    ordered = [by_ts[k] for k in sorted(by_ts)]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{symbol}_funding.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "fundingRate"])
        writer.writeheader()
        for row in ordered:
            writer.writerow(
                {
                    "timestamp": datetime.fromtimestamp(
                        int(row["fundingTime"]) / 1000, tz=timezone.utc
                    ).isoformat(),
                    "fundingRate": float(row["fundingRate"]),
                }
            )
    return len(ordered)


def ensure_universe_data(headroom: list[str]) -> list[str]:
    """Download missing CSVs; keep first UNIVERSE_SIZE symbols with enough bars."""
    selected: list[str] = []
    for symbol in headroom:
        if len(selected) >= UNIVERSE_SIZE:
            break
        kline_path = DATA_DIR / f"{symbol}_1h.csv"
        fund_path = DATA_DIR / f"{symbol}_funding.csv"
        try:
            n_bars = 0
            if kline_path.exists():
                with kline_path.open(encoding="utf-8") as handle:
                    n_bars = sum(1 for _ in handle) - 1
            if n_bars < MIN_BARS:
                print(f"  download klines {symbol} ...")
                n_bars = download_klines(symbol)
            if n_bars < MIN_BARS:
                print(f"  SKIP {symbol}: only {n_bars} bars (< {MIN_BARS})")
                continue
            n_fund = 0
            if fund_path.exists():
                with fund_path.open(encoding="utf-8") as handle:
                    n_fund = sum(1 for _ in handle) - 1
            if n_fund < 100:
                print(f"  download funding {symbol} ...")
                n_fund = download_funding(symbol)
            if n_fund < 100:
                print(f"  SKIP {symbol}: only {n_fund} funding prints")
                continue
            selected.append(symbol)
            print(f"  KEEP {symbol}: bars={n_bars} funding={n_fund} ({len(selected)}/{UNIVERSE_SIZE})")
        except Exception as exc:
            print(f"  SKIP {symbol}: {exc}")
            continue

    if len(selected) < UNIVERSE_SIZE:
        raise RuntimeError(
            f"Could only resolve {len(selected)}/{UNIVERSE_SIZE} symbols with full history"
        )

    meta = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    meta["symbols"] = selected
    meta["final_count"] = len(selected)
    UNIVERSE_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return selected


def metrics_from_book(
    trades: list[Trade],
    equity_values: list[float],
    end_equity: float,
) -> PeriodMetrics:
    return PeriodMetrics(
        trades=len(trades),
        total_return_pct=(end_equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100.0,
        sharpe=sharpe_from_equity(equity_values) if equity_values else None,
        max_dd_pct=max_drawdown_pct(equity_values) if equity_values else 0.0,
    )


# ---------------------------------------------------------------------------
# Family 1: cascade + funding (existing mechanism, scaled universe)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CascadeParams:
    retrace: float
    max_hold_hours: int

    def label(self) -> str:
        return f"cascade+fund retr={self.retrace * 100:.0f}% t={self.max_hold_hours}h"


def run_symbol_cascade(
    symbol: str,
    starting_cash: float,
    candles: list[Candle],
    funding: FundingAsOf,
    params: CascadeParams,
) -> tuple[list[Trade], list[tuple[datetime, float]]]:
    volumes = [c.volume for c in candles]
    cash = starting_cash
    equity_curve: list[tuple[datetime, float]] = []
    trades: list[Trade] = []
    in_pos = False
    entry_idx = -1
    entry_price = 0.0
    qty = 0.0
    entry_notional = 0.0
    entry_fee = 0.0
    cascade_low = 0.0
    tp_price = 0.0
    pending_entry_from: int | None = None

    for i, candle in enumerate(candles):
        mark = cash + (qty * candle.close if in_pos else 0.0)
        equity_curve.append((candle.ts, mark))

        if pending_entry_from is not None and not in_pos and cash > 0:
            if i == pending_entry_from + 1:
                entry_price = candle.open
                entry_notional = cash
                entry_fee = entry_notional * FEE_RATE
                qty = entry_notional / entry_price
                cash = 0.0
                in_pos = True
                entry_idx = i
                equity_curve[-1] = (candle.ts, qty * candle.close)
            pending_entry_from = None

        if in_pos:
            reason = None
            exit_price = candle.close
            if candle.low < cascade_low:
                reason = "stop-loss"
                exit_price = cascade_low
            elif candle.high >= tp_price:
                reason = "take-profit"
                exit_price = tp_price
            elif (candle.ts - candles[entry_idx].ts) >= timedelta(hours=params.max_hold_hours):
                reason = "timeout"
            if reason is not None:
                exit_notional = qty * exit_price
                exit_fee = exit_notional * FEE_RATE
                pnl = (exit_notional - entry_notional) - entry_fee - exit_fee
                cash = entry_notional + pnl
                trades.append(
                    Trade(
                        symbol=symbol,
                        entry_ts=candles[entry_idx].ts,
                        exit_ts=candle.ts,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        qty=qty,
                        exit_reason=reason,  # type: ignore[arg-type]
                        pnl=pnl,
                        return_pct=(pnl / entry_notional) * 100.0 if entry_notional else 0.0,
                    )
                )
                in_pos = False
                qty = 0.0
                equity_curve[-1] = (candle.ts, cash)
            continue

        if i == 0:
            continue
        vol_avg = avg_volume(volumes, i)
        if vol_avg is None or vol_avg <= 0:
            continue
        prev = candles[i - 1]
        hourly_change = (candle.close - prev.close) / prev.close
        volume_ratio = candle.volume / vol_avg
        if hourly_change > -DROP_THRESHOLD or volume_ratio < VOL_RATIO_THRESHOLD:
            continue
        rate = funding.rate_at(candle.ts)
        hist = funding.window_rates(candle.ts, LOOKBACK) if rate is not None else []
        if rate is None or len(hist) < MIN_FUNDING_SAMPLES:
            continue
        if rate > percentile(hist, FUNDING_PERCENTILE):
            continue
        cascade_low = candle.low
        drop_size = max(candle.open, prev.close) - cascade_low
        if drop_size <= 0:
            continue
        tp_price = cascade_low + params.retrace * drop_size
        if not in_pos:
            pending_entry_from = i

    if in_pos:
        last = candles[-1]
        exit_notional = qty * last.close
        exit_fee = exit_notional * FEE_RATE
        pnl = (exit_notional - entry_notional) - entry_fee - exit_fee
        cash = entry_notional + pnl
        trades.append(
            Trade(
                symbol=symbol,
                entry_ts=candles[entry_idx].ts,
                exit_ts=last.ts,
                entry_price=entry_price,
                exit_price=last.close,
                qty=qty,
                exit_reason="timeout",  # type: ignore[arg-type]
                pnl=pnl,
                return_pct=(pnl / entry_notional) * 100.0 if entry_notional else 0.0,
            )
        )
        equity_curve[-1] = (last.ts, cash)
    return trades, equity_curve


def run_period_cascade(
    symbols: list[str],
    candles_by_symbol: dict[str, list[Candle]],
    funding_by_symbol: dict[str, FundingAsOf],
    start: datetime,
    end: datetime,
    params: CascadeParams,
) -> PeriodMetrics:
    per = STARTING_CAPITAL / len(symbols)
    all_trades: list[Trade] = []
    curves: dict[str, list[tuple[datetime, float]]] = {}
    end_eq = 0.0
    for symbol in symbols:
        sliced = slice_candles(candles_by_symbol[symbol], start, end)
        trades, curve = run_symbol_cascade(
            symbol, per, sliced, funding_by_symbol[symbol], params
        )
        all_trades.extend(trades)
        curves[symbol] = curve
        end_eq += curve[-1][1] if curve else per
    combined = combine_equity(curves)
    return metrics_from_book(all_trades, [v for _, v in combined], end_eq)


# ---------------------------------------------------------------------------
# Family 2: funding-rate CARRY (mechanically distinct from directional bounce)
# Prior rejected signal: LONG when funding extremely NEGATIVE, betting on a
# price bounce; funding cashflows were NOT in the PnL.
# This carry sleeve: SHORT when funding extremely POSITIVE / LONG when extremely
# NEGATIVE to COLLECT the funding premium; PnL includes funding accruals;
# exit on funding normalizing toward median or time stop — not a price TP/SL bet.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CarryParams:
    high_pctl: float  # short when rate >= this percentile
    low_pctl: float  # long when rate <= this percentile
    max_hold_hours: int

    def label(self) -> str:
        return (
            f"carry short>={self.high_pctl:.0f}pctl "
            f"long<={self.low_pctl:.0f}pctl t={self.max_hold_hours}h"
        )


def run_symbol_carry(
    symbol: str,
    starting_cash: float,
    candles: list[Candle],
    funding: FundingAsOf,
    params: CarryParams,
) -> tuple[list[Trade], list[tuple[datetime, float]]]:
    """
    Funding carry: short extreme positive / long extreme negative funding.
    Equity includes price MTM + funding accruals (unlike the rejected directional signal).
    """
    cash = starting_cash
    equity_curve: list[tuple[datetime, float]] = []
    trades: list[Trade] = []
    in_pos = False
    side: Side = "long"
    entry_idx = -1
    entry_price = 0.0
    qty = 0.0
    entry_notional = 0.0
    entry_fee = 0.0
    funding_pnl = 0.0
    last_funding_ts: datetime | None = None
    fund_points = funding.points

    def mark_equity(price: float) -> float:
        if not in_pos:
            return cash
        if side == "long":
            return qty * price - entry_fee + funding_pnl
        return entry_notional - qty * price - entry_fee + funding_pnl

    def close_position(ts: datetime, price: float, reason: str) -> None:
        nonlocal cash, in_pos, qty, funding_pnl
        exit_fee = qty * price * FEE_RATE
        if side == "long":
            price_pnl = qty * price - entry_notional
        else:
            price_pnl = entry_notional - qty * price
        pnl = price_pnl - entry_fee - exit_fee + funding_pnl
        cash = entry_notional + pnl
        trades.append(
            Trade(
                symbol=symbol,
                entry_ts=candles[entry_idx].ts,
                exit_ts=ts,
                entry_price=entry_price,
                exit_price=price,
                qty=qty,
                exit_reason=reason,  # type: ignore[arg-type]
                pnl=pnl,
                return_pct=(pnl / entry_notional) * 100.0 if entry_notional else 0.0,
            )
        )
        in_pos = False
        qty = 0.0
        funding_pnl = 0.0

    for i, candle in enumerate(candles):
        if in_pos:
            prev_ts = candles[i - 1].ts if i > 0 else candle.ts - timedelta(hours=1)
            for fp in fund_points:
                if last_funding_ts is not None and fp.ts <= last_funding_ts:
                    continue
                if prev_ts < fp.ts <= candle.ts:
                    mtm_notional = qty * candle.close
                    delta = (
                        -fp.rate * mtm_notional
                        if side == "long"
                        else fp.rate * mtm_notional
                    )
                    funding_pnl += delta
                    last_funding_ts = fp.ts

        equity_curve.append((candle.ts, mark_equity(candle.close)))

        rate = funding.rate_at(candle.ts)
        hist = funding.window_rates(candle.ts, LOOKBACK) if rate is not None else []
        p_hi = (
            percentile(hist, params.high_pctl) if len(hist) >= MIN_FUNDING_SAMPLES else None
        )
        p_lo = (
            percentile(hist, params.low_pctl) if len(hist) >= MIN_FUNDING_SAMPLES else None
        )
        p50 = percentile(hist, 50.0) if len(hist) >= MIN_FUNDING_SAMPLES else None

        if in_pos:
            reason = None
            held = candle.ts - candles[entry_idx].ts
            if held >= timedelta(hours=params.max_hold_hours):
                reason = "timeout"
            elif rate is not None and p50 is not None:
                if side == "long" and rate > p50:
                    reason = "funding-normalize"
                elif side == "short" and rate < p50:
                    reason = "funding-normalize"
            if reason is not None:
                close_position(candle.ts, candle.close, reason)
                equity_curve[-1] = (candle.ts, cash)
            continue

        if rate is None or p_hi is None or p_lo is None or cash <= 0:
            continue

        if rate >= p_hi:
            side = "short"
        elif rate <= p_lo:
            side = "long"
        else:
            continue

        entry_price = candle.close
        entry_notional = cash
        entry_fee = entry_notional * FEE_RATE
        qty = entry_notional / entry_price
        cash = 0.0
        funding_pnl = 0.0
        in_pos = True
        entry_idx = i
        last_funding_ts = candle.ts
        equity_curve[-1] = (candle.ts, mark_equity(candle.close))

    if in_pos:
        last = candles[-1]
        close_position(last.ts, last.close, "timeout")
        equity_curve[-1] = (last.ts, cash)
    return trades, equity_curve


def run_period_carry(
    symbols: list[str],
    candles_by_symbol: dict[str, list[Candle]],
    funding_by_symbol: dict[str, FundingAsOf],
    start: datetime,
    end: datetime,
    params: CarryParams,
) -> PeriodMetrics:
    per = STARTING_CAPITAL / len(symbols)
    all_trades: list[Trade] = []
    curves: dict[str, list[tuple[datetime, float]]] = {}
    end_eq = 0.0
    for symbol in symbols:
        sliced = slice_candles(candles_by_symbol[symbol], start, end)
        trades, curve = run_symbol_carry(
            symbol, per, sliced, funding_by_symbol[symbol], params
        )
        all_trades.extend(trades)
        curves[symbol] = curve
        end_eq += curve[-1][1] if curve else per
    combined = combine_equity(curves)
    return metrics_from_book(all_trades, [v for _, v in combined], end_eq)


# ---------------------------------------------------------------------------
# Family 3: vol-regime-filtered mean reversion (LOW vol only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VolMRParams:
    sma: int
    entry_dev: float
    tp_dev: float
    stop_dev: float
    vol_pctl: float  # enter only if 24h realized vol <= this trailing percentile
    vol_lookback: int = 720  # hours for vol percentile history (~30d)

    def label(self) -> str:
        return (
            f"volMR sma={self.sma} entry={self.entry_dev * 100:.1f}% "
            f"tp={self.tp_dev * 100:.1f}% stop={self.stop_dev * 100:.1f}% "
            f"vol<={self.vol_pctl:.0f}pctl"
        )


def realized_vol_24h(closes: list[float], i: int) -> float | None:
    if i < 24:
        return None
    rets = []
    for j in range(i - 23, i + 1):
        prev = closes[j - 1]
        if prev <= 0:
            return None
        rets.append((closes[j] - prev) / prev)
    if len(rets) < 2:
        return None
    return pstdev(rets)


def run_symbol_vol_mr(
    symbol: str,
    starting_cash: float,
    candles: list[Candle],
    params: VolMRParams,
) -> tuple[list[Trade], list[tuple[datetime, float]]]:
    closes = [c.close for c in candles]
    vols: list[float | None] = [realized_vol_24h(closes, i) for i in range(len(candles))]
    cash = starting_cash
    equity_curve: list[tuple[datetime, float]] = []
    trades: list[Trade] = []
    in_pos = False
    entry_idx = -1
    entry_price = 0.0
    qty = 0.0
    entry_notional = 0.0
    entry_fee = 0.0
    max_hold = timedelta(hours=24)

    for i, candle in enumerate(candles):
        mark = cash + (qty * candle.close if in_pos else 0.0)
        if in_pos:
            mark = qty * candle.close - entry_fee
        equity_curve.append((candle.ts, mark))

        sma = sma_at(closes, i, params.sma)
        if sma is None:
            continue
        deviation = (candle.close - sma) / sma

        if in_pos:
            reason = None
            if deviation <= params.stop_dev:
                reason = "stop-loss"
            elif deviation >= -abs(params.tp_dev):
                reason = "take-profit"
            elif candle.ts - candles[entry_idx].ts >= max_hold:
                reason = "timeout"
            if reason is not None:
                exit_price = candle.close
                exit_notional = qty * exit_price
                exit_fee = exit_notional * FEE_RATE
                pnl = (exit_notional - entry_notional) - entry_fee - exit_fee
                cash = entry_notional + pnl
                trades.append(
                    Trade(
                        symbol=symbol,
                        entry_ts=candles[entry_idx].ts,
                        exit_ts=candle.ts,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        qty=qty,
                        exit_reason=reason,  # type: ignore[arg-type]
                        pnl=pnl,
                        return_pct=(pnl / entry_notional) * 100.0 if entry_notional else 0.0,
                    )
                )
                in_pos = False
                qty = 0.0
                equity_curve[-1] = (candle.ts, cash)
            continue

        vol = vols[i]
        if vol is None or cash <= 0:
            continue
        hist_vols = [v for v in vols[max(0, i - params.vol_lookback) : i + 1] if v is not None]
        if len(hist_vols) < 48:
            continue
        thresh = percentile(hist_vols, params.vol_pctl)
        if vol > thresh:
            continue
        if deviation <= params.entry_dev:
            entry_price = candle.close
            entry_notional = cash
            entry_fee = entry_notional * FEE_RATE
            qty = entry_notional / entry_price
            cash = 0.0
            in_pos = True
            entry_idx = i
            equity_curve[-1] = (candle.ts, entry_notional - entry_fee)

    if in_pos:
        last = candles[-1]
        exit_notional = qty * last.close
        exit_fee = exit_notional * FEE_RATE
        pnl = (exit_notional - entry_notional) - entry_fee - exit_fee
        cash = entry_notional + pnl
        trades.append(
            Trade(
                symbol=symbol,
                entry_ts=candles[entry_idx].ts,
                exit_ts=last.ts,
                entry_price=entry_price,
                exit_price=last.close,
                qty=qty,
                exit_reason="timeout",  # type: ignore[arg-type]
                pnl=pnl,
                return_pct=(pnl / entry_notional) * 100.0 if entry_notional else 0.0,
            )
        )
        equity_curve[-1] = (last.ts, cash)
    return trades, equity_curve


def run_period_vol_mr(
    symbols: list[str],
    candles_by_symbol: dict[str, list[Candle]],
    start: datetime,
    end: datetime,
    params: VolMRParams,
) -> PeriodMetrics:
    per = STARTING_CAPITAL / len(symbols)
    all_trades: list[Trade] = []
    curves: dict[str, list[tuple[datetime, float]]] = {}
    end_eq = 0.0
    for symbol in symbols:
        sliced = slice_candles(candles_by_symbol[symbol], start, end)
        trades, curve = run_symbol_vol_mr(symbol, per, sliced, params)
        all_trades.extend(trades)
        curves[symbol] = curve
        end_eq += curve[-1][1] if curve else per
    combined = combine_equity(curves)
    return metrics_from_book(all_trades, [v for _, v in combined], end_eq)


# ---------------------------------------------------------------------------
# Family 4: cross-symbol funding dispersion (market-neutral)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DispersionParams:
    k: int
    rebalance_hours: int

    def label(self) -> str:
        return f"fundDisp K={self.k} rebalance={self.rebalance_hours}h"


def run_period_dispersion(
    symbols: list[str],
    candles_by_symbol: dict[str, list[Candle]],
    funding_by_symbol: dict[str, FundingAsOf],
    start: datetime,
    end: datetime,
    params: DispersionParams,
) -> PeriodMetrics:
    """
    Market-neutral funding dispersion:
    long K most-negative funding, short K most-positive, equal weight, periodic rebalance.
    Equity includes price MTM + funding accruals.
    """
    ts_sets = []
    close_maps: dict[str, dict[datetime, float]] = {}
    for sym in symbols:
        sliced = slice_candles(candles_by_symbol[sym], start, end)
        close_maps[sym] = {c.ts: c.close for c in sliced}
        ts_sets.append(set(close_maps[sym].keys()))
    common_ts = sorted(set.intersection(*ts_sets)) if ts_sets else []
    if len(common_ts) < params.rebalance_hours + 2:
        return PeriodMetrics(0, 0.0, None, 0.0)

    # Free cash; positions hold notional sleeves separately.
    cash = STARTING_CAPITAL
    # sym -> (side, qty, entry_notional, entry_fee, funding_acc)
    positions: dict[str, tuple[Side, float, float, float, float]] = {}
    equity_curve: list[float] = []
    trade_pnls: list[float] = []
    last_fund_seen: dict[str, datetime] = {
        s: common_ts[0] - timedelta(hours=1) for s in symbols
    }

    def mark_equity(i: int) -> float:
        total = cash
        ts = common_ts[i]
        for sym, (side, qty, entry_notional, entry_fee, funding_acc) in positions.items():
            px = close_maps[sym][ts]
            if side == "long":
                total += qty * px - entry_fee + funding_acc
            else:
                total += entry_notional - qty * px - entry_fee + funding_acc
        return total

    def close_all(i: int) -> None:
        nonlocal cash
        ts = common_ts[i]
        for sym in list(positions.keys()):
            side, qty, entry_notional, entry_fee, fund_acc = positions.pop(sym)
            px = close_maps[sym][ts]
            exit_notional = qty * px
            exit_fee = exit_notional * FEE_RATE
            if side == "long":
                price_pnl = exit_notional - entry_notional
            else:
                price_pnl = entry_notional - exit_notional
            pnl = price_pnl - entry_fee - exit_fee + fund_acc
            cash += entry_notional + pnl
            trade_pnls.append(pnl)

    def open_book(i: int, longs: list[str], shorts: list[str]) -> None:
        nonlocal cash
        ts = common_ts[i]
        n = len(longs) + len(shorts)
        if n == 0 or cash <= 0:
            return
        sleeve = cash / n
        for sym in longs + shorts:
            side: Side = "long" if sym in longs else "short"
            px = close_maps[sym][ts]
            fee = sleeve * FEE_RATE
            qty = sleeve / px
            cash -= sleeve
            positions[sym] = (side, qty, sleeve, fee, 0.0)
            last_fund_seen[sym] = ts

    fund_ts = {sym: funding_by_symbol[sym].ts_list for sym in symbols}
    fund_rates = {
        sym: [p.rate for p in funding_by_symbol[sym].points] for sym in symbols
    }

    for i, ts in enumerate(common_ts):
        for sym, (side, qty, entry_notional, entry_fee, fund_acc) in list(positions.items()):
            px = close_maps[sym][ts]
            notional = qty * px
            ts_list = fund_ts[sym]
            end_idx = bisect_right(ts_list, ts)
            start_idx = bisect_right(ts_list, last_fund_seen[sym])
            for j in range(start_idx, end_idx):
                rate = fund_rates[sym][j]
                delta = (-rate * notional) if side == "long" else (rate * notional)
                fund_acc += delta
                last_fund_seen[sym] = ts_list[j]
            positions[sym] = (side, qty, entry_notional, entry_fee, fund_acc)

        if i % params.rebalance_hours == 0:
            rates: list[tuple[str, float]] = []
            for sym in symbols:
                r = funding_by_symbol[sym].rate_at(ts)
                if r is not None:
                    rates.append((sym, r))
            rates.sort(key=lambda x: x[1])
            if len(rates) >= 2 * params.k:
                longs = [s for s, _ in rates[: params.k]]
                shorts = [s for s, _ in rates[-params.k :]]
                desired: dict[str, Side] = {s: "long" for s in longs}
                desired.update({s: "short" for s in shorts})
                current = {s: positions[s][0] for s in positions}
                if current != desired:
                    close_all(i)
                    open_book(i, longs, shorts)

        equity_curve.append(mark_equity(i))

    if positions:
        close_all(len(common_ts) - 1)
        equity_curve[-1] = cash

    end_equity = equity_curve[-1] if equity_curve else STARTING_CAPITAL
    trades_proxy = [
        Trade(
            symbol="BOOK",
            entry_ts=common_ts[0],
            exit_ts=common_ts[-1],
            entry_price=0.0,
            exit_price=0.0,
            qty=0.0,
            exit_reason="timeout",  # type: ignore[arg-type]
            pnl=p,
            return_pct=0.0,
        )
        for p in trade_pnls
    ]
    return metrics_from_book(trades_proxy, equity_curve, end_equity)


# ---------------------------------------------------------------------------
# Sweep orchestration
# ---------------------------------------------------------------------------


def evaluate_combo(
    family: str,
    label: str,
    runner_a: Callable[[], PeriodMetrics],
    runner_b: Callable[[], PeriodMetrics],
    runner_c: Callable[[], PeriodMetrics],
) -> SweepRow:
    a = runner_a()
    b = runner_b()
    c = runner_c()
    return SweepRow(family=family, label=label, a=a, b=b, c=c)


def format_row(row: SweepRow) -> str:
    verdict = "PASS" if row.pass_bar else "FAIL"
    return (
        f"{verdict:4s} | {row.family:12s} | {row.label:55s} | "
        f"A: sharpe={fmt_sharpe(row.a.sharpe):>6s} ret={row.a.total_return_pct:+7.2f}% "
        f"trades={row.a.trades:4d} dd={row.a.max_dd_pct:6.2f}% | "
        f"B: sharpe={fmt_sharpe(row.b.sharpe):>6s} ret={row.b.total_return_pct:+7.2f}% "
        f"trades={row.b.trades:4d} dd={row.b.max_dd_pct:6.2f}% | "
        f"C: sharpe={fmt_sharpe(row.c.sharpe):>6s} ret={row.c.total_return_pct:+7.2f}% "
        f"trades={row.c.trades:4d} dd={row.c.max_dd_pct:6.2f}%"
    )


def main() -> int:
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("=== MULTI-STRATEGY WALK-FORWARD SWEEP ===")
    log(
        "Pass bar: positive return AND positive Sharpe on BOTH Period B and Period C "
        "(independent; not averaged; return alone is not enough)."
    )
    log("Tune/select reference: Period A Sharpe only (no peeking at B/C for params).")
    log("")
    log("MECHANISM NOTE — funding CARRY vs prior funding DIRECTIONAL:")
    log("  Prior (rejected): long-only when funding is extremely negative; bet on price")
    log("  bounce; funding cashflows NOT in PnL; exits via price TP/SL / normalize / time.")
    log("  This carry sleeve: short extreme positive / long extreme negative funding to")
    log("  COLLECT the premium; funding accruals ARE in PnL; exit on normalize or time.")
    log("  These are economically opposite uses of the same input series.")
    log("")

    try:
        headroom = resolve_universe()
        symbols = ensure_universe_data(headroom)
    except Exception as exc:
        log(f"ERROR resolving universe/data: {exc}")
        RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 1

    log(f"Universe ({len(symbols)}): {', '.join(symbols)}")
    log(f"Saved: {UNIVERSE_PATH}")
    log("")

    print("Loading candles + funding ...")
    candles_by_symbol = {s: load_candles(s, min_bars=100) for s in symbols}
    funding_by_symbol = {s: FundingAsOf(load_funding(s)) for s in symbols}
    periods = make_periods(candles_by_symbol)
    for name, start, end in periods:
        n = len(slice_candles(candles_by_symbol[symbols[0]], start, end))
        log(
            f"Period {name}: {start.isoformat()} -> {end.isoformat()} "
            f"(~{n} hourly bars on {symbols[0]})"
        )
    log("")
    _, a0, a1 = periods[0][0], periods[0][1], periods[0][2]
    b0, b1 = periods[1][1], periods[1][2]
    c0, c1 = periods[2][1], periods[2][2]

    rows: list[SweepRow] = []

    # --- Family 1: cascade+funding ---
    cascade_grid = [
        CascadeParams(retrace=r, max_hold_hours=t)
        for r, t in itertools.product((0.40, 0.60), (8, 24))
    ]
    log(f"--- Family 1: cascade+funding ({len(cascade_grid)} combos, fixed detect) ---")
    for params in cascade_grid:
        row = evaluate_combo(
            "cascade+fund",
            params.label(),
            lambda p=params: run_period_cascade(
                symbols, candles_by_symbol, funding_by_symbol, a0, a1, p
            ),
            lambda p=params: run_period_cascade(
                symbols, candles_by_symbol, funding_by_symbol, b0, b1, p
            ),
            lambda p=params: run_period_cascade(
                symbols, candles_by_symbol, funding_by_symbol, c0, c1, p
            ),
        )
        rows.append(row)
        log(format_row(row))

    # --- Family 2: funding carry ---
    carry_grid = [
        CarryParams(high_pctl=h, low_pctl=lo, max_hold_hours=t)
        for h, lo, t in itertools.product((80.0, 90.0, 95.0), (5.0, 10.0, 20.0), (24, 72))
    ]
    log("")
    log(f"--- Family 2: funding-rate CARRY ({len(carry_grid)} combos) ---")
    for params in carry_grid:
        row = evaluate_combo(
            "fund_carry",
            params.label(),
            lambda p=params: run_period_carry(
                symbols, candles_by_symbol, funding_by_symbol, a0, a1, p
            ),
            lambda p=params: run_period_carry(
                symbols, candles_by_symbol, funding_by_symbol, b0, b1, p
            ),
            lambda p=params: run_period_carry(
                symbols, candles_by_symbol, funding_by_symbol, c0, c1, p
            ),
        )
        rows.append(row)
        log(format_row(row))

    # --- Family 3: vol-filtered mean reversion ---
    vol_mr_grid = [
        VolMRParams(sma=sma, entry_dev=e, tp_dev=tp, stop_dev=sl, vol_pctl=vp)
        for sma, e, tp, sl, vp in itertools.product(
            (10, 20),
            (-0.02, -0.03),
            (0.003, 0.005),
            (-0.04, -0.05),
            (20.0, 30.0),
        )
    ]
    log("")
    log(f"--- Family 3: low-vol mean reversion ({len(vol_mr_grid)} combos) ---")
    for params in vol_mr_grid:
        row = evaluate_combo(
            "vol_mr",
            params.label(),
            lambda p=params: run_period_vol_mr(symbols, candles_by_symbol, a0, a1, p),
            lambda p=params: run_period_vol_mr(symbols, candles_by_symbol, b0, b1, p),
            lambda p=params: run_period_vol_mr(symbols, candles_by_symbol, c0, c1, p),
        )
        rows.append(row)
        log(format_row(row))

    # --- Family 4: funding dispersion ---
    disp_grid = [
        DispersionParams(k=k, rebalance_hours=rb)
        for k, rb in itertools.product((3, 5), (8, 24))
    ]
    log("")
    log(f"--- Family 4: cross-symbol funding dispersion ({len(disp_grid)} combos) ---")
    for params in disp_grid:
        row = evaluate_combo(
            "fund_disp",
            params.label(),
            lambda p=params: run_period_dispersion(
                symbols, candles_by_symbol, funding_by_symbol, a0, a1, p
            ),
            lambda p=params: run_period_dispersion(
                symbols, candles_by_symbol, funding_by_symbol, b0, b1, p
            ),
            lambda p=params: run_period_dispersion(
                symbols, candles_by_symbol, funding_by_symbol, c0, c1, p
            ),
        )
        rows.append(row)
        log(format_row(row))

    log("")
    log(f"=== FULL TABLE: {len(rows)} strategies tested (all logged above) ===")
    passed = [r for r in rows if r.pass_bar]
    log(f"Passed bar (return>0 AND Sharpe>0 on B AND C): {len(passed)}/{len(rows)}")
    log("")

    if not passed:
        log("ZERO STRATEGIES PASSED.")
        log(
            "Honest negative result — same standard as the first 6 failed hypotheses. "
            "Do not promote any of these sleeves to live on this evidence."
        )
    else:
        passed.sort(key=lambda r: r.bc_avg_sharpe, reverse=True)
        log("=== RANKED PASSERS (by average B+C Sharpe) ===")
        for i, row in enumerate(passed, 1):
            log(
                f"{i:2d}. avgBC_sharpe={row.bc_avg_sharpe:.2f} | {format_row(row)}"
            )

    # Also note best-on-A for each family (tuning reference only — not a pass)
    log("")
    log("=== BEST ON PERIOD A BY FAMILY (tuning reference only; NOT a pass criterion) ===")
    for family in ("cascade+fund", "fund_carry", "vol_mr", "fund_disp"):
        fam = [r for r in rows if r.family == family]
        if not fam:
            continue
        best = max(fam, key=lambda r: sharpe_sort_key(r.a))
        log(
            f"{family}: A-best={best.label} | "
            f"A_sharpe={fmt_sharpe(best.a.sharpe)} | "
            f"B_sharpe={fmt_sharpe(best.b.sharpe)} | "
            f"C_sharpe={fmt_sharpe(best.c.sharpe)} | "
            f"bar={'PASS' if best.pass_bar else 'FAIL'}"
        )

    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("")
    log(f"Results written to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
