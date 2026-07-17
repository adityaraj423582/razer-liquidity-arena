"""
Liquidation-cascade reversal (long bounce) + walk-forward validation.

Detect violent down-hours (large drop + volume spike), long the next bar,
target a partial retracement of the cascade, tight stop under cascade low.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import itertools
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from backtesting.backtest_mean_reversion import (
    STARTING_CAPITAL,
    Candle,
    Trade,
    combine_equity,
    load_candles,
    max_drawdown_pct,
    sharpe_from_equity,
)

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

DROP_GRID = (0.03, 0.04, 0.05)  # 3%, 4%, 5%
# Grid asks for 2 volume / 2 retrace / 2 time values (3x2x2x2 = 24)
VOL_GRID = (2.0, 4.0)  # span of stated 2x/3x/4x tests
RETRACE_GRID = (0.40, 0.60)  # span of 40%/50%/60%
TIME_GRID = (8, 24)  # span of 8h/12h/24h
VOL_AVG_PERIOD = 24
FEE_RATE = 0.0005
TOP_N_SELECT = 5
CHUNK_DAYS = 60
ExitReason = Literal["take-profit", "stop-loss", "timeout", "end-of-data"]


@dataclass
class Params:
    drop: float
    vol_ratio: float
    retrace: float
    max_hold_hours: int

    def label(self) -> str:
        return (
            f"drop={self.drop * 100:.0f}% vol>={self.vol_ratio:.0f}x "
            f"retr={self.retrace * 100:.0f}% t={self.max_hold_hours}h"
        )


@dataclass
class PeriodMetrics:
    trades: int
    win_rate: float
    total_return_pct: float
    sharpe: float | None
    max_dd_pct: float
    cascade_events: int


def slice_candles(
    candles: list[Candle],
    start: datetime,
    end: datetime,
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

    return [
        ("A", a0, b0),
        ("B", b0, c0),
        ("C", c0, c1),
    ]


def avg_volume(volumes: list[float], index: int, period: int = VOL_AVG_PERIOD) -> float | None:
    if index < period - 1:
        return None
    window = volumes[index - period + 1 : index + 1]
    return sum(window) / period


def run_symbol_cascade(
    symbol: str,
    starting_cash: float,
    candles: list[Candle],
    params: Params,
) -> tuple[list[Trade], list[tuple[datetime, float]], int]:
    volumes = [c.volume for c in candles]
    cash = starting_cash
    equity_curve: list[tuple[datetime, float]] = []
    trades: list[Trade] = []
    cascade_events = 0

    in_pos = False
    entry_idx = -1
    entry_price = 0.0
    qty = 0.0
    entry_notional = 0.0
    entry_fee = 0.0
    cascade_low = 0.0
    tp_price = 0.0
    pending_entry_from: int | None = None  # cascade bar index

    for i, candle in enumerate(candles):
        mark = cash + (qty * candle.close if in_pos else 0.0)
        equity_curve.append((candle.ts, mark))

        # Enter at open of the bar immediately after a cascade
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
            reason: ExitReason | None = None
            # Intrabar-ish checks using high/low; fill assumptions:
            # stop hit if low breaches cascade low; TP if high reaches target
            if candle.low < cascade_low:
                reason = "stop-loss"
                # adverse fill at cascade_low (stop)
                exit_price = cascade_low
            elif candle.high >= tp_price:
                reason = "take-profit"
                exit_price = tp_price
            elif (candle.ts - candles[entry_idx].ts) >= timedelta(hours=params.max_hold_hours):
                reason = "timeout"
                exit_price = candle.close
            else:
                exit_price = candle.close

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

        # Cascade detection on this bar (only when flat / not pending)
        if i == 0:
            continue
        vol_avg = avg_volume(volumes, i)
        if vol_avg is None or vol_avg <= 0:
            continue

        prev = candles[i - 1]
        hourly_change = (candle.close - prev.close) / prev.close
        volume_ratio = candle.volume / vol_avg

        if hourly_change <= -params.drop and volume_ratio >= params.vol_ratio:
            cascade_events += 1
            # Cascade geometry for TP/SL
            cascade_low = candle.low
            cascade_ref_high = max(candle.open, prev.close)
            drop_size = cascade_ref_high - cascade_low
            if drop_size <= 0:
                # Degenerate candle; skip entry
                continue
            tp_price = cascade_low + params.retrace * drop_size
            if not in_pos:
                pending_entry_from = i

    if in_pos:
        last = candles[-1]
        exit_price = last.close
        exit_notional = qty * exit_price
        exit_fee = exit_notional * FEE_RATE
        pnl = (exit_notional - entry_notional) - entry_fee - exit_fee
        cash = entry_notional + pnl
        trades.append(
            Trade(
                symbol=symbol,
                entry_ts=candles[entry_idx].ts,
                exit_ts=last.ts,
                entry_price=entry_price,
                exit_price=exit_price,
                qty=qty,
                exit_reason="end-of-data",  # type: ignore[arg-type]
                pnl=pnl,
                return_pct=(pnl / entry_notional) * 100.0 if entry_notional else 0.0,
            )
        )
        equity_curve[-1] = (last.ts, cash)

    return trades, equity_curve, cascade_events


def run_period(
    candles_by_symbol: dict[str, list[Candle]],
    start: datetime,
    end: datetime,
    params: Params,
) -> PeriodMetrics:
    per_symbol_capital = STARTING_CAPITAL / len(SYMBOLS)
    all_trades: list[Trade] = []
    curves: dict[str, list[tuple[datetime, float]]] = {}
    end_equities: dict[str, float] = {}
    total_cascades = 0

    for symbol in SYMBOLS:
        sliced = slice_candles(candles_by_symbol[symbol], start, end)
        if len(sliced) < VOL_AVG_PERIOD + 2:
            raise RuntimeError(f"{symbol}: not enough bars in window")
        trades, curve, cascades = run_symbol_cascade(
            symbol, per_symbol_capital, sliced, params
        )
        all_trades.extend(trades)
        curves[symbol] = curve
        end_equities[symbol] = curve[-1][1] if curve else per_symbol_capital
        total_cascades += cascades

    combined = combine_equity(curves)
    equity_values = [v for _, v in combined]
    end_equity = sum(end_equities.values())
    n = len(all_trades)
    wins = sum(1 for t in all_trades if t.pnl > 0)
    return PeriodMetrics(
        trades=n,
        win_rate=(wins / n * 100.0) if n else 0.0,
        total_return_pct=(end_equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100.0,
        sharpe=sharpe_from_equity(equity_values),
        max_dd_pct=max_drawdown_pct(equity_values),
        cascade_events=total_cascades,
    )


def sharpe_sort_key(metrics: PeriodMetrics) -> float:
    s = metrics.sharpe
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return -999.0
    if math.isinf(s):
        return 999.0 if s > 0 else -999.0
    return s


def fmt_sharpe(value: float | None) -> str:
    if value is None:
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}"


def held_up(a: PeriodMetrics, out: PeriodMetrics) -> str:
    if out.trades == 0 and out.total_return_pct == 0:
        return "NO_TRADES"
    oos_sharpe = out.sharpe if out.sharpe is not None else -999.0
    if isinstance(oos_sharpe, float) and math.isinf(oos_sharpe):
        oos_sharpe = 999.0 if oos_sharpe > 0 else -999.0
    if out.total_return_pct <= 0 or oos_sharpe <= 0:
        return "COLLAPSED"
    a_sharpe = sharpe_sort_key(a)
    if out.total_return_pct < a.total_return_pct * 0.25 or oos_sharpe < a_sharpe * 0.25:
        return "WEAK"
    return "HELD"


def main() -> int:
    combos = [
        Params(drop=d, vol_ratio=v, retrace=r, max_hold_hours=t)
        for d, v, r, t in itertools.product(DROP_GRID, VOL_GRID, RETRACE_GRID, TIME_GRID)
    ]
    print("Walk-forward CASCADE REVERSAL (liquidation bounce)")
    print(f"Symbols ({len(SYMBOLS)}): {', '.join(SYMBOLS)}")
    print(
        f"Grid: drop={list(DROP_GRID)}, vol={list(VOL_GRID)}, "
        f"retrace={list(RETRACE_GRID)}, time={list(TIME_GRID)}h "
        f"-> {len(combos)} combos"
    )
    print(
        "Detect: hourly return <= -drop AND volume >= vol_ratio * 24h avg | "
        "Enter next bar open | TP=cascade_low+retrace*drop | SL < cascade low"
    )
    print(f"fee={FEE_RATE:.2%} per side | capital={STARTING_CAPITAL:.2f}")
    print(f"Top-{TOP_N_SELECT} on Period A Sharpe; validate on B and C\n")

    try:
        candles_by_symbol = {
            sym: load_candles(sym, min_bars=VOL_AVG_PERIOD + 5) for sym in SYMBOLS
        }
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        print("Run test_backtest_data.py first.")
        return 1

    full_start = max(c[0].ts for c in candles_by_symbol.values())
    full_end = min(c[-1].ts for c in candles_by_symbol.values()) + timedelta(hours=1)

    print("=== CASCADE EVENT SANITY (full ~180d) ===")
    # Event counts depend only on drop + vol (not exit params) — dedupe by those
    seen_detect: dict[tuple[float, float], int] = {}
    for params in combos:
        key = (params.drop, params.vol_ratio)
        if key in seen_detect:
            continue
        m = run_period(candles_by_symbol, full_start, full_end, params)
        seen_detect[key] = m.cascade_events
        print(
            f"  drop={params.drop * 100:.0f}% vol>={params.vol_ratio:.0f}x: "
            f"{m.cascade_events} cascade events "
            f"(~{m.cascade_events / 180:.2f}/day across all symbols)"
        )
    counts = list(seen_detect.values())
    print(
        f"\nCascade count range (by detect params): "
        f"min={min(counts)}, max={max(counts)}\n"
    )

    periods = make_periods(candles_by_symbol)
    for name, start, end in periods:
        n = len(slice_candles(candles_by_symbol["BTCUSDT"], start, end))
        print(
            f"Period {name}: {start.isoformat()} -> {end.isoformat()} "
            f"(~{n} hourly bars)"
        )
    print()

    period_a, period_b, period_c = periods

    print(f"Step 1-2: running all {len(combos)} combos on Period A...")
    a_results: list[tuple[Params, PeriodMetrics]] = []
    for i, params in enumerate(combos, start=1):
        metrics = run_period(
            candles_by_symbol, period_a[1], period_a[2], params
        )
        a_results.append((params, metrics))
        if i % 8 == 0 or i == len(combos):
            print(f"  Period A progress: {i}/{len(combos)}", flush=True)

    a_ranked = sorted(a_results, key=lambda item: sharpe_sort_key(item[1]), reverse=True)
    top = a_ranked[:TOP_N_SELECT]

    print(f"\nStep 3: Top {TOP_N_SELECT} by Period A Sharpe:")
    for rank, (params, metrics) in enumerate(top, start=1):
        print(
            f"  #{rank} {params.label()} | "
            f"A sharpe={fmt_sharpe(metrics.sharpe)} "
            f"ret={metrics.total_return_pct:.2f}% "
            f"trades={metrics.trades} cascades={metrics.cascade_events}"
        )

    print("\nStep 4: evaluating top combos on Period B and Period C...")
    rows: list[tuple[Params, PeriodMetrics, PeriodMetrics, PeriodMetrics]] = []
    for params, _ in top:
        a_m = run_period(candles_by_symbol, period_a[1], period_a[2], params)
        b_m = run_period(candles_by_symbol, period_b[1], period_b[2], params)
        c_m = run_period(candles_by_symbol, period_c[1], period_c[2], params)
        rows.append((params, a_m, b_m, c_m))

    print("\n=== WALK-FORWARD TABLE (Top 5 from Period A) ===\n")
    header = (
        f"{'#':>2} {'params':<42} "
        f"{'A_sharpe':>8} {'A_ret%':>8} "
        f"{'B_sharpe':>8} {'B_ret%':>8} {'B_flag':>10} "
        f"{'C_sharpe':>8} {'C_ret%':>8} {'C_flag':>10}"
    )
    print(header)
    print("-" * len(header))
    for i, (params, a_m, b_m, c_m) in enumerate(rows, start=1):
        print(
            f"{i:>2} {params.label():<42} "
            f"{fmt_sharpe(a_m.sharpe):>8} {a_m.total_return_pct:7.2f}% "
            f"{fmt_sharpe(b_m.sharpe):>8} {b_m.total_return_pct:7.2f}% {held_up(a_m, b_m):>10} "
            f"{fmt_sharpe(c_m.sharpe):>8} {c_m.total_return_pct:7.2f}% {held_up(a_m, c_m):>10}"
        )

    print("\nFlag legend: HELD / WEAK / COLLAPSED (same rules as prior walk-forwards).")

    strict_both = sum(
        1
        for _, a_m, b_m, c_m in rows
        if held_up(a_m, b_m) == "HELD" and held_up(a_m, c_m) == "HELD"
    )
    collapsed_any = sum(
        1
        for _, a_m, b_m, c_m in rows
        if held_up(a_m, b_m) == "COLLAPSED" or held_up(a_m, c_m) == "COLLAPSED"
    )
    avg_a = sum(a.total_return_pct for _, a, _, _ in rows) / len(rows)
    avg_b = sum(b.total_return_pct for _, _, b, _ in rows) / len(rows)
    avg_c = sum(c.total_return_pct for _, _, _, c in rows) / len(rows)
    a_positive = sum(
        1 for _, a, _, _ in rows if a.total_return_pct > 0 and sharpe_sort_key(a) > 0
    )

    print("\n=== BLUNT VERDICT ===")
    print(
        f"Of the top {TOP_N_SELECT} Period-A winners: both HELD={strict_both}; "
        f"collapsed on B or C={collapsed_any}."
    )
    print(
        f"Average return among top {TOP_N_SELECT}: "
        f"A={avg_a:.2f}% | B={avg_b:.2f}% | C={avg_c:.2f}%"
    )
    print(
        f"Period A quality: {a_positive}/{TOP_N_SELECT} had positive return AND positive Sharpe."
    )

    if a_positive == 0 and strict_both == 0:
        print(
            "\nALSO FAILS. Cascade-reversal did not produce a trustworthy edge "
            "in-sample or out-of-sample. Do not trade this as-is."
        )
    elif strict_both == 0:
        print(
            "\nDOES NOT HOLD UP OUT-OF-SAMPLE. Top Period-A cascade combos failed to "
            "HELD in both B and C. Do not trade this as-is."
        )
    else:
        print(
            f"\n{strict_both}/{TOP_N_SELECT} HELD in both B and C. "
            "Hint only - still verify further before live capital."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
