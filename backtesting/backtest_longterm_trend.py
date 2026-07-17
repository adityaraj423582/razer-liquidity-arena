"""
Long-horizon trend following + walk-forward validation.

Entry: price > fast SMA and price > slow SMA and fast > slow.
Exit: close below slow SMA, optional 15% trailing stop from peak.
No time stop. Small 4-combo grid (SMA pair x trail on/off).
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

# Hourly bars: N-day SMA = N * 24 periods
SMA_PAIRS = (
    (7 * 24, 14 * 24),   # 7d / 14d
    (10 * 24, 20 * 24),  # 10d / 20d
)
TRAIL_OPTIONS = (False, True)
TRAIL_PCT = 0.15
FEE_RATE = 0.0005
CHUNK_DAYS = 60
ExitReason = Literal["trend-break", "trailing-stop", "end-of-data"]


@dataclass
class Params:
    fast: int
    slow: int
    use_trail: bool

    def label(self) -> str:
        fast_d = self.fast // 24
        slow_d = self.slow // 24
        trail = "trail15%" if self.use_trail else "no-trail"
        return f"SMA {fast_d}d/{slow_d}d {trail}"


@dataclass
class PeriodMetrics:
    trades: int
    win_rate: float
    total_return_pct: float
    sharpe: float | None
    max_dd_pct: float
    time_in_pos_pct: dict[str, float]


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


def sma_at(closes: list[float], index: int, period: int) -> float | None:
    if index < period - 1:
        return None
    window = closes[index - period + 1 : index + 1]
    return sum(window) / period


def run_symbol_trend(
    symbol: str,
    starting_cash: float,
    candles: list[Candle],
    params: Params,
) -> tuple[list[Trade], list[tuple[datetime, float]], float]:
    closes = [c.close for c in candles]
    cash = starting_cash
    equity_curve: list[tuple[datetime, float]] = []
    trades: list[Trade] = []

    in_pos = False
    entry_idx = -1
    entry_price = 0.0
    qty = 0.0
    entry_notional = 0.0
    entry_fee = 0.0
    peak_price = 0.0
    bars_in_pos = 0

    for i, candle in enumerate(candles):
        if in_pos:
            bars_in_pos += 1
            mark = qty * candle.close
        else:
            mark = cash
        equity_curve.append((candle.ts, mark))

        fast = sma_at(closes, i, params.fast)
        slow = sma_at(closes, i, params.slow)
        if fast is None or slow is None:
            continue

        if in_pos:
            if candle.close > peak_price:
                peak_price = candle.close

            reason: ExitReason | None = None
            if candle.close < slow:
                reason = "trend-break"
            elif params.use_trail and peak_price > 0:
                dd = (peak_price - candle.close) / peak_price
                if dd >= TRAIL_PCT:
                    reason = "trailing-stop"

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

        # Entry: confirmed uptrend
        if (
            candle.close > fast
            and candle.close > slow
            and fast > slow
            and cash > 0
        ):
            entry_price = candle.close
            entry_notional = cash
            entry_fee = entry_notional * FEE_RATE
            qty = entry_notional / entry_price
            peak_price = entry_price
            cash = 0.0
            in_pos = True
            entry_idx = i
            equity_curve[-1] = (candle.ts, entry_notional - entry_fee)

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

    time_in_pos_pct = (bars_in_pos / len(candles) * 100.0) if candles else 0.0
    return trades, equity_curve, time_in_pos_pct


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
    time_in_pos: dict[str, float] = {}

    for symbol in SYMBOLS:
        sliced = slice_candles(candles_by_symbol[symbol], start, end)
        if len(sliced) < params.slow + 1:
            raise RuntimeError(
                f"{symbol}: not enough bars ({len(sliced)}) for slow SMA={params.slow}"
            )
        trades, curve, tip = run_symbol_trend(
            symbol, per_symbol_capital, sliced, params
        )
        all_trades.extend(trades)
        curves[symbol] = curve
        end_equities[symbol] = curve[-1][1] if curve else per_symbol_capital
        time_in_pos[symbol] = tip

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
        time_in_pos_pct=time_in_pos,
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
        Params(fast=fast, slow=slow, use_trail=trail)
        for (fast, slow), trail in itertools.product(SMA_PAIRS, TRAIL_OPTIONS)
    ]
    print("Walk-forward LONG-HORIZON TREND FOLLOWING")
    print(f"Symbols ({len(SYMBOLS)}): {', '.join(SYMBOLS)}")
    print(f"Grid ({len(combos)} combos): {[c.label() for c in combos]}")
    print(
        f"Entry: close > fast & slow SMA and fast > slow | "
        f"Exit: close < slow SMA"
        + (f" and/or {TRAIL_PCT:.0%} trailing stop" if True else "")
    )
    print(f"No time stop | fee={FEE_RATE:.2%} per side | capital={STARTING_CAPITAL:.2f}\n")

    try:
        max_slow = max(p.slow for p in combos)
        candles_by_symbol = {
            sym: load_candles(sym, min_bars=max_slow + 1) for sym in SYMBOLS
        }
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        print("Run test_backtest_data.py first.")
        return 1

    # Full-sample time-in-position sanity (use first combo as representative + all)
    full_start = max(c[0].ts for c in candles_by_symbol.values())
    full_end = min(c[-1].ts for c in candles_by_symbol.values()) + timedelta(hours=1)
    print("=== TIME IN POSITION (full ~180d, per combo / per symbol) ===")
    for params in combos:
        m = run_period(candles_by_symbol, full_start, full_end, params)
        avg_tip = sum(m.time_in_pos_pct.values()) / len(m.time_in_pos_pct)
        print(f"\n{params.label()} | avg across symbols: {avg_tip:.1f}% in position")
        for sym in SYMBOLS:
            print(f"  {sym}: {m.time_in_pos_pct[sym]:5.1f}%")
    print()

    periods = make_periods(candles_by_symbol)
    for name, start, end in periods:
        n = len(slice_candles(candles_by_symbol["BTCUSDT"], start, end))
        print(
            f"Period {name}: {start.isoformat()} -> {end.isoformat()} "
            f"(~{n} hourly bars)"
        )
    print()

    period_a, period_b, period_c = periods

    print(f"Step 1: running all {len(combos)} combos on Period A...")
    a_results: list[tuple[Params, PeriodMetrics]] = []
    for params in combos:
        metrics = run_period(
            candles_by_symbol, period_a[1], period_a[2], params
        )
        a_results.append((params, metrics))
        print(
            f"  A {params.label()}: sharpe={fmt_sharpe(metrics.sharpe)} "
            f"ret={metrics.total_return_pct:.2f}% trades={metrics.trades}"
        )

    a_ranked = sorted(a_results, key=lambda item: sharpe_sort_key(item[1]), reverse=True)
    print("\nPeriod A ranking (by Sharpe):")
    for rank, (params, metrics) in enumerate(a_ranked, start=1):
        print(
            f"  #{rank} {params.label()} | "
            f"sharpe={fmt_sharpe(metrics.sharpe)} ret={metrics.total_return_pct:.2f}%"
        )

    print("\nStep 2-3: evaluating ALL 4 combos on Period B and Period C...")
    rows: list[tuple[Params, PeriodMetrics, PeriodMetrics, PeriodMetrics]] = []
    # Keep A-rank order for the table
    for params, _ in a_ranked:
        a_m = run_period(candles_by_symbol, period_a[1], period_a[2], params)
        b_m = run_period(candles_by_symbol, period_b[1], period_b[2], params)
        c_m = run_period(candles_by_symbol, period_c[1], period_c[2], params)
        rows.append((params, a_m, b_m, c_m))

    print("\n=== WALK-FORWARD TABLE (all 4 combos, ranked by Period A Sharpe) ===\n")
    header = (
        f"{'#':>2} {'params':<28} "
        f"{'A_sharpe':>8} {'A_ret%':>8} "
        f"{'B_sharpe':>8} {'B_ret%':>8} {'B_flag':>10} "
        f"{'C_sharpe':>8} {'C_ret%':>8} {'C_flag':>10}"
    )
    print(header)
    print("-" * len(header))
    for i, (params, a_m, b_m, c_m) in enumerate(rows, start=1):
        print(
            f"{i:>2} {params.label():<28} "
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
        f"Of all {len(rows)} combos: both HELD={strict_both}; "
        f"collapsed on B or C={collapsed_any}."
    )
    print(f"Average return: A={avg_a:.2f}% | B={avg_b:.2f}% | C={avg_c:.2f}%")
    print(
        f"Period A quality: {a_positive}/{len(rows)} had positive return AND positive Sharpe."
    )

    if a_positive == 0 and strict_both == 0:
        print(
            "\nALSO FAILS. No combo was good in-sample on A and none HELD across B and C. "
            "Long-horizon trend following does not show a usable edge on this sample."
        )
    elif strict_both == 0:
        print(
            "\nDOES NOT HOLD UP OUT-OF-SAMPLE. Even if A looked okay for some combos, "
            "nothing cleanly HELD in both B and C. Do not trade this as-is."
        )
    else:
        print(
            f"\n{strict_both}/{len(rows)} HELD in both B and C. "
            "Possible hint only - still verify further before live capital."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
