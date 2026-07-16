"""
Momentum (long-only) backtest + walk-forward validation.

Grid: ROC lookback N x entry threshold x trailing stop = 27 combos.
Tune on Period A (60d), validate top-5 on Period B and C (no re-fit).
"""

from __future__ import annotations

import itertools
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from backtest_mean_reversion import (
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

N_GRID = (6, 12, 24)
ENTRY_GRID = (0.015, 0.02, 0.03)
TRAIL_GRID = (0.02, 0.03, 0.04)
MAX_HOLD = timedelta(hours=48)
FEE_RATE = 0.0005
TOP_N = 5
CHUNK_DAYS = 60
ExitReason = Literal["trailing-stop", "momentum-reverse", "timeout"]


@dataclass
class Params:
    n: int
    entry: float
    trail: float

    def label(self) -> str:
        return f"N={self.n} entry=+{self.entry * 100:.1f}% trail={self.trail * 100:.1f}%"


@dataclass
class PeriodMetrics:
    trades: int
    win_rate: float
    total_return_pct: float
    sharpe: float | None
    max_dd_pct: float


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


def roc_at(closes: list[float], index: int, n: int) -> float | None:
    """Percentage price change over the last N periods: (close[i]/close[i-N]) / close[i-N]."""
    if index < n:
        return None
    base = closes[index - n]
    if base == 0:
        return None
    return (closes[index] - base) / base


def run_symbol_momentum(
    symbol: str,
    starting_cash: float,
    candles: list[Candle],
    params: Params,
) -> tuple[list[Trade], list[tuple[datetime, float]]]:
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

    for i, candle in enumerate(candles):
        roc = roc_at(closes, i, params.n)
        mark_equity = cash
        if in_pos:
            mark_equity = qty * candle.close - entry_fee
        equity_curve.append((candle.ts, mark_equity))

        if in_pos:
            if candle.close > peak_price:
                peak_price = candle.close

            reason: ExitReason | None = None
            drawdown_from_peak = (peak_price - candle.close) / peak_price if peak_price > 0 else 0.0
            if drawdown_from_peak >= params.trail:
                reason = "trailing-stop"
            elif roc is not None and roc < 0:
                reason = "momentum-reverse"
            elif candle.ts - candles[entry_idx].ts >= MAX_HOLD:
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

        if roc is None:
            continue

        if roc >= params.entry and cash > 0:
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
                exit_reason="timeout",  # type: ignore[arg-type]
                pnl=pnl,
                return_pct=(pnl / entry_notional) * 100.0 if entry_notional else 0.0,
            )
        )
        equity_curve[-1] = (last.ts, cash)

    return trades, equity_curve


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

    for symbol in SYMBOLS:
        sliced = slice_candles(candles_by_symbol[symbol], start, end)
        if len(sliced) < params.n + 1:
            raise RuntimeError(
                f"{symbol}: not enough bars in window for N={params.n}"
            )
        trades, curve = run_symbol_momentum(
            symbol, per_symbol_capital, sliced, params
        )
        all_trades.extend(trades)
        curves[symbol] = curve
        end_equities[symbol] = curve[-1][1] if curve else per_symbol_capital

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
    if out.trades == 0:
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
        Params(n=n, entry=entry, trail=trail)
        for n, entry, trail in itertools.product(N_GRID, ENTRY_GRID, TRAIL_GRID)
    ]
    print("Walk-forward MOMENTUM test (long-only)")
    print(f"Symbols ({len(SYMBOLS)}): {', '.join(SYMBOLS)}")
    print(
        f"Grid: N={list(N_GRID)}, entry={list(ENTRY_GRID)}, "
        f"trail={list(TRAIL_GRID)} -> {len(combos)} combos"
    )
    print(
        f"Exits: trailing stop from peak, ROC<0 reverse, or {MAX_HOLD} timeout | "
        f"fee={FEE_RATE:.2%} per side"
    )
    print(f"Top-{TOP_N} selected on Period A Sharpe; validate on B and C\n")

    try:
        candles_by_symbol = {
            sym: load_candles(sym, min_bars=max(N_GRID) + 1) for sym in SYMBOLS
        }
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        print("Run test_backtest_data.py first.")
        return 1

    periods = make_periods(candles_by_symbol)
    for name, start, end in periods:
        n_bars = len(slice_candles(candles_by_symbol["BTCUSDT"], start, end))
        print(
            f"Period {name}: {start.isoformat()} -> {end.isoformat()} "
            f"(~{n_bars} hourly bars on BTCUSDT)"
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
        if i % 9 == 0 or i == len(combos):
            print(f"  Period A progress: {i}/{len(combos)}", flush=True)

    a_ranked = sorted(a_results, key=lambda item: sharpe_sort_key(item[1]), reverse=True)
    top5 = a_ranked[:TOP_N]

    print(f"\nStep 3: Top {TOP_N} by Period A Sharpe:")
    for rank, (params, metrics) in enumerate(top5, start=1):
        print(
            f"  #{rank} {params.label()} | "
            f"A sharpe={fmt_sharpe(metrics.sharpe)} "
            f"ret={metrics.total_return_pct:.2f}% "
            f"trades={metrics.trades} "
            f"win={metrics.win_rate:.1f}%"
        )

    print("\nStep 4-5: evaluating top 5 on Period B and Period C (no re-tuning)...")
    rows: list[tuple[Params, PeriodMetrics, PeriodMetrics, PeriodMetrics]] = []
    for params, _ in top5:
        a_m = run_period(candles_by_symbol, period_a[1], period_a[2], params)
        b_m = run_period(candles_by_symbol, period_b[1], period_b[2], params)
        c_m = run_period(candles_by_symbol, period_c[1], period_c[2], params)
        rows.append((params, a_m, b_m, c_m))

    print("\n=== WALK-FORWARD TABLE (Top 5 from Period A) ===\n")
    header = (
        f"{'#':>2} {'params':<40} "
        f"{'A_sharpe':>8} {'A_ret%':>8} "
        f"{'B_sharpe':>8} {'B_ret%':>8} {'B_flag':>10} "
        f"{'C_sharpe':>8} {'C_ret%':>8} {'C_flag':>10}"
    )
    print(header)
    print("-" * len(header))

    for i, (params, a_m, b_m, c_m) in enumerate(rows, start=1):
        flag_b = held_up(a_m, b_m)
        flag_c = held_up(a_m, c_m)
        print(
            f"{i:>2} {params.label():<40} "
            f"{fmt_sharpe(a_m.sharpe):>8} {a_m.total_return_pct:7.2f}% "
            f"{fmt_sharpe(b_m.sharpe):>8} {b_m.total_return_pct:7.2f}% {flag_b:>10} "
            f"{fmt_sharpe(c_m.sharpe):>8} {c_m.total_return_pct:7.2f}% {flag_c:>10}"
        )

    print("\nFlag legend: HELD = still profitable + positive Sharpe and not gutted vs A;")
    print("             WEAK = still >0 return and Sharpe but much weaker than A;")
    print("             COLLAPSED = return<=0 or Sharpe<=0 out-of-sample.")

    strict_b = sum(1 for _, a_m, b_m, _ in rows if held_up(a_m, b_m) == "HELD")
    strict_c = sum(1 for _, a_m, _, c_m in rows if held_up(a_m, c_m) == "HELD")
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
    weak_or_held_both = sum(
        1
        for _, a_m, b_m, c_m in rows
        if held_up(a_m, b_m) in {"HELD", "WEAK"}
        and held_up(a_m, c_m) in {"HELD", "WEAK"}
    )

    avg_a = sum(a.total_return_pct for _, a, _, _ in rows) / len(rows)
    avg_b = sum(b.total_return_pct for _, _, b, _ in rows) / len(rows)
    avg_c = sum(c.total_return_pct for _, _, _, c in rows) / len(rows)

    # How strong was Period A selection itself?
    a_positive = sum(1 for _, a, _, _ in rows if a.total_return_pct > 0 and sharpe_sort_key(a) > 0)
    a_sharpes = [sharpe_sort_key(a) for _, a, _, _ in rows]

    print("\n=== BLUNT VERDICT ===")
    print(
        f"Of the top {TOP_N} Period-A winners: "
        f"B HELD={strict_b}; C HELD={strict_c}; "
        f"both HELD={strict_both}; "
        f"both HELD-or-WEAK={weak_or_held_both}; "
        f"collapsed on B or C={collapsed_any}."
    )
    print(
        f"Average return among top {TOP_N}: "
        f"A={avg_a:.2f}% | B={avg_b:.2f}% | C={avg_c:.2f}%"
    )
    print(
        f"Period A quality of top {TOP_N}: "
        f"{a_positive}/{TOP_N} had positive return AND positive Sharpe; "
        f"A Sharpe range [{min(a_sharpes):.2f}, {max(a_sharpes):.2f}]"
    )

    if a_positive == 0:
        print(
            "\nNOTHING TO TRUST. Period A 'top' combos were not even good in-sample "
            "(no positive return+Sharpe among the top 5). Ranking least-bad losers is not "
            "finding an edge. Momentum failed this walk-forward screen."
        )
    elif strict_both == 0 and weak_or_held_both == 0:
        print(
            "\nNOTHING SURVIVED. Best Period-A combos collapsed out-of-sample in B/C. "
            "Looks like curve-fitting / regime luck, not a genuine robust momentum edge. "
            "Do not trade this setup as-is."
        )
    elif strict_both == 0:
        print(
            "\nWEAK / FRAGILE. Some combos stayed barely positive OOS, but none cleanly "
            "HELD in both B and C. Not good enough to trust live."
        )
    else:
        # Don't round up: require meaningful A quality too
        if max(a_sharpes) < 0.5 or avg_a < 1.0:
            print(
                f"\n{strict_both}/{TOP_N} HELD in both B and C on the flag rules, BUT "
                "Period A edges were small/mediocre. Do not round that up into a real edge - "
                "treat as inconclusive at best, not a strategy to ship."
            )
        else:
            print(
                f"\n{strict_both}/{TOP_N} combination(s) HELD in both B and C with a "
                "non-trivial Period A selection. Hint of possible edge only - still need "
                "more regimes, costs, and shorts/risk controls before any live capital."
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
