"""
Walk-forward check for mean-reversion parameter combinations.

Tune on Period A (days 1-60), then evaluate the top-5 by Sharpe on
Period B (61-120) and Period C (121-180) with no re-fitting.
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

from backtesting.backtest_mean_reversion import (
    STARTING_CAPITAL,
    Candle,
    combine_equity,
    load_candles,
    max_drawdown_pct,
    run_symbol,
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

ENTRY_GRID = (-0.015, -0.02, -0.025, -0.03)
TP_GRID = (0.002, 0.003, 0.005)
STOP_GRID = (-0.03, -0.04, -0.05)
SMA_GRID = (10, 20, 30)
TOP_N = 5
CHUNK_DAYS = 60


@dataclass
class Params:
    entry: float
    tp: float
    stop: float
    sma: int

    def label(self) -> str:
        return (
            f"entry={self.entry * 100:.1f}% "
            f"tp={self.tp * 100:.1f}% "
            f"stop={self.stop * 100:.1f}% "
            f"sma={self.sma}"
        )


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
    """Half-open window: start <= ts < end."""
    return [c for c in candles if start <= c.ts < end]


def make_periods(
    candles_by_symbol: dict[str, list[Candle]],
) -> tuple[datetime, list[tuple[str, datetime, datetime]]]:
    # Align chunks to the earliest common start / latest common end
    starts = [candes[0].ts for candes in candles_by_symbol.values()]
    ends = [candes[-1].ts for candes in candles_by_symbol.values()]
    global_start = max(starts)
    global_end = min(ends)

    # Use exact 60-day chunks from global_start
    a0 = global_start
    b0 = a0 + timedelta(days=CHUNK_DAYS)
    c0 = b0 + timedelta(days=CHUNK_DAYS)
    c1 = c0 + timedelta(days=CHUNK_DAYS)

    # If data is slightly shorter/longer than 180d, clamp C end to available data
    if c1 > global_end + timedelta(hours=1):
        c1 = global_end + timedelta(hours=1)

    periods = [
        ("A", a0, b0),
        ("B", b0, c0),
        ("C", c0, c1),
    ]
    return global_start, periods


def run_period(
    candles_by_symbol: dict[str, list[Candle]],
    start: datetime,
    end: datetime,
    params: Params,
) -> PeriodMetrics:
    per_symbol_capital = STARTING_CAPITAL / len(SYMBOLS)
    all_trades = []
    curves = {}
    end_equities = {}

    for symbol in SYMBOLS:
        sliced = slice_candles(candles_by_symbol[symbol], start, end)
        if len(sliced) < params.sma + 1:
            raise RuntimeError(
                f"{symbol}: not enough bars in window "
                f"[{start.isoformat()} -> {end.isoformat()}] for sma={params.sma}"
            )
        trades, curve = run_symbol(
            symbol,
            per_symbol_capital,
            sma_period=params.sma,
            entry_dev=params.entry,
            take_profit_dev=-abs(params.tp),
            stop_loss_dev=params.stop,
            candles=sliced,
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
    """Blunt flag for whether OOS period kept a usable edge."""
    if out.trades == 0:
        return "NO_TRADES"
    oos_sharpe = out.sharpe if out.sharpe is not None else -999.0
    if isinstance(oos_sharpe, float) and math.isinf(oos_sharpe):
        oos_sharpe = 999.0 if oos_sharpe > 0 else -999.0

    # Collapse: negative return or non-positive Sharpe out-of-sample
    if out.total_return_pct <= 0 or oos_sharpe <= 0:
        return "COLLAPSED"
    # Weak hold: still profitable + positive Sharpe, but much weaker than A
    a_sharpe = sharpe_sort_key(a)
    if out.total_return_pct < a.total_return_pct * 0.25 or oos_sharpe < a_sharpe * 0.25:
        return "WEAK"
    return "HELD"


def main() -> int:
    combos = [
        Params(entry=e, tp=tp, stop=sl, sma=sma)
        for e, tp, sl, sma in itertools.product(ENTRY_GRID, TP_GRID, STOP_GRID, SMA_GRID)
    ]
    print("Walk-forward mean-reversion test")
    print(f"Symbols ({len(SYMBOLS)}): {', '.join(SYMBOLS)}")
    print(f"Grid size: {len(combos)} | Top-{TOP_N} selected on Period A Sharpe")
    print(f"Chunks: {CHUNK_DAYS} days each (A tune / B+C validate)\n")

    try:
        candles_by_symbol = {
            sym: load_candles(sym, min_bars=max(SMA_GRID) + 1) for sym in SYMBOLS
        }
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        print("Run test_backtest_data.py first.")
        return 1

    _, periods = make_periods(candles_by_symbol)
    for name, start, end in periods:
        # Approximate bar counts from BTC
        n = len(slice_candles(candles_by_symbol["BTCUSDT"], start, end))
        print(
            f"Period {name}: {start.isoformat()} -> {end.isoformat()} "
            f"(~{n} hourly bars on BTCUSDT)"
        )
    print()

    period_a = periods[0]
    period_b = periods[1]
    period_c = periods[2]

    print(f"Step 1-2: running all {len(combos)} combos on Period A...")
    a_results: list[tuple[Params, PeriodMetrics]] = []
    for i, params in enumerate(combos, start=1):
        metrics = run_period(
            candles_by_symbol, period_a[1], period_a[2], params
        )
        a_results.append((params, metrics))
        if i % 20 == 0 or i == len(combos):
            print(f"  Period A progress: {i}/{len(combos)}", flush=True)

    a_ranked = sorted(a_results, key=lambda item: sharpe_sort_key(item[1]), reverse=True)
    top5 = a_ranked[:TOP_N]

    print(f"\nStep 3: Top {TOP_N} by Period A Sharpe:")
    for rank, (params, metrics) in enumerate(top5, start=1):
        print(
            f"  #{rank} {params.label()} | "
            f"A sharpe={fmt_sharpe(metrics.sharpe)} "
            f"ret={metrics.total_return_pct:.2f}% "
            f"trades={metrics.trades}"
        )

    print("\nStep 4-5: evaluating top 5 on Period B and Period C (no re-tuning)...")
    rows: list[tuple[Params, PeriodMetrics, PeriodMetrics, PeriodMetrics]] = []
    for params, _a_metrics in top5:
        # Recompute A metrics from stored; also run B and C fresh
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

    held_b = 0
    held_c = 0
    held_both = 0
    for i, (params, a_m, b_m, c_m) in enumerate(rows, start=1):
        flag_b = held_up(a_m, b_m)
        flag_c = held_up(a_m, c_m)
        if flag_b in {"HELD", "WEAK"}:
            held_b += 1
        if flag_c in {"HELD", "WEAK"}:
            held_c += 1
        if flag_b in {"HELD", "WEAK"} and flag_c in {"HELD", "WEAK"}:
            held_both += 1
        # Strict hold counts separately below
        print(
            f"{i:>2} {params.label():<42} "
            f"{fmt_sharpe(a_m.sharpe):>8} {a_m.total_return_pct:7.2f}% "
            f"{fmt_sharpe(b_m.sharpe):>8} {b_m.total_return_pct:7.2f}% {flag_b:>10} "
            f"{fmt_sharpe(c_m.sharpe):>8} {c_m.total_return_pct:7.2f}% {flag_c:>10}"
        )

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

    print("\nFlag legend: HELD = still profitable + positive Sharpe and not gutted vs A;")
    print("             WEAK = still >0 return and Sharpe but much weaker than A;")
    print("             COLLAPSED = return<=0 or Sharpe<=0 out-of-sample.")

    print("\n=== BLUNT VERDICT ===")
    print(
        f"Of the top {TOP_N} Period-A winners: "
        f"B HELD={strict_b}, B WEAK/HELD={held_b}; "
        f"C HELD={strict_c}, C WEAK/HELD={held_c}; "
        f"both HELD={strict_both}; "
        f"collapsed on B or C={collapsed_any}."
    )

    # Average OOS returns of top5
    avg_b = sum(b.total_return_pct for _, _, b, _ in rows) / len(rows)
    avg_c = sum(c.total_return_pct for _, _, _, c in rows) / len(rows)
    avg_a = sum(a.total_return_pct for _, a, _, _ in rows) / len(rows)
    print(
        f"Average return among top {TOP_N}: "
        f"A={avg_a:.2f}% | B={avg_b:.2f}% | C={avg_c:.2f}%"
    )

    if strict_both == 0 and held_both == 0:
        print(
            "\nNOTHING SURVIVED. The best in-sample (Period A) combinations did not "
            "carry a usable edge into both later periods. This looks like curve-fitting "
            "/ regime luck, not a genuine robust edge. Do not trade this mean-reversion "
            "setup as-is."
        )
    elif strict_both == 0:
        print(
            "\nWEAK / FRAGILE. A couple of combos stayed barely positive out-of-sample, "
            "but none cleanly HELD up in both B and C. Not good enough to trust live."
        )
    else:
        print(
            f"\n{strict_both}/{TOP_N} combination(s) HELD in both B and C. "
            "That is the only pattern that even hints at a real edge — still verify "
            "with more regimes/costs before any live capital."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
