"""
Cross-sectional relative-strength backtest + walk-forward validation.

At each rebalance: rank 8 symbols by ROC(N), long top-2 with ROC>0
(equal weight), rotate until next rebalance.
Tune on Period A; validate top-3 on Period B and C.
"""

from __future__ import annotations

import itertools
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta

from backtest_mean_reversion import (
    STARTING_CAPITAL,
    Candle,
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

N_GRID = (12, 24, 48)
REBALANCE_HOURS_GRID = (1, 4)
TOP_K = 2
FEE_RATE = 0.0005
TOP_N_SELECT = 3
CHUNK_DAYS = 60


@dataclass
class Params:
    n: int
    rebalance_hours: int

    def label(self) -> str:
        return f"N={self.n}h rebalance={self.rebalance_hours}h"


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


def build_price_panel(
    candles_by_symbol: dict[str, list[Candle]],
    start: datetime,
    end: datetime,
) -> tuple[list[datetime], dict[str, list[float]]]:
    """Align symbols on the intersection of timestamps in [start, end)."""
    ts_sets = []
    close_maps: dict[str, dict[datetime, float]] = {}
    for sym, candles in candles_by_symbol.items():
        sliced = slice_candles(candles, start, end)
        close_maps[sym] = {c.ts: c.close for c in sliced}
        ts_sets.append(set(close_maps[sym].keys()))

    common_ts = sorted(set.intersection(*ts_sets)) if ts_sets else []
    prices = {
        sym: [close_maps[sym][ts] for ts in common_ts] for sym in SYMBOLS
    }
    return common_ts, prices


def roc(prices: list[float], index: int, n: int) -> float | None:
    if index < n:
        return None
    base = prices[index - n]
    if base == 0:
        return None
    return (prices[index] - base) / base


def run_cross_sectional(
    timestamps: list[datetime],
    prices: dict[str, list[float]],
    params: Params,
    starting_capital: float = STARTING_CAPITAL,
) -> tuple[list[float], int, int]:
    """
    Returns (equity_curve, trade_count, winning_trade_count).

    A "trade" = one symbol entry->exit round trip (fees on both sides).
    """
    cash = starting_capital
    # symbol -> (qty, entry_notional, entry_fee)
    positions: dict[str, tuple[float, float, float]] = {}
    equity_curve: list[float] = []
    trade_pnls: list[float] = []

    n_bars = len(timestamps)
    if n_bars == 0:
        return [], 0, 0

    def mark_equity(i: int) -> float:
        total = cash
        for sym, (qty, _en, _ef) in positions.items():
            total += qty * prices[sym][i]
        return total

    def close_position(sym: str, i: int) -> None:
        nonlocal cash
        qty, entry_notional, entry_fee = positions.pop(sym)
        exit_notional = qty * prices[sym][i]
        exit_fee = exit_notional * FEE_RATE
        pnl = (exit_notional - entry_notional) - entry_fee - exit_fee
        cash += exit_notional - exit_fee
        trade_pnls.append(pnl)

    def open_position(sym: str, i: int, notional: float) -> None:
        nonlocal cash
        if notional <= 0 or cash < notional:
            notional = min(notional, cash)
        if notional <= 0:
            return
        price = prices[sym][i]
        entry_fee = notional * FEE_RATE
        qty = notional / price
        cash -= notional
        # Pay entry fee from remaining cash when possible; else absorb in pnl only
        if cash >= entry_fee:
            cash -= entry_fee
        positions[sym] = (qty, notional, entry_fee)

    for i in range(n_bars):
        is_rebalance = (i % params.rebalance_hours == 0) and (i >= params.n)

        if is_rebalance:
            scores: list[tuple[str, float]] = []
            for sym in SYMBOLS:
                r = roc(prices[sym], i, params.n)
                if r is not None:
                    scores.append((sym, r))
            scores.sort(key=lambda item: item[1], reverse=True)

            top = scores[:TOP_K]
            targets = [sym for sym, r in top if r > 0]

            if targets:
                # Full flatten + equal-weight reopen whenever target set changes
                if set(positions.keys()) != set(targets):
                    for sym in list(positions.keys()):
                        close_position(sym, i)
                    equity_now = cash
                    weight = equity_now / len(targets)
                    for sym in targets:
                        open_position(sym, i, weight)
            else:
                for sym in list(positions.keys()):
                    close_position(sym, i)

        equity_curve.append(mark_equity(i))

    if positions:
        last_i = n_bars - 1
        for sym in list(positions.keys()):
            close_position(sym, last_i)
        equity_curve[-1] = cash

    wins = sum(1 for p in trade_pnls if p > 0)
    return equity_curve, len(trade_pnls), wins


def run_period(
    candles_by_symbol: dict[str, list[Candle]],
    start: datetime,
    end: datetime,
    params: Params,
) -> PeriodMetrics:
    timestamps, prices = build_price_panel(candles_by_symbol, start, end)
    if len(timestamps) < params.n + 1:
        raise RuntimeError(
            f"Not enough aligned bars in window for N={params.n} "
            f"({len(timestamps)} bars)"
        )

    equity, trades, wins = run_cross_sectional(timestamps, prices, params)
    if not equity:
        return PeriodMetrics(0, 0.0, 0.0, None, 0.0)

    end_equity = equity[-1]
    return PeriodMetrics(
        trades=trades,
        win_rate=(wins / trades * 100.0) if trades else 0.0,
        total_return_pct=(end_equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100.0,
        sharpe=sharpe_from_equity(equity),
        max_dd_pct=max_drawdown_pct(equity),
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
        # Could be all-cash period; treat non-positive as collapsed for edge purposes
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
        Params(n=n, rebalance_hours=rb)
        for n, rb in itertools.product(N_GRID, REBALANCE_HOURS_GRID)
    ]
    print("Walk-forward CROSS-SECTIONAL relative strength test")
    print(f"Symbols ({len(SYMBOLS)}): {', '.join(SYMBOLS)}")
    print(
        f"Grid: N={list(N_GRID)}, rebalance_hours={list(REBALANCE_HOURS_GRID)} "
        f"-> {len(combos)} combos | long top-{TOP_K} if ROC>0"
    )
    print(f"fee={FEE_RATE:.2%} per side | capital={STARTING_CAPITAL:.2f} USDT")
    print(f"Top-{TOP_N_SELECT} selected on Period A Sharpe; validate on B and C\n")

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
        ts, _ = build_price_panel(candles_by_symbol, start, end)
        print(
            f"Period {name}: {start.isoformat()} -> {end.isoformat()} "
            f"({len(ts)} aligned hourly bars)"
        )
    print()

    period_a, period_b, period_c = periods

    print(f"Step 1-2: running all {len(combos)} combos on Period A...")
    a_results: list[tuple[Params, PeriodMetrics]] = []
    for i, params in enumerate(combos, start=1):
        metrics = run_period(candles_by_symbol, period_a[1], period_a[2], params)
        a_results.append((params, metrics))
        print(
            f"  A [{i}/{len(combos)}] {params.label()}: "
            f"sharpe={fmt_sharpe(metrics.sharpe)} ret={metrics.total_return_pct:.2f}% "
            f"trades={metrics.trades}"
        )

    a_ranked = sorted(a_results, key=lambda item: sharpe_sort_key(item[1]), reverse=True)
    top = a_ranked[:TOP_N_SELECT]

    print(f"\nStep 3: Top {TOP_N_SELECT} by Period A Sharpe:")
    for rank, (params, metrics) in enumerate(top, start=1):
        print(
            f"  #{rank} {params.label()} | "
            f"A sharpe={fmt_sharpe(metrics.sharpe)} "
            f"ret={metrics.total_return_pct:.2f}% "
            f"trades={metrics.trades} win={metrics.win_rate:.1f}%"
        )

    print("\nStep 4: evaluating top combos on Period B and Period C (no re-tuning)...")
    rows: list[tuple[Params, PeriodMetrics, PeriodMetrics, PeriodMetrics]] = []
    for params, _ in top:
        a_m = run_period(candles_by_symbol, period_a[1], period_a[2], params)
        b_m = run_period(candles_by_symbol, period_b[1], period_b[2], params)
        c_m = run_period(candles_by_symbol, period_c[1], period_c[2], params)
        rows.append((params, a_m, b_m, c_m))

    print("\n=== WALK-FORWARD TABLE (Top 3 from Period A) ===\n")
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

    print("\nFlag legend: HELD = still profitable + positive Sharpe and not gutted vs A;")
    print("             WEAK = still >0 return and Sharpe but much weaker than A;")
    print("             COLLAPSED = return<=0 or Sharpe<=0 out-of-sample.")

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

    if a_positive == 0:
        print(
            "\nNOTHING TO TRUST. Period A top combos were not good in-sample. "
            "Cross-sectional relative strength failed this screen."
        )
    elif strict_both == 0 and collapsed_any == len(rows):
        print(
            "\nNOTHING SURVIVED. Best Period-A combos collapsed in B and/or C. "
            "Three strategy families in a row with no walk-forward edge is useful: "
            "do not force a tradeable story out of these. Do not trade this setup as-is."
        )
    elif strict_both == 0:
        print(
            "\nWEAK / FRAGILE. No combo cleanly HELD in both B and C. "
            "Not good enough to trust live."
        )
    else:
        print(
            f"\n{strict_both}/{TOP_N_SELECT} HELD in both B and C. "
            "Only a hint - still verify further before any live capital."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
