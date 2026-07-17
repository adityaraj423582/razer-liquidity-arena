"""
Parameter sensitivity grid for the mean-reversion strategy.

Runs every combination of entry / TP / SL / SMA on the 180-day, 8-symbol CSVs.
No live API calls — historical replay only.
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

from backtesting.backtest_mean_reversion import (
    STARTING_CAPITAL,
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
# Stored as positive "within X% of SMA"; engine uses negative threshold
TP_GRID = (0.002, 0.003, 0.005)
STOP_GRID = (-0.03, -0.04, -0.05)
SMA_GRID = (10, 20, 30)

# Original baseline from first backtest
BASELINE = {
    "entry": -0.02,
    "tp": 0.003,
    "stop": -0.04,
    "sma": 20,
}


@dataclass
class GridResult:
    entry: float
    tp: float
    stop: float
    sma: int
    trades: int
    win_rate: float
    total_return_pct: float
    sharpe: float | None
    max_dd_pct: float
    is_baseline: bool


def run_combo(
    candles_by_symbol: dict,
    entry: float,
    tp: float,
    stop: float,
    sma: int,
) -> GridResult:
    per_symbol_capital = STARTING_CAPITAL / len(SYMBOLS)
    take_profit_dev = -abs(tp)
    all_trades = []
    curves = {}
    end_equities = {}

    for symbol in SYMBOLS:
        trades, curve = run_symbol(
            symbol,
            per_symbol_capital,
            sma_period=sma,
            entry_dev=entry,
            take_profit_dev=take_profit_dev,
            stop_loss_dev=stop,
            candles=candles_by_symbol[symbol],
        )
        all_trades.extend(trades)
        curves[symbol] = curve
        end_equities[symbol] = curve[-1][1] if curve else per_symbol_capital

    combined = combine_equity(curves)
    equity_values = [v for _, v in combined]
    end_equity = sum(end_equities.values())
    n = len(all_trades)
    wins = sum(1 for t in all_trades if t.pnl > 0)
    win_rate = (wins / n * 100.0) if n else 0.0
    total_return_pct = (end_equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100.0
    sharpe = sharpe_from_equity(equity_values)
    mdd = max_drawdown_pct(equity_values)

    return GridResult(
        entry=entry,
        tp=tp,
        stop=stop,
        sma=sma,
        trades=n,
        win_rate=win_rate,
        total_return_pct=total_return_pct,
        sharpe=sharpe,
        max_dd_pct=mdd,
        is_baseline=(
            entry == BASELINE["entry"]
            and tp == BASELINE["tp"]
            and stop == BASELINE["stop"]
            and sma == BASELINE["sma"]
        ),
    )


def fmt_sharpe(value: float | None) -> str:
    if value is None:
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value:6.2f}"


def print_table(results: list[GridResult]) -> None:
    # Sort by Sharpe descending (None last), then return
    def sort_key(r: GridResult):
        s = r.sharpe if r.sharpe is not None and not math.isnan(r.sharpe) else -999.0
        if math.isinf(s):
            s = 999.0 if s > 0 else -999.0
        return (s, r.total_return_pct)

    ordered = sorted(results, key=sort_key, reverse=True)

    header = (
        f"{'entry':>7} {'tp':>6} {'stop':>6} {'sma':>4} "
        f"{'trades':>6} {'win%':>7} {'ret%':>8} {'sharpe':>7} {'maxDD%':>7} {'note':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in ordered:
        note = "BASELINE" if r.is_baseline else ""
        print(
            f"{r.entry * 100:6.1f}% "
            f"{r.tp * 100:5.1f}% "
            f"{r.stop * 100:5.1f}% "
            f"{r.sma:4d} "
            f"{r.trades:6d} "
            f"{r.win_rate:6.1f}% "
            f"{r.total_return_pct:7.2f}% "
            f"{fmt_sharpe(r.sharpe):>7} "
            f"{r.max_dd_pct:6.2f}% "
            f"{note:>8}"
        )


def assess(results: list[GridResult]) -> None:
    baseline = next((r for r in results if r.is_baseline), None)
    valid = [r for r in results if r.trades > 0]

    profitable = [r for r in valid if r.total_return_pct > 0]
    sharpe_pos = [
        r for r in valid if r.sharpe is not None and not math.isinf(r.sharpe) and r.sharpe > 0
    ]
    sharpe_gt_1 = [
        r for r in valid if r.sharpe is not None and not math.isinf(r.sharpe) and r.sharpe > 1.0
    ]
    losing = [r for r in valid if r.total_return_pct <= 0]

    print("\n=== SENSITIVITY ASSESSMENT ===")
    print(f"Grid size: {len(results)} combinations")
    print(f"Combinations with trades: {len(valid)}")
    print(
        f"Profitable (ret>0): {len(profitable)}/{len(valid)} "
        f"({len(profitable) / len(valid) * 100:.1f}%)"
        if valid
        else "Profitable: n/a"
    )
    print(
        f"Sharpe > 0: {len(sharpe_pos)}/{len(valid)} "
        f"({len(sharpe_pos) / len(valid) * 100:.1f}%)"
        if valid
        else "Sharpe>0: n/a"
    )
    print(
        f"Sharpe > 1: {len(sharpe_gt_1)}/{len(valid)} "
        f"({len(sharpe_gt_1) / len(valid) * 100:.1f}%)"
        if valid
        else "Sharpe>1: n/a"
    )
    print(
        f"Losing or flat (ret<=0): {len(losing)}/{len(valid)} "
        f"({len(losing) / len(valid) * 100:.1f}%)"
        if valid
        else "Losing: n/a"
    )

    if baseline is not None:
        print(
            f"\nBaseline (entry=-2%, tp=0.3%, stop=-4%, sma=20): "
            f"ret={baseline.total_return_pct:.2f}%, "
            f"sharpe={fmt_sharpe(baseline.sharpe).strip()}, "
            f"win={baseline.win_rate:.1f}%, "
            f"trades={baseline.trades}, "
            f"maxDD={baseline.max_dd_pct:.2f}%"
        )

    # Nearby = differ in exactly one parameter dimension from baseline
    if baseline is not None:
        nearby = []
        for r in results:
            diffs = 0
            if r.entry != baseline.entry:
                diffs += 1
            if r.tp != baseline.tp:
                diffs += 1
            if r.stop != baseline.stop:
                diffs += 1
            if r.sma != baseline.sma:
                diffs += 1
            if diffs == 1:
                nearby.append(r)
        if nearby:
            near_profit = sum(1 for r in nearby if r.total_return_pct > 0)
            near_sharpe_pos = sum(
                1
                for r in nearby
                if r.sharpe is not None and not math.isinf(r.sharpe) and r.sharpe > 0
            )
            print(
                f"One-step neighbors of baseline: {len(nearby)} "
                f"(profitable {near_profit}/{len(nearby)}, "
                f"Sharpe>0 {near_sharpe_pos}/{len(nearby)})"
            )
            print("Neighbor detail:")
            for r in sorted(nearby, key=lambda x: x.total_return_pct, reverse=True):
                changed = []
                if r.entry != baseline.entry:
                    changed.append(f"entry={r.entry * 100:.1f}%")
                if r.tp != baseline.tp:
                    changed.append(f"tp={r.tp * 100:.1f}%")
                if r.stop != baseline.stop:
                    changed.append(f"stop={r.stop * 100:.1f}%")
                if r.sma != baseline.sma:
                    changed.append(f"sma={r.sma}")
                print(
                    f"  {', '.join(changed):<18} "
                    f"ret={r.total_return_pct:7.2f}%  "
                    f"sharpe={fmt_sharpe(r.sharpe).strip():>6}  "
                    f"win={r.win_rate:5.1f}%  trades={r.trades}"
                )

    returns = [r.total_return_pct for r in valid]
    sharpes = [
        r.sharpe
        for r in valid
        if r.sharpe is not None and not math.isinf(r.sharpe)
    ]
    if returns:
        print(
            f"\nReturn distribution: min={min(returns):.2f}%, "
            f"median={sorted(returns)[len(returns) // 2]:.2f}%, "
            f"max={max(returns):.2f}%"
        )
    if sharpes:
        print(
            f"Sharpe distribution: min={min(sharpes):.2f}, "
            f"median={sorted(sharpes)[len(sharpes) // 2]:.2f}, "
            f"max={max(sharpes):.2f}"
        )

    # Honest verdict
    print("\n=== VERDICT ===")
    if not valid:
        print("No valid runs with trades — cannot assess robustness.")
        return

    profit_share = len(profitable) / len(valid)
    sharpe_share = len(sharpe_pos) / len(valid) if valid else 0.0

    if profit_share >= 0.7 and sharpe_share >= 0.7:
        print(
            "Performance stays reasonably good across MOST of the grid "
            "(majority profitable with positive Sharpe). That is more consistent with "
            "a real-ish edge on this sample than a single lucky parameter spike — "
            "but this is still in-sample on the same 180 days, not true out-of-sample."
        )
    elif profit_share >= 0.45 and baseline is not None and baseline.total_return_pct > 0:
        print(
            "MIXED robustness: a sizable fraction of nearby/other combos still work, "
            "but results vary a lot by parameters. Treat the original setting as "
            "suggestive, not proven — do not assume the edge is stable."
        )
    else:
        print(
            "Performance collapses outside a narrow band (or is weak across the grid). "
            "That pattern looks more like overfitting / luck on the original parameters "
            "than a robust edge. Do not trust the first-pass result for live trading."
        )

    if baseline is not None and nearby:
        near_profit_share = near_profit / len(nearby)
        if near_profit_share < 0.5:
            print(
                "Specifically: most one-step neighbors of the baseline are NOT profitable, "
                "which is a red flag for fragility."
            )
        elif near_profit_share >= 0.75:
            print(
                "Specifically: most one-step neighbors of the baseline remain profitable, "
                "which is the healthier pattern."
            )


def main() -> int:
    combos = list(itertools.product(ENTRY_GRID, TP_GRID, STOP_GRID, SMA_GRID))
    print("Mean-reversion parameter sensitivity")
    print(f"Symbols ({len(SYMBOLS)}): {', '.join(SYMBOLS)}")
    print(
        f"Grid: entry={list(ENTRY_GRID)}, tp={list(TP_GRID)}, "
        f"stop={list(STOP_GRID)}, sma={list(SMA_GRID)}"
    )
    print(f"Combinations: {len(combos)}")
    print(f"Capital: {STARTING_CAPITAL:.2f} USDT split evenly across symbols\n")

    try:
        candles_by_symbol = {
            sym: load_candles(sym, min_bars=max(SMA_GRID) + 1) for sym in SYMBOLS
        }
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        print("Run test_backtest_data.py first.")
        return 1

    results: list[GridResult] = []
    for i, (entry, tp, stop, sma) in enumerate(combos, start=1):
        result = run_combo(candles_by_symbol, entry, tp, stop, sma)
        results.append(result)
        if i % 20 == 0 or i == len(combos):
            print(f"Progress: {i}/{len(combos)}", flush=True)

    print("\n=== RESULTS (sorted by Sharpe, then return) ===\n")
    print_table(results)
    assess(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
