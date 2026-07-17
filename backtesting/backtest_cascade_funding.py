"""
Cascade + crowded-short funding filter, walk-forward validated.

Cascade: hourly drop >= 4% AND volume >= 2x 24h avg.
Filter: as-of funding in bottom 20th percentile of ~30d history.
Enter next bar open; TP = cascade retracement; SL < cascade low; time stop.
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

from backtesting.backtest_cascade_reversal import avg_volume
from backtesting.backtest_funding_signal import (
    LOOKBACK,
    MIN_FUNDING_SAMPLES,
    FundingAsOf,
    load_funding,
    percentile,
)
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

DROP_THRESHOLD = 0.04
VOL_RATIO_THRESHOLD = 2.0
FUNDING_PERCENTILE = 20.0
RETRACE_GRID = (0.40, 0.60)
TIME_GRID = (8, 24)
VOL_AVG_PERIOD = 24
FEE_RATE = 0.0005
CHUNK_DAYS = 60
ExitReason = Literal["take-profit", "stop-loss", "timeout", "end-of-data"]


@dataclass
class Params:
    retrace: float
    max_hold_hours: int

    def label(self) -> str:
        return f"retr={self.retrace * 100:.0f}% t={self.max_hold_hours}h"


@dataclass
class PeriodMetrics:
    trades: int
    win_rate: float
    total_return_pct: float
    sharpe: float | None
    max_dd_pct: float
    cascades_raw: int
    cascades_filtered: int


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


def run_symbol_combo(
    symbol: str,
    starting_cash: float,
    candles: list[Candle],
    funding: FundingAsOf,
    params: Params,
) -> tuple[list[Trade], list[tuple[datetime, float]], int, int]:
    volumes = [c.volume for c in candles]
    cash = starting_cash
    equity_curve: list[tuple[datetime, float]] = []
    trades: list[Trade] = []
    cascades_raw = 0
    cascades_filtered = 0

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
            reason: ExitReason | None = None
            if candle.low < cascade_low:
                reason = "stop-loss"
                exit_price = cascade_low
            elif candle.high >= tp_price:
                reason = "take-profit"
                exit_price = tp_price
            elif (candle.ts - candles[entry_idx].ts) >= timedelta(
                hours=params.max_hold_hours
            ):
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

        if i == 0:
            continue
        vol_avg = avg_volume(volumes, i)
        if vol_avg is None or vol_avg <= 0:
            continue

        prev = candles[i - 1]
        hourly_change = (candle.close - prev.close) / prev.close
        volume_ratio = candle.volume / vol_avg

        if hourly_change <= -DROP_THRESHOLD and volume_ratio >= VOL_RATIO_THRESHOLD:
            cascades_raw += 1

            rate = funding.rate_at(candle.ts)
            hist = funding.window_rates(candle.ts, LOOKBACK) if rate is not None else []
            if rate is None or len(hist) < MIN_FUNDING_SAMPLES:
                continue
            p_entry = percentile(hist, FUNDING_PERCENTILE)
            if rate > p_entry:
                # Cascade without crowded-short funding confirmation
                continue

            cascades_filtered += 1
            cascade_low = candle.low
            cascade_ref_high = max(candle.open, prev.close)
            drop_size = cascade_ref_high - cascade_low
            if drop_size <= 0:
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

    return trades, equity_curve, cascades_raw, cascades_filtered


def run_period(
    candles_by_symbol: dict[str, list[Candle]],
    funding_by_symbol: dict[str, FundingAsOf],
    start: datetime,
    end: datetime,
    params: Params,
) -> PeriodMetrics:
    per_symbol_capital = STARTING_CAPITAL / len(SYMBOLS)
    all_trades: list[Trade] = []
    curves: dict[str, list[tuple[datetime, float]]] = {}
    end_equities: dict[str, float] = {}
    total_raw = 0
    total_filtered = 0

    for symbol in SYMBOLS:
        sliced = slice_candles(candles_by_symbol[symbol], start, end)
        if len(sliced) < VOL_AVG_PERIOD + 2:
            raise RuntimeError(f"{symbol}: not enough bars in window")
        trades, curve, raw, filtered = run_symbol_combo(
            symbol,
            per_symbol_capital,
            sliced,
            funding_by_symbol[symbol],
            params,
        )
        all_trades.extend(trades)
        curves[symbol] = curve
        end_equities[symbol] = curve[-1][1] if curve else per_symbol_capital
        total_raw += raw
        total_filtered += filtered

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
        cascades_raw=total_raw,
        cascades_filtered=total_filtered,
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
        Params(retrace=r, max_hold_hours=t)
        for r, t in itertools.product(RETRACE_GRID, TIME_GRID)
    ]
    print("Walk-forward CASCADE + FUNDING combo")
    print(f"Symbols ({len(SYMBOLS)}): {', '.join(SYMBOLS)}")
    print(
        f"Fixed detect: drop>={DROP_THRESHOLD * 100:.0f}% vol>={VOL_RATIO_THRESHOLD:.0f}x "
        f"+ funding <= bottom {FUNDING_PERCENTILE:.0f}th pctl (~30d)"
    )
    print(
        f"Grid: retrace={list(RETRACE_GRID)}, time={list(TIME_GRID)}h "
        f"-> {len(combos)} combos"
    )
    print(f"fee={FEE_RATE:.2%} per side | capital={STARTING_CAPITAL:.2f}\n")

    try:
        candles_by_symbol = {
            sym: load_candles(sym, min_bars=VOL_AVG_PERIOD + 5) for sym in SYMBOLS
        }
        funding_by_symbol = {
            sym: FundingAsOf(load_funding(sym)) for sym in SYMBOLS
        }
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        print("Run test_backtest_data.py and test_funding_data.py first.")
        return 1

    full_start = max(c[0].ts for c in candles_by_symbol.values())
    full_end = min(c[-1].ts for c in candles_by_symbol.values()) + timedelta(hours=1)

    # Qualifying count independent of exit grid — use first combo
    sanity = run_period(
        candles_by_symbol,
        funding_by_symbol,
        full_start,
        full_end,
        combos[0],
    )
    print("=== QUALIFYING TRADE SANITY (full ~180d) ===")
    print(f"Raw cascades (drop>=4% vol>=2x): {sanity.cascades_raw}")
    print(
        f"After funding filter (bottom {FUNDING_PERCENTILE:.0f}th pctl): "
        f"{sanity.cascades_filtered}"
    )
    print(f"Trades taken (same as filtered if always flat to enter): {sanity.trades}")
    if sanity.cascades_filtered == 0:
        print(
            "\nSTOP: zero qualifying events after funding filter. "
            "Combo never fires on this sample."
        )
        return 1
    print(
        f"Filter keep-rate: "
        f"{sanity.cascades_filtered / sanity.cascades_raw * 100:.1f}% of raw cascades\n"
        if sanity.cascades_raw
        else "\n"
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

    print(f"Step 1: running all {len(combos)} combos on Period A...")
    a_results: list[tuple[Params, PeriodMetrics]] = []
    for params in combos:
        metrics = run_period(
            candles_by_symbol,
            funding_by_symbol,
            period_a[1],
            period_a[2],
            params,
        )
        a_results.append((params, metrics))
        print(
            f"  A {params.label()}: sharpe={fmt_sharpe(metrics.sharpe)} "
            f"ret={metrics.total_return_pct:.2f}% trades={metrics.trades} "
            f"filtered={metrics.cascades_filtered}"
        )

    a_ranked = sorted(a_results, key=lambda item: sharpe_sort_key(item[1]), reverse=True)
    print("\nPeriod A ranking:")
    for rank, (params, metrics) in enumerate(a_ranked, start=1):
        print(
            f"  #{rank} {params.label()} | "
            f"sharpe={fmt_sharpe(metrics.sharpe)} ret={metrics.total_return_pct:.2f}%"
        )

    print("\nStep 2-3: evaluating ALL 4 combos on Period B and Period C...")
    rows: list[tuple[Params, PeriodMetrics, PeriodMetrics, PeriodMetrics]] = []
    for params, _ in a_ranked:
        a_m = run_period(
            candles_by_symbol, funding_by_symbol, period_a[1], period_a[2], params
        )
        b_m = run_period(
            candles_by_symbol, funding_by_symbol, period_b[1], period_b[2], params
        )
        c_m = run_period(
            candles_by_symbol, funding_by_symbol, period_c[1], period_c[2], params
        )
        rows.append((params, a_m, b_m, c_m))

    print("\n=== WALK-FORWARD TABLE (all 4 combos, ranked by Period A Sharpe) ===\n")
    header = (
        f"{'#':>2} {'params':<20} "
        f"{'A_sharpe':>8} {'A_ret%':>8} "
        f"{'B_sharpe':>8} {'B_ret%':>8} {'B_flag':>10} "
        f"{'C_sharpe':>8} {'C_ret%':>8} {'C_flag':>10}"
    )
    print(header)
    print("-" * len(header))
    for i, (params, a_m, b_m, c_m) in enumerate(rows, start=1):
        print(
            f"{i:>2} {params.label():<20} "
            f"{fmt_sharpe(a_m.sharpe):>8} {a_m.total_return_pct:7.2f}% "
            f"{fmt_sharpe(b_m.sharpe):>8} {b_m.total_return_pct:7.2f}% {held_up(a_m, b_m):>10} "
            f"{fmt_sharpe(c_m.sharpe):>8} {c_m.total_return_pct:7.2f}% {held_up(a_m, c_m):>10}"
        )

    print("\nFlag legend: HELD / WEAK / COLLAPSED / NO_TRADES.")

    strict_both = sum(
        1
        for _, a_m, b_m, c_m in rows
        if held_up(a_m, b_m) == "HELD" and held_up(a_m, c_m) == "HELD"
    )
    collapsed_or_none = sum(
        1
        for _, a_m, b_m, c_m in rows
        if held_up(a_m, b_m) in {"COLLAPSED", "NO_TRADES"}
        or held_up(a_m, c_m) in {"COLLAPSED", "NO_TRADES"}
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
        f"hit COLLAPSED/NO_TRADES on B or C={collapsed_or_none}."
    )
    print(f"Average return: A={avg_a:.2f}% | B={avg_b:.2f}% | C={avg_c:.2f}%")
    print(
        f"Period A quality: {a_positive}/{len(rows)} had positive return AND positive Sharpe."
    )

    if sanity.cascades_filtered < 10:
        print(
            f"\nNOTE: only {sanity.cascades_filtered} qualifying events over ~180d — "
            "sample is extremely thin even before walk-forward."
        )

    if a_positive == 0 and strict_both == 0:
        print(
            "\nALSO FAILS. Cascade+funding combo did not produce a trustworthy edge. "
            "Do not trade this as-is."
        )
    elif strict_both == 0:
        print(
            "\nDOES NOT HOLD UP OUT-OF-SAMPLE. No combo HELD in both B and C. "
            "Do not trade this as-is."
        )
    else:
        print(
            f"\n{strict_both}/{len(rows)} HELD in both B and C. "
            "Treat as a thin hint only — event count is low."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
