"""
Funding-rate extreme long signal + walk-forward validation.

Aligns 8h funding onto hourly prices with as-of (no lookahead).
Entry: funding in bottom P percentile of rolling ~30d history.
Exit: TP / SL / funding back above 50th pct / 72h timeout.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import csv
import itertools
import math
import sys
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from backtesting.backtest_mean_reversion import (
    DATA_DIR,
    STARTING_CAPITAL,
    Candle,
    Trade,
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

PERCENTILE_GRID = (5.0, 10.0, 15.0)  # bottom P%
TP_GRID = (0.005, 0.01, 0.015)
# Grid asks for 2 SL values; use the lower/upper of the stated tests
SL_GRID = (0.015, 0.03)
LOOKBACK = timedelta(days=30)
MAX_HOLD = timedelta(hours=72)
FEE_RATE = 0.0005
MIN_FUNDING_SAMPLES = 10
TOP_N_SELECT = 3
CHUNK_DAYS = 60
ExitReason = Literal["take-profit", "stop-loss", "funding-normalize", "timeout"]


@dataclass
class Params:
    percentile: float
    tp: float
    sl: float

    def label(self) -> str:
        return f"pctl={self.percentile:.0f}% tp=+{self.tp * 100:.1f}% sl=-{self.sl * 100:.1f}%"


@dataclass
class PeriodMetrics:
    trades: int
    win_rate: float
    total_return_pct: float
    sharpe: float | None
    max_dd_pct: float
    entry_signals: int


@dataclass
class FundingPoint:
    ts: datetime
    rate: float


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


def load_funding(symbol: str) -> list[FundingPoint]:
    path = Path(DATA_DIR) / f"{symbol}_funding.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing funding file: {path}")
    points: list[FundingPoint] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            points.append(
                FundingPoint(ts=parse_ts(row["timestamp"]), rate=float(row["fundingRate"]))
            )
    points.sort(key=lambda p: p.ts)
    return points


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


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile, p in [0, 100]."""
    if not values:
        raise ValueError("empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


class FundingAsOf:
    """Most recent funding at or before a timestamp (no lookahead)."""

    def __init__(self, points: list[FundingPoint]):
        self.points = points
        self.ts_list = [p.ts for p in points]

    def rate_at(self, ts: datetime) -> float | None:
        idx = bisect_right(self.ts_list, ts) - 1
        if idx < 0:
            return None
        return self.points[idx].rate

    def window_rates(self, ts: datetime, lookback: timedelta) -> list[float]:
        """Funding prints with lookback_start < print_ts <= ts."""
        end_idx = bisect_right(self.ts_list, ts)
        start_bound = ts - lookback
        start_idx = bisect_right(self.ts_list, start_bound)
        return [self.points[i].rate for i in range(start_idx, end_idx)]


def run_symbol_funding(
    symbol: str,
    starting_cash: float,
    candles: list[Candle],
    funding: FundingAsOf,
    params: Params,
) -> tuple[list[Trade], list[tuple[datetime, float]], int]:
    cash = starting_cash
    equity_curve: list[tuple[datetime, float]] = []
    trades: list[Trade] = []
    entry_signals = 0

    in_pos = False
    entry_idx = -1
    entry_price = 0.0
    qty = 0.0
    entry_notional = 0.0
    entry_fee = 0.0

    for i, candle in enumerate(candles):
        mark = cash + (qty * candle.close if in_pos else 0.0)
        equity_curve.append((candle.ts, mark))

        rate = funding.rate_at(candle.ts)
        hist = funding.window_rates(candle.ts, LOOKBACK) if rate is not None else []
        p50 = percentile(hist, 50.0) if len(hist) >= MIN_FUNDING_SAMPLES else None
        p_entry = (
            percentile(hist, params.percentile)
            if len(hist) >= MIN_FUNDING_SAMPLES
            else None
        )

        if in_pos:
            reason: ExitReason | None = None
            ret = (candle.close - entry_price) / entry_price
            if ret >= params.tp:
                reason = "take-profit"
            elif ret <= -params.sl:
                reason = "stop-loss"
            elif rate is not None and p50 is not None and rate > p50:
                reason = "funding-normalize"
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

        if rate is None or p_entry is None:
            continue

        # Bottom-P: current funding at or below the P-th percentile of recent history
        if rate <= p_entry and cash > 0:
            entry_signals += 1
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

    return trades, equity_curve, entry_signals


def combine_equity(
    curves: dict[str, list[tuple[datetime, float]]],
) -> list[tuple[datetime, float]]:
    all_ts = sorted({ts for curve in curves.values() for ts, _ in curve})
    pointers = {sym: 0 for sym in curves}
    last_val = {sym: curves[sym][0][1] if curves[sym] else 0.0 for sym in curves}
    combined: list[tuple[datetime, float]] = []
    for ts in all_ts:
        total = 0.0
        for sym, curve in curves.items():
            while pointers[sym] < len(curve) and curve[pointers[sym]][0] <= ts:
                last_val[sym] = curve[pointers[sym]][1]
                pointers[sym] += 1
            total += last_val[sym]
        combined.append((ts, total))
    return combined


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
    total_signals = 0

    for symbol in SYMBOLS:
        sliced = slice_candles(candles_by_symbol[symbol], start, end)
        if len(sliced) < 2:
            raise RuntimeError(f"{symbol}: not enough bars in window")
        trades, curve, signals = run_symbol_funding(
            symbol,
            per_symbol_capital,
            sliced,
            funding_by_symbol[symbol],
            params,
        )
        all_trades.extend(trades)
        curves[symbol] = curve
        end_equities[symbol] = curve[-1][1] if curve else per_symbol_capital
        total_signals += signals

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
        entry_signals=total_signals,
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
        Params(percentile=p, tp=tp, sl=sl)
        for p, tp, sl in itertools.product(PERCENTILE_GRID, TP_GRID, SL_GRID)
    ]
    print("Walk-forward FUNDING-RATE extreme long signal")
    print(f"Symbols ({len(SYMBOLS)}): {', '.join(SYMBOLS)}")
    print(
        f"Grid: pctl={list(PERCENTILE_GRID)}, tp={list(TP_GRID)}, sl={list(SL_GRID)} "
        f"-> {len(combos)} combos"
    )
    print(
        f"Lookback~{LOOKBACK.days}d funding history | exits: TP/SL/funding>p50/{MAX_HOLD} | "
        f"fee={FEE_RATE:.2%} per side"
    )
    print("Alignment: hourly bars use latest funding with fundingTime <= candle time (no lookahead)")
    print(f"Top-{TOP_N_SELECT} on Period A Sharpe; validate on B and C\n")

    try:
        candles_by_symbol = {
            sym: load_candles(sym, min_bars=2) for sym in SYMBOLS
        }
        funding_by_symbol = {
            sym: FundingAsOf(load_funding(sym)) for sym in SYMBOLS
        }
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        print("Run test_backtest_data.py and test_funding_data.py first.")
        return 1

    # Full-sample signal sanity check for all combos
    full_start = max(c[0].ts for c in candles_by_symbol.values())
    full_end = min(c[-1].ts for c in candles_by_symbol.values()) + timedelta(hours=1)
    print("=== ENTRY SIGNAL SANITY (full ~180d, all combos) ===")
    signal_rows: list[tuple[Params, int]] = []
    for params in combos:
        m = run_period(
            candles_by_symbol, funding_by_symbol, full_start, full_end, params
        )
        signal_rows.append((params, m.entry_signals))
        print(f"  {params.label()}: entry_signals={m.entry_signals} trades={m.trades}")
    all_signals = [s for _, s in signal_rows]
    print(
        f"\nSignal count range across grid: min={min(all_signals)}, "
        f"median={sorted(all_signals)[len(all_signals) // 2]}, max={max(all_signals)}"
    )
    print(
        f"Total entry signals summed over all combos (not unique events): {sum(all_signals)}\n"
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
            candles_by_symbol, funding_by_symbol, period_a[1], period_a[2], params
        )
        a_results.append((params, metrics))
        if i % 6 == 0 or i == len(combos):
            print(f"  Period A progress: {i}/{len(combos)}", flush=True)

    a_ranked = sorted(a_results, key=lambda item: sharpe_sort_key(item[1]), reverse=True)
    top = a_ranked[:TOP_N_SELECT]

    print(f"\nStep 3: Top {TOP_N_SELECT} by Period A Sharpe:")
    for rank, (params, metrics) in enumerate(top, start=1):
        print(
            f"  #{rank} {params.label()} | "
            f"A sharpe={fmt_sharpe(metrics.sharpe)} "
            f"ret={metrics.total_return_pct:.2f}% "
            f"trades={metrics.trades} signals={metrics.entry_signals}"
        )

    print("\nStep 4: evaluating top combos on Period B and Period C...")
    rows: list[tuple[Params, PeriodMetrics, PeriodMetrics, PeriodMetrics]] = []
    for params, _ in top:
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

    print("\n=== WALK-FORWARD TABLE (Top 3 from Period A) ===\n")
    header = (
        f"{'#':>2} {'params':<36} "
        f"{'A_sharpe':>8} {'A_ret%':>8} "
        f"{'B_sharpe':>8} {'B_ret%':>8} {'B_flag':>10} "
        f"{'C_sharpe':>8} {'C_ret%':>8} {'C_flag':>10}"
    )
    print(header)
    print("-" * len(header))
    for i, (params, a_m, b_m, c_m) in enumerate(rows, start=1):
        print(
            f"{i:>2} {params.label():<36} "
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

    if a_positive == 0:
        print(
            "\nALSO FAILS. Period A top combos were not good in-sample. "
            "Funding-extreme long signal does not hold up."
        )
    elif strict_both == 0:
        print(
            "\nALSO FAILS OUT-OF-SAMPLE. Best Period-A funding combos did not HELD in both "
            "B and C. Same blunt standard as mean-reversion / momentum / cross-sectional: "
            "do not trade this setup as-is."
        )
    else:
        print(
            f"\n{strict_both}/{TOP_N_SELECT} HELD in both B and C. "
            "Only a possible hint - still not automatic approval for live trading."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
