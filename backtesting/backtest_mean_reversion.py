"""
Mean-reversion backtest on local 1h CSVs (BTCUSDT, ETHUSDT, SOLUSDT).

Long-only first pass: enter when close is >=2% below 20-SMA;
exit on revert to within 0.3% of SMA, -4% stop, or 24h timeout.
No leverage. Flat 0.05% fee on entry and exit.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import csv
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
SMA_PERIOD = 20
ENTRY_DEV = -0.02
TAKE_PROFIT_DEV = -0.003  # recovered to within 0.3% below SMA (or better)
STOP_LOSS_DEV = -0.04
MAX_HOLD = timedelta(hours=24)
FEE_RATE = 0.0005  # 0.05% per trade side
STARTING_CAPITAL = 1000.0
# Hourly bars → annualize Sharpe with 24*365 periods/year
HOURS_PER_YEAR = 24 * 365
ExitReason = Literal["take-profit", "stop-loss", "timeout"]


@dataclass
class Candle:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Trade:
    symbol: str
    entry_ts: datetime
    exit_ts: datetime
    entry_price: float
    exit_price: float
    qty: float
    exit_reason: ExitReason
    pnl: float
    return_pct: float


def parse_ts(value: str) -> datetime:
    # CSV written as ISO with +00:00
    return datetime.fromisoformat(value)


def load_candles(symbol: str, min_bars: int = SMA_PERIOD + 1) -> list[Candle]:
    path = DATA_DIR / f"{symbol}_1h.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    candles: list[Candle] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            candles.append(
                Candle(
                    ts=parse_ts(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    if len(candles) < min_bars:
        raise RuntimeError(f"{symbol}: not enough candles ({len(candles)} < {min_bars})")
    return candles


def sma_at(closes: list[float], index: int, period: int) -> float | None:
    if index < period - 1:
        return None
    window = closes[index - period + 1 : index + 1]
    return sum(window) / period


def max_drawdown_pct(equity: list[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for value in equity:
        if value > peak:
            peak = value
        if peak > 0:
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd * 100.0


def sharpe_from_equity(equity: list[float]) -> float | None:
    """
    Sharpe from hourly equity returns, annualized with sqrt(24*365).
    Uses population stdev; returns None if undefined.
    """
    if len(equity) < 2:
        return None
    returns: list[float] = []
    for i in range(1, len(equity)):
        prev = equity[i - 1]
        if prev <= 0:
            continue
        returns.append((equity[i] - prev) / prev)
    if len(returns) < 2:
        return None
    mu = mean(returns)
    sigma = pstdev(returns)
    if sigma == 0:
        return 0.0 if mu == 0 else float("inf") if mu > 0 else float("-inf")
    return (mu / sigma) * math.sqrt(HOURS_PER_YEAR)


def run_symbol(
    symbol: str,
    starting_cash: float,
    *,
    sma_period: int = SMA_PERIOD,
    entry_dev: float = ENTRY_DEV,
    take_profit_dev: float = TAKE_PROFIT_DEV,
    stop_loss_dev: float = STOP_LOSS_DEV,
    max_hold: timedelta = MAX_HOLD,
    fee_rate: float = FEE_RATE,
    candles: list[Candle] | None = None,
) -> tuple[list[Trade], list[tuple[datetime, float]]]:
    candles = candles if candles is not None else load_candles(symbol, min_bars=sma_period + 1)
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

    for i, candle in enumerate(candles):
        sma = sma_at(closes, i, sma_period)
        mark_equity = cash
        if in_pos:
            # MTM position value minus sunk entry fee (exit fee applied only on close)
            mark_equity = qty * candle.close - entry_fee
        equity_curve.append((candle.ts, mark_equity))

        if sma is None:
            continue

        deviation = (candle.close - sma) / sma

        if in_pos:
            reason: ExitReason | None = None
            if deviation <= stop_loss_dev:
                reason = "stop-loss"
            elif deviation >= take_profit_dev:
                # within TP band of SMA from below, or above SMA
                reason = "take-profit"
            elif candle.ts - candles[entry_idx].ts >= max_hold:
                reason = "timeout"

            if reason is not None:
                exit_price = candle.close
                exit_notional = qty * exit_price
                exit_fee = exit_notional * fee_rate
                # Fees on both sides; qty sized from full capital at entry
                pnl = (exit_notional - entry_notional) - entry_fee - exit_fee
                cash = entry_notional + pnl
                ret = pnl / entry_notional if entry_notional else 0.0
                trades.append(
                    Trade(
                        symbol=symbol,
                        entry_ts=candles[entry_idx].ts,
                        exit_ts=candle.ts,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        qty=qty,
                        exit_reason=reason,
                        pnl=pnl,
                        return_pct=ret * 100.0,
                    )
                )
                in_pos = False
                qty = 0.0
                equity_curve[-1] = (candle.ts, cash)
            continue

        # Flat: entry check (long only, one position)
        if deviation <= entry_dev and cash > 0:
            entry_price = candle.close
            entry_notional = cash
            entry_fee = entry_notional * fee_rate
            qty = entry_notional / entry_price
            cash = 0.0
            in_pos = True
            entry_idx = i
            # Mark equity net of entry fee immediately
            equity_curve[-1] = (candle.ts, entry_notional - entry_fee)

    # If still open at end of data, force mark exit as timeout for accounting clarity
    if in_pos:
        last = candles[-1]
        exit_price = last.close
        exit_notional = qty * exit_price
        exit_fee = exit_notional * fee_rate
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
                exit_reason="timeout",
                pnl=pnl,
                return_pct=(pnl / entry_notional) * 100.0 if entry_notional else 0.0,
            )
        )
        equity_curve[-1] = (last.ts, cash)

    return trades, equity_curve


def print_report(
    title: str,
    trades: list[Trade],
    start_capital: float,
    end_equity: float,
    equity_values: list[float],
) -> None:
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    total_pnl = sum(t.pnl for t in trades)
    # Prefer equity-based PnL (includes fees already in trade pnl)
    equity_pnl = end_equity - start_capital
    win_rate = (wins / n * 100.0) if n else 0.0
    ret_pct = (equity_pnl / start_capital * 100.0) if start_capital else 0.0
    sharpe = sharpe_from_equity(equity_values)
    mdd = max_drawdown_pct(equity_values)
    reasons = {"take-profit": 0, "stop-loss": 0, "timeout": 0}
    for t in trades:
        reasons[t.exit_reason] += 1

    print(f"=== {title} ===")
    print(f"Trades: {n}")
    if n == 0:
        print("Win rate: n/a (no trades)")
    else:
        print(f"Win rate: {win_rate:.1f}% ({wins}/{n} profitable)")
    print(f"Total P&L: {equity_pnl:.4f} USDT ({ret_pct:.2f}% return)")
    print(f"  (sum of per-trade P&L after fees: {total_pnl:.4f} USDT)")
    if sharpe is None:
        print("Sharpe ratio: n/a (insufficient return samples)")
    elif math.isinf(sharpe):
        print(f"Sharpe ratio: {sharpe} (zero volatility)")
    else:
        print(
            f"Sharpe ratio: {sharpe:.3f} "
            f"(hourly returns, annualized with sqrt({HOURS_PER_YEAR}) = sqrt(24*365))"
        )
    print(f"Max drawdown: {mdd:.2f}%")
    print(
        "Exit reasons: "
        f"take-profit={reasons['take-profit']}, "
        f"stop-loss={reasons['stop-loss']}, "
        f"timeout={reasons['timeout']}"
    )
    print()


def combine_equity(
    curves: dict[str, list[tuple[datetime, float]]],
) -> list[tuple[datetime, float]]:
    """Sum per-symbol marked equity on the union of timestamps (forward-fill)."""
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


def main() -> int:
    per_symbol_capital = STARTING_CAPITAL / len(SYMBOLS)
    print("Mean-reversion backtest (long-only, no leverage)")
    print(
        f"Params: SMA={SMA_PERIOD}, entry<= {ENTRY_DEV:.0%}, "
        f"TP>= {TAKE_PROFIT_DEV:.1%}, SL<= {STOP_LOSS_DEV:.0%}, "
        f"max hold=24h, fee={FEE_RATE:.2%} per side"
    )
    print(
        f"Capital: {STARTING_CAPITAL:.2f} USDT total, "
        f"{per_symbol_capital:.2f} USDT per symbol\n"
    )

    all_trades: list[Trade] = []
    curves: dict[str, list[tuple[datetime, float]]] = {}
    end_equities: dict[str, float] = {}

    try:
        for symbol in SYMBOLS:
            trades, curve = run_symbol(symbol, per_symbol_capital)
            all_trades.extend(trades)
            curves[symbol] = curve
            end_equity = curve[-1][1] if curve else per_symbol_capital
            end_equities[symbol] = end_equity
            print_report(
                symbol,
                trades,
                per_symbol_capital,
                end_equity,
                [v for _, v in curve],
            )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        print("Run test_backtest_data.py first to download CSVs.")
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    combined_curve = combine_equity(curves)
    combined_end = sum(end_equities.values())
    print_report(
        "COMBINED (all symbols)",
        all_trades,
        STARTING_CAPITAL,
        combined_end,
        [v for _, v in combined_curve],
    )

    # Honest summary line
    combined_pnl = combined_end - STARTING_CAPITAL
    if combined_pnl < 0:
        print(
            f"VERDICT: Strategy LOST money over this ~30-day sample "
            f"({combined_pnl:.2f} USDT / {combined_pnl / STARTING_CAPITAL * 100:.2f}%). "
            "First-pass only — not a final verdict."
        )
    elif combined_pnl == 0:
        print("VERDICT: Flat result (no net P&L) on this sample. First-pass only.")
    else:
        print(
            f"VERDICT: Strategy made {combined_pnl:.2f} USDT "
            f"({combined_pnl / STARTING_CAPITAL * 100:.2f}%) on this ~30-day sample. "
            "First-pass only - not a final verdict; short sample, no costs beyond flat fee."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
