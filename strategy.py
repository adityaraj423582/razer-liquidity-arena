# Opportunistic cascade+funding longs (BTC/ETH). Flat when no signal. Not live-wired yet.

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from backtest_funding_signal import (
    LOOKBACK,
    MIN_FUNDING_SAMPLES,
    FundingAsOf,
    load_funding,
    percentile,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
DROP_THRESHOLD = 0.04
VOL_RATIO_THRESHOLD = 2.0
FUNDING_PERCENTILE = 20.0
VOL_AVG_PERIOD = 24
TAKE_PROFIT = 0.005
STOP_LOSS = 0.015
RISK_FRACTION = 0.01
STARTING_EQUITY = 1000.0
CIRCUIT_BREAKER = 900.0
DISQUALIFICATION_FLOOR = 800.0
FEE_RATE = 0.0005
MAX_POSITIONS = 2


@dataclass
class Candle:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Position:
    symbol: str
    entry_price: float
    qty: float
    notional: float
    tp_price: float
    sl_price: float
    entry_index: int


@dataclass
class Trade:
    symbol: str
    entry_ts: datetime
    exit_ts: datetime
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    exit_reason: str


def check_entry_signal(
    candles: list[Candle],
    index: int,
    funding: FundingAsOf,
) -> bool:
    # True on the bar AFTER a qualifying cascade (enter this bar)
    cascade_i = index - 1
    if cascade_i < VOL_AVG_PERIOD:
        return False
    cur = candles[cascade_i]
    prev = candles[cascade_i - 1]
    vol_window = [c.volume for c in candles[cascade_i - VOL_AVG_PERIOD + 1 : cascade_i + 1]]
    vol_avg = sum(vol_window) / VOL_AVG_PERIOD
    if vol_avg <= 0:
        return False
    hourly_change = (cur.close - prev.close) / prev.close
    volume_ratio = cur.volume / vol_avg
    if hourly_change > -DROP_THRESHOLD or volume_ratio < VOL_RATIO_THRESHOLD:
        return False
    rate = funding.rate_at(cur.ts)
    hist = funding.window_rates(cur.ts, LOOKBACK) if rate is not None else []
    if rate is None or len(hist) < MIN_FUNDING_SAMPLES:
        return False
    return rate <= percentile(hist, FUNDING_PERCENTILE)


def calculate_position_size(equity: float, entry_price: float) -> tuple[float, float]:
    # Risk 1% of equity at STOP_LOSS distance
    if equity <= 0 or entry_price <= 0:
        return 0.0, 0.0
    risk_usdt = equity * RISK_FRACTION
    notional = risk_usdt / STOP_LOSS
    qty = notional / entry_price
    return notional, qty


def check_circuit_breaker(equity: float) -> bool:
    # True = halt new entries
    return equity < CIRCUIT_BREAKER


def _load_candles(symbol: str) -> list[Candle]:
    path = DATA_DIR / f"{symbol}_1h.csv"
    rows: list[Candle] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                Candle(
                    ts=datetime.fromisoformat(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    return rows


def simulate() -> tuple[list[Trade], list[float], dict[str, float]]:
    series = {sym: _load_candles(sym) for sym in SYMBOLS}
    funding = {sym: FundingAsOf(load_funding(sym)) for sym in SYMBOLS}
    n = min(len(series[s]) for s in SYMBOLS)
    candles = {sym: series[sym][:n] for sym in SYMBOLS}
    timestamps = [candles[SYMBOLS[0]][i].ts for i in range(n)]

    cash = STARTING_EQUITY
    positions: dict[str, Position] = {}
    trades: list[Trade] = []
    equity_curve: list[float] = []
    risk_fractions: list[float] = []
    entries_blocked_by_breaker = 0
    last_close = {sym: candles[sym][0].close for sym in SYMBOLS}

    for i in range(n):
        for sym in SYMBOLS:
            last_close[sym] = candles[sym][i].close

        equity = cash + sum(pos.qty * last_close[sym] for sym, pos in positions.items())
        equity_curve.append(equity)

        # Exits (TP/SL brackets)
        for sym in list(positions.keys()):
            pos = positions[sym]
            px = last_close[sym]
            reason = None
            exit_px = px
            if px >= pos.tp_price:
                reason = "take-profit"
                exit_px = pos.tp_price
            elif px <= pos.sl_price:
                reason = "stop-loss"
                exit_px = pos.sl_price
            if reason is None:
                continue
            exit_notional = pos.qty * exit_px
            exit_fee = exit_notional * FEE_RATE
            entry_fee = pos.notional * FEE_RATE
            pnl = (exit_notional - pos.notional) - entry_fee - exit_fee
            cash += exit_notional - exit_fee
            trades.append(
                Trade(
                    symbol=sym,
                    entry_ts=timestamps[pos.entry_index],
                    exit_ts=timestamps[i],
                    entry_price=pos.entry_price,
                    exit_price=exit_px,
                    qty=pos.qty,
                    pnl=pnl,
                    exit_reason=reason,
                )
            )
            del positions[sym]

        equity = cash + sum(pos.qty * last_close[sym] for sym, pos in positions.items())
        breaker_on = check_circuit_breaker(equity)

        # Entries only on cascade+funding; otherwise stay flat
        for sym in SYMBOLS:
            if sym in positions:
                continue
            if len(positions) >= MAX_POSITIONS:
                break
            if not check_entry_signal(candles[sym], i, funding[sym]):
                continue
            if breaker_on:
                entries_blocked_by_breaker += 1
                continue

            entry_price = candles[sym][i].open
            notional, qty = calculate_position_size(equity, entry_price)
            if notional > cash:
                notional = cash
                qty = notional / entry_price if entry_price > 0 else 0.0
            if notional <= 0 or qty <= 0:
                continue

            risk_usdt = notional * STOP_LOSS
            risk_frac = risk_usdt / equity if equity > 0 else 0.0
            if risk_frac > RISK_FRACTION + 1e-9:
                continue
            risk_fractions.append(risk_frac)

            entry_fee = notional * FEE_RATE
            cash -= notional
            if cash >= entry_fee:
                cash -= entry_fee
            positions[sym] = Position(
                symbol=sym,
                entry_price=entry_price,
                qty=qty,
                notional=notional,
                tp_price=entry_price * (1.0 + TAKE_PROFIT),
                sl_price=entry_price * (1.0 - STOP_LOSS),
                entry_index=i,
            )

        equity_curve[-1] = cash + sum(
            pos.qty * last_close[sym] for sym, pos in positions.items()
        )

    for sym, pos in list(positions.items()):
        px = last_close[sym]
        exit_notional = pos.qty * px
        exit_fee = exit_notional * FEE_RATE
        entry_fee = pos.notional * FEE_RATE
        pnl = (exit_notional - pos.notional) - entry_fee - exit_fee
        cash += exit_notional - exit_fee
        trades.append(
            Trade(
                symbol=sym,
                entry_ts=timestamps[pos.entry_index],
                exit_ts=timestamps[-1],
                entry_price=pos.entry_price,
                exit_price=px,
                qty=pos.qty,
                pnl=pnl,
                exit_reason="end-of-data",
            )
        )
        del positions[sym]
    if equity_curve:
        equity_curve[-1] = cash

    stats = {
        "final_equity": cash,
        "min_equity": min(equity_curve) if equity_curve else STARTING_EQUITY,
        "max_risk_fraction": max(risk_fractions) if risk_fractions else 0.0,
        "entries_blocked_by_breaker": float(entries_blocked_by_breaker),
        "trades": float(len(trades)),
    }
    return trades, equity_curve, stats


def run_unit_checks() -> int:
    assert check_circuit_breaker(899.99) is True
    assert check_circuit_breaker(900.0) is False

    notional, qty = calculate_position_size(1000.0, 50000.0)
    assert abs(notional - (1000.0 * RISK_FRACTION / STOP_LOSS)) < 1e-9
    assert abs((notional * STOP_LOSS) / 1000.0 - RISK_FRACTION) < 1e-12

    # Synthetic cascade bar then entry bar; funding forced extreme-negative
    class _FakeFunding:
        def rate_at(self, ts: datetime) -> float:
            return -0.01

        def window_rates(self, ts: datetime, lookback: timedelta) -> list[float]:
            return [0.0] * 20 + [-0.01]

    base_ts = datetime(2026, 1, 1)
    candles = []
    for i in range(30):
        candles.append(
            Candle(
                ts=base_ts + timedelta(hours=i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=100.0,
            )
        )
    # Cascade at index 28: -5% close-to-close, 3x volume
    candles[27] = Candle(candles[27].ts, 100.0, 100.0, 100.0, 100.0, 100.0)
    candles[28] = Candle(candles[28].ts, 100.0, 100.0, 94.0, 95.0, 300.0)
    candles[29] = Candle(candles[29].ts, 95.0, 96.0, 94.5, 95.5, 100.0)
    assert check_entry_signal(candles, 29, _FakeFunding()) is True  # type: ignore[arg-type]
    assert check_entry_signal(candles, 28, _FakeFunding()) is False  # type: ignore[arg-type]
    print("Unit checks passed: breaker, 1% risk sizing, cascade+funding entry.")

    trades, _equity_curve, stats = simulate()
    n_trades = len(trades)
    min_eq = stats["min_equity"]
    max_risk = stats["max_risk_fraction"]
    final_eq = stats["final_equity"]
    blocked = int(stats["entries_blocked_by_breaker"])
    pnl = final_eq - STARTING_EQUITY

    by_sym = {s: 0 for s in SYMBOLS}
    reasons: dict[str, int] = {}
    for t in trades:
        by_sym[t.symbol] += 1
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    print("\n=== HISTORICAL SIM (~180d, BTC+ETH, cascade+funding only) ===")
    print(f"Trades: {n_trades}")
    print(f"  by symbol: {by_sym}")
    print(f"  exits: {reasons}")
    print(f"Final equity: {final_eq:.2f} USDT")
    print(f"P&L: {pnl:+.2f} USDT ({pnl / STARTING_EQUITY * 100:+.2f}%)")
    print(f"Min equity: {min_eq:.2f} USDT")
    print(f"Max risk/equity on any entry: {max_risk * 100:.3f}%")
    print(f"Entries blocked by circuit breaker: {blocked}")

    if abs(pnl) < 1.0:
        verdict = "essentially FLAT / breakeven"
    elif pnl < 0:
        verdict = "still LOSES money"
    else:
        verdict = "slightly PROFITABLE (tiny sample — not proof)"
    print(f"\nVERDICT: {verdict}")

    ok_risk = max_risk <= RISK_FRACTION + 1e-9 if risk_fractions_ok(stats) else True
    if n_trades > 0:
        ok_risk = max_risk <= RISK_FRACTION + 1e-9
    print(f"Risk never > 1%/trade: {'YES' if ok_risk or n_trades == 0 else 'NO'}")
    print(f"Min equity above 800: {'YES' if min_eq > DISQUALIFICATION_FLOOR else 'NO'}")

    if n_trades > 0 and max_risk > RISK_FRACTION + 1e-9:
        return 1
    if min_eq <= DISQUALIFICATION_FLOOR:
        return 1
    return 0


def risk_fractions_ok(stats: dict[str, float]) -> bool:
    return stats.get("trades", 0) > 0


if __name__ == "__main__":
    sys.exit(run_unit_checks())
