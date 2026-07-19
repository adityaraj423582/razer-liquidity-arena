"""
Walk-forward: BTC trailing return -> same-direction next-bar trade in
BTC-correlated altcoins (official competition_universe.json).

Discipline (same as prior sweeps):
  - Period A / B / C = sequential 60-day chunks
  - Correlation ranking + param tune ONLY on Period A
  - PASS: return>0 AND Sharpe>0 on BOTH B and C independently

Research only — does not modify strategy.py or live_trading_loop.py.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Literal

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtesting.backtest_mean_reversion import (  # noqa: E402
    DATA_DIR,
    FEE_RATE,
    STARTING_CAPITAL,
    Candle,
    Trade,
    load_candles,
    max_drawdown_pct,
    sharpe_from_equity,
)
from multi_strategy.multi_strategy_sweep import (  # noqa: E402
    CHUNK_DAYS,
    HISTORY_DAYS,
    MIN_BARS,
    PeriodMetrics,
    download_klines,
    metrics_from_book,
)

UNIVERSE_PATH = _ROOT / "competition_universe.json"
RESULTS_PATH = Path(__file__).resolve().parent / "multi_strategy_sweep_results.txt"
RISK_FRACTION = 0.01
BTC = "BTCUSDT"

LAGS = (1, 2, 3)
THRESHOLDS = (0.003, 0.005, 0.008, 0.010, 0.015)  # fractional BTC move
TOP_KS = (5, 8, 10)
HOLD_BARS = 1  # next 1h return (lead-lag target)

Side = Literal["long", "short"]


@dataclass
class ComboResult:
    label: str
    lag: int
    thresh: float
    top_k: int
    alts: list[str]
    a: PeriodMetrics
    b: PeriodMetrics
    c: PeriodMetrics

    @property
    def pass_bar(self) -> bool:
        return (
            self.b.total_return_pct > 0
            and self.c.total_return_pct > 0
            and self.b.sharpe is not None
            and self.c.sharpe is not None
            and self.b.sharpe > 0
            and self.c.sharpe > 0
        )


def load_universe() -> list[str]:
    meta = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    symbols = list(meta["symbols"])
    if BTC not in symbols:
        raise RuntimeError(f"{BTC} missing from competition_universe.json")
    return symbols


def ensure_klines(symbols: list[str]) -> list[str]:
    """Download missing/short 1h CSVs. Does not rewrite competition_universe.json."""
    kept: list[str] = []
    for symbol in symbols:
        path = DATA_DIR / f"{symbol}_1h.csv"
        n_bars = 0
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                n_bars = sum(1 for _ in handle) - 1
        try:
            if n_bars < MIN_BARS:
                print(f"  download klines {symbol} (have {n_bars}) ...")
                n_bars = download_klines(symbol, days=HISTORY_DAYS)
            if n_bars < MIN_BARS:
                print(f"  SKIP {symbol}: only {n_bars} bars (< {MIN_BARS})")
                continue
            kept.append(symbol)
            print(f"  OK {symbol}: {n_bars} bars")
        except Exception as exc:
            print(f"  SKIP {symbol}: {exc}")
    return kept


def slice_candles(
    candles: list[Candle], start: datetime, end: datetime
) -> list[Candle]:
    return [c for c in candles if start <= c.ts < end]


def make_periods(
    btc: list[Candle],
) -> tuple[datetime, list[tuple[str, datetime, datetime]]]:
    global_start = btc[0].ts
    global_end = btc[-1].ts
    a0 = global_start
    b0 = a0 + timedelta(days=CHUNK_DAYS)
    c0 = b0 + timedelta(days=CHUNK_DAYS)
    c1 = c0 + timedelta(days=CHUNK_DAYS)
    if c1 > global_end + timedelta(hours=1):
        c1 = global_end + timedelta(hours=1)
    return global_start, [("A", a0, b0), ("B", b0, c0), ("C", c0, c1)]


def hourly_returns(closes: list[float]) -> list[float | None]:
    out: list[float | None] = [None]
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        out.append(None if prev == 0 else (closes[i] - prev) / prev)
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 30 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def align_closes(
    btc: list[Candle], alt: list[Candle]
) -> tuple[list[datetime], list[float], list[float]]:
    alt_map = {c.ts: c.close for c in alt}
    ts: list[datetime] = []
    b: list[float] = []
    a: list[float] = []
    for c in btc:
        if c.ts in alt_map:
            ts.append(c.ts)
            b.append(c.close)
            a.append(alt_map[c.ts])
    return ts, b, a


def corr_on_period(
    btc_full: list[Candle],
    alt_full: list[Candle],
    start: datetime,
    end: datetime,
) -> float | None:
    btc = slice_candles(btc_full, start, end)
    alt = slice_candles(alt_full, start, end)
    _, b_closes, a_closes = align_closes(btc, alt)
    if len(b_closes) < 100:
        return None
    br = hourly_returns(b_closes)
    ar = hourly_returns(a_closes)
    xs: list[float] = []
    ys: list[float] = []
    for i in range(1, len(br)):
        if br[i] is None or ar[i] is None:
            continue
        xs.append(br[i])  # type: ignore[arg-type]
        ys.append(ar[i])  # type: ignore[arg-type]
    return pearson(xs, ys)


def rank_alts_by_btc_corr(
    btc: list[Candle],
    candles: dict[str, list[Candle]],
    start: datetime,
    end: datetime,
) -> list[tuple[str, float]]:
    ranked: list[tuple[str, float]] = []
    for sym, series in candles.items():
        if sym == BTC:
            continue
        c = corr_on_period(btc, series, start, end)
        if c is None:
            continue
        # Same-direction lead-lag needs positive correlation
        if c <= 0:
            continue
        ranked.append((sym, c))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def trailing_ret(closes: list[float], i: int, lag: int) -> float | None:
    """BTC return over the prior `lag` completed bars ending at i-1 (signal for bar i)."""
    end = i - 1
    start = end - lag
    if start < 0 or end < 0:
        return None
    base = closes[start]
    if base == 0:
        return None
    return (closes[end] - base) / base


def run_leadlag_period(
    btc: list[Candle],
    candles: dict[str, list[Candle]],
    alts: list[str],
    start: datetime,
    end: datetime,
    lag: int,
    thresh: float,
) -> PeriodMetrics:
    btc_p = slice_candles(btc, start, end)
    if len(btc_p) < lag + HOLD_BARS + 5:
        return PeriodMetrics(0, 0.0, None, 0.0)

    # Align each alt to BTC period timestamps
    alt_close: dict[str, dict[datetime, float]] = {}
    alt_open: dict[str, dict[datetime, float]] = {}
    for sym in alts:
        series = slice_candles(candles[sym], start, end)
        alt_close[sym] = {c.ts: c.close for c in series}
        alt_open[sym] = {c.ts: c.open for c in series}

    btc_closes = [c.close for c in btc_p]
    equity = STARTING_CAPITAL
    equity_curve: list[float] = []
    trades: list[Trade] = []

    # open positions: sym -> (side, entry_price, qty, entry_fee, entry_idx, entry_ts)
    open_pos: dict[str, tuple[Side, float, float, float, int, datetime]] = {}

    for i, bar in enumerate(btc_p):
        # Mark-to-market + exits first
        still: dict[str, tuple[Side, float, float, float, int, datetime]] = {}
        for sym, (side, entry_px, qty, entry_fee, entry_i, entry_ts) in open_pos.items():
            px = alt_close[sym].get(bar.ts)
            if px is None:
                still[sym] = (side, entry_px, qty, entry_fee, entry_i, entry_ts)
                continue
            held = i - entry_i
            if held >= HOLD_BARS:
                exit_fee = qty * px * FEE_RATE
                if side == "long":
                    pnl = qty * (px - entry_px) - entry_fee - exit_fee
                else:
                    pnl = qty * (entry_px - px) - entry_fee - exit_fee
                equity += pnl
                trades.append(
                    Trade(
                        symbol=sym,
                        entry_ts=entry_ts,
                        exit_ts=bar.ts,
                        entry_price=entry_px,
                        exit_price=px,
                        qty=qty,
                        exit_reason="timeout",  # type: ignore[arg-type]
                        pnl=pnl,
                        return_pct=(pnl / (qty * entry_px)) * 100.0 if entry_px else 0.0,
                    )
                )
            else:
                still[sym] = (side, entry_px, qty, entry_fee, entry_i, entry_ts)
        open_pos = still

        # Entries at this bar's open (signal from prior lag ending previous close)
        trail = trailing_ret(btc_closes, i, lag)
        if trail is not None and abs(trail) >= thresh:
            side: Side = "long" if trail > 0 else "short"
            for sym in alts:
                if sym in open_pos:
                    continue
                entry_px = alt_open[sym].get(bar.ts)
                if entry_px is None or entry_px <= 0:
                    continue
                # Need HOLD_BARS bars of room
                if i + HOLD_BARS >= len(btc_p):
                    continue
                risk_budget = equity * RISK_FRACTION
                # Stop distance proxy: use thresh as scale (no attached stop — 1h hold)
                # Size so |1% move| ≈ risk_budget (same spirit as 1% risk)
                stop_dist = max(thresh, 0.005)
                notional = risk_budget / stop_dist
                qty = notional / entry_px
                if qty <= 0:
                    continue
                entry_fee = notional * FEE_RATE
                # Reserve fee drag into equity book via trade PnL at exit only;
                # track entry_fee for netting.
                open_pos[sym] = (side, entry_px, qty, entry_fee, i, bar.ts)

        # Equity mark
        mtm = equity
        for sym, (side, entry_px, qty, entry_fee, _ei, _ets) in open_pos.items():
            px = alt_close[sym].get(bar.ts, entry_px)
            if side == "long":
                mtm += qty * (px - entry_px) - entry_fee
            else:
                mtm += qty * (entry_px - px) - entry_fee
        equity_curve.append(mtm)

    # Force-close leftovers at last bar
    if btc_p and open_pos:
        last = btc_p[-1]
        for sym, (side, entry_px, qty, entry_fee, _ei, entry_ts) in list(open_pos.items()):
            px = alt_close[sym].get(last.ts, entry_px)
            exit_fee = qty * px * FEE_RATE
            if side == "long":
                pnl = qty * (px - entry_px) - entry_fee - exit_fee
            else:
                pnl = qty * (entry_px - px) - entry_fee - exit_fee
            equity += pnl
            trades.append(
                Trade(
                    symbol=sym,
                    entry_ts=entry_ts,
                    exit_ts=last.ts,
                    entry_price=entry_px,
                    exit_price=px,
                    qty=qty,
                    exit_reason="timeout",  # type: ignore[arg-type]
                    pnl=pnl,
                    return_pct=(pnl / (qty * entry_px)) * 100.0 if entry_px else 0.0,
                )
            )
        if equity_curve:
            equity_curve[-1] = equity

    return metrics_from_book(trades, equity_curve, equity)


def fmt_m(m: PeriodMetrics) -> str:
    sh = f"{m.sharpe:6.2f}" if m.sharpe is not None and math.isfinite(m.sharpe) else "   n/a"
    return (
        f"sharpe={sh} ret={m.total_return_pct:+7.2f}% "
        f"trades={m.trades:4d} dd={m.max_dd_pct:6.2f}%"
    )


def main() -> int:
    print("=== BTC lead-lag walk-forward (official 50-universe) ===")
    symbols = load_universe()
    print(f"Universe symbols: {len(symbols)}")
    print("Ensuring 1h klines ...")
    available = ensure_klines(symbols)
    if BTC not in available:
        raise RuntimeError("BTCUSDT data unavailable")
    alts_available = [s for s in available if s != BTC]
    print(f"Usable: BTC + {len(alts_available)} alts")

    print("Loading candles ...")
    candles = {s: load_candles(s, min_bars=500) for s in available}
    btc = candles[BTC]
    _, periods = make_periods(btc)
    a_name, a0, a1 = periods[0]
    b_name, b0, b1 = periods[1]
    c_name, c0, c1 = periods[2]
    assert a_name == "A" and b_name == "B" and c_name == "C"

    print(f"Period A: {a0.isoformat()} -> {a1.isoformat()}")
    print(f"Period B: {b0.isoformat()} -> {b1.isoformat()}")
    print(f"Period C: {c0.isoformat()} -> {c1.isoformat()}")

    print("Ranking alts by Period-A correlation with BTC (positive only) ...")
    ranked = rank_alts_by_btc_corr(btc, candles, a0, a1)
    print(f"  positively correlated alts with enough A overlap: {len(ranked)}")
    for i, (sym, c) in enumerate(ranked[:15], 1):
        print(f"  #{i:2d} {sym:16} corr_A={c:.4f}")

    if len(ranked) < 5:
        raise RuntimeError(
            f"Need >=5 positively correlated alts on A; got {len(ranked)}"
        )

    rows: list[ComboResult] = []
    for lag in LAGS:
        for thresh in THRESHOLDS:
            for top_k in TOP_KS:
                if len(ranked) < top_k:
                    continue
                chosen = [sym for sym, _ in ranked[:top_k]]
                label = (
                    f"btcLead lag={lag}h thr={thresh*100:.1f}% top{top_k} "
                    f"hold={HOLD_BARS}h alts={','.join(s.replace('USDT','') for s in chosen)}"
                )
                print(f"run {label[:90]} ...")
                ma = run_leadlag_period(btc, candles, chosen, a0, a1, lag, thresh)
                mb = run_leadlag_period(btc, candles, chosen, b0, b1, lag, thresh)
                mc = run_leadlag_period(btc, candles, chosen, c0, c1, lag, thresh)
                rows.append(
                    ComboResult(
                        label=label,
                        lag=lag,
                        thresh=thresh,
                        top_k=top_k,
                        alts=chosen,
                        a=ma,
                        b=mb,
                        c=mc,
                    )
                )

    # Select best on A by Sharpe (tune reference)
    def a_key(r: ComboResult) -> float:
        if r.a.sharpe is None or not math.isfinite(r.a.sharpe):
            return float("-inf")
        return r.a.sharpe

    rows_sorted_a = sorted(rows, key=a_key, reverse=True)
    best_a = rows_sorted_a[0] if rows_sorted_a else None
    passed = [r for r in rows if r.pass_bar]

    lines: list[str] = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("=== ADDON: BTC->altcoin lead-lag (final new hypothesis this cycle) ===")
    lines.append(
        "Hypothesis: BTC trailing 1-3h return predicts next 1h same-direction "
        "move in BTC-correlated alts."
    )
    lines.append(
        "Universe: official competition_universe.json 50-list "
        f"({len(available)} symbols with >= {MIN_BARS} hourly bars used)."
    )
    lines.append(
        "Corr ranking: Period A only, positive Pearson of hourly returns vs BTC; "
        "top_k in {5,8,10}."
    )
    lines.append(
        "Signal: if |BTC ret over prior lag hours| >= thresh -> "
        f"same-direction {HOLD_BARS}h trade in each selected alt "
        f"(risk={RISK_FRACTION:.0%} / stop-proxy=thresh, fee={FEE_RATE})."
    )
    lines.append(
        "Tune on A only (max A Sharpe). Pass bar: return>0 AND Sharpe>0 on BOTH B and C."
    )
    lines.append(
        f"Period A: {a0.isoformat()} -> {a1.isoformat()} | "
        f"B: {b0.isoformat()} -> {b1.isoformat()} | "
        f"C: {c0.isoformat()} -> {c1.isoformat()}"
    )
    lines.append("Period-A BTC correlation leaders (positive):")
    for i, (sym, c) in enumerate(ranked[:10], 1):
        lines.append(f"  #{i} {sym} corr_A={c:.4f}")
    lines.append("")
    lines.append(f"--- All combos ({len(rows)}) ---")
    for r in rows:
        verdict = "PASS" if r.pass_bar else "FAIL"
        lines.append(
            f"{verdict} | btc_lead | {r.label[:70]:70} | "
            f"A: {fmt_m(r.a)} | B: {fmt_m(r.b)} | C: {fmt_m(r.c)}"
        )
    lines.append("")
    lines.append("--- Best on Period A (tuning reference; NOT a pass criterion) ---")
    if best_a:
        lines.append(
            f"{'PASS' if best_a.pass_bar else 'FAIL'} | A-best | {best_a.label}"
        )
        lines.append(f"  A: {fmt_m(best_a.a)}")
        lines.append(f"  B: {fmt_m(best_a.b)}")
        lines.append(f"  C: {fmt_m(best_a.c)}")
        lines.append(
            f"  => A-best B+C outcome: {'PASS' if best_a.pass_bar else 'FAIL'}"
        )
    lines.append("")
    lines.append(
        f"Combos with return>0 AND Sharpe>0 on B AND C: {len(passed)}/{len(rows)}"
    )
    if passed:
        passed_sorted = sorted(
            passed,
            key=lambda r: (
                ((r.b.sharpe or 0) + (r.c.sharpe or 0)) / 2
            ),
            reverse=True,
        )
        for i, r in enumerate(passed_sorted[:5], 1):
            lines.append(f"  PASSER#{i}: {r.label}")
            lines.append(f"    A: {fmt_m(r.a)} | B: {fmt_m(r.b)} | C: {fmt_m(r.c)}")
    lines.append("")
    lines.append("BLUNT VERDICT:")
    if not passed:
        lines.append(
            "- BTC->alt lead-lag: FAIL. No combo clears return>0 AND Sharpe>0 "
            "on both B and C after A-only correlation ranking + A-only param tune."
        )
        lines.append(
            "- Final stance for this competition cycle: stop searching; keep the "
            "existing capital-preservation cascade+funding live design - no further "
            "strategy changes before July 20."
        )
    else:
        lines.append(
            f"- BTC->alt lead-lag: {len(passed)} combo(s) clear B+C. Review A-best "
            "and passers before any promotion (A-tuning discipline still applies)."
        )
        if best_a and not best_a.pass_bar:
            lines.append(
                "- Caution: A-best combo itself FAILS the B+C bar — any passer "
                "would not be selected by A-only tuning."
            )
    lines.append("=" * 72)

    text = "\n".join(lines) + "\n"
    print(text)
    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(text)
    print(f"Appended results to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
