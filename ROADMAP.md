# RAZER RESEARCH - Status

## Done
- Project setup, credentials secured
- Connected to LTP test API (api.ltp-contest.com)
- Getting live market prices (BTC, working)
- Order placement and cancellation confirmed working (tested via LIMIT BUY far below market price, safely never filled, then fully cancelled - final orderState confirmed CANCELLED)
- Strategy research: tested 7 signal hypotheses (mean-reversion, momentum, cross-sectional relative strength, funding-extreme, long-horizon trend, cascade-reversal, cascade+funding combo) via rigorous walk-forward validation on 180-day/8-symbol data. None showed a robust standalone edge.
- Final strategy decision: cascade+funding opportunistic entry (rare, ~2 trades/180 days), 1% risk per trade, tp/sl attached, 900 USDT circuit breaker. Simulated result: +0.53% over 180 days, near-zero drawdown, capital-preservation focused given no provable directional edge exists.

## Not done yet
- Wire strategy.py into live order execution
- Build standalone circuit-breaker safety layer (independent equity check before any order)
- Design the mandatory AI Agent piece (regime commentary/position-size adjustment, not signal generation since no alpha was found)
- Real competition keys (still waiting on LTP)

## Next step
- Wire strategy into live execution using the already-proven order lifecycle, then build the AI Agent piece
