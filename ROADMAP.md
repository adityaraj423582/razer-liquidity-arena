# RAZER RESEARCH - Status

## Done
- Project setup, credentials secured
- Connected to LTP test API (api.ltp-contest.com)
- Getting live market prices (BTC, working)
- Order placement and cancellation confirmed working (tested via LIMIT BUY far below market price, safely never filled, then fully cancelled - final orderState confirmed CANCELLED)
- Strategy research: tested 7 signal hypotheses (mean-reversion, momentum, cross-sectional relative strength, funding-extreme, long-horizon trend, cascade-reversal, cascade+funding combo) via rigorous walk-forward validation on 180-day/8-symbol data. None showed a robust standalone edge.
- Final strategy decision: cascade+funding opportunistic entry (rare, ~2 trades/180 days), 1% risk per trade, tp/sl attached, 900 USDT circuit breaker. Simulated result: +0.53% over 180 days, near-zero drawdown, capital-preservation focused given no provable directional edge exists.
- live_trading_loop.py built and verified: leverage confirmed set to 2x on BTC and ETH (competition cap), equity correctly parsed from confirmed BINANCE entry field, circuit breaker logic verified, one supervised manual iteration ran cleanly (no signal present, correctly stayed flat, no forced trade)

## Not done yet
- AI Agent piece (mandatory per competition rules) - regime commentary/monitoring, not signal generation
- Decide on continuous unattended running (--loop mode) - when and how to supervise it
- Real competition API keys (still waiting on LTP, expected before July 20)
- Final pre-launch checklist before July 20

## Next step
- Build the AI Agent piece, then work through the pre-launch checklist before July 20
