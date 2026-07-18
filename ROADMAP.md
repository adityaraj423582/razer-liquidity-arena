# RAZER RESEARCH - Status

## Done
- Project setup, credentials secured
- Connected to LTP test API (api.ltp-contest.com)
- Getting live market prices (BTC, working)
- Order placement and cancellation confirmed working (tested via LIMIT BUY far below market price, safely never filled, then fully cancelled - final orderState confirmed CANCELLED)
- Strategy research: tested 7 signal hypotheses (mean-reversion, momentum, cross-sectional relative strength, funding-extreme, long-horizon trend, cascade-reversal, cascade+funding combo) via rigorous walk-forward validation on 180-day/8-symbol data. None showed a robust standalone edge.
- Final strategy decision: cascade+funding opportunistic entry (rare, ~2 trades/180 days), 1% risk per trade, tp/sl attached, 900 USDT circuit breaker. Simulated result: +0.53% over 180 days, near-zero drawdown, capital-preservation focused given no provable directional edge exists.
- live_trading_loop.py built and verified: leverage confirmed set to 2x on BTC and ETH (competition cap), equity correctly parsed from confirmed BINANCE entry field, circuit breaker logic verified, one supervised manual iteration ran cleanly (no signal present, correctly stayed flat, no forced trade)
- AI Agent regime gate built (ai_agent.py): pluggable get_regime_assessment(); mock backend intact; real LTP AI backend coded behind AI_BACKEND flag; PAUSE/NORMAL decisions logged to ai_decisions.log and durable dated audit/ai_decisions_YYYYMMDD.jsonl with backend field
- Continuous unattended live loop with heartbeat: live_trading_loop.py runs hourly by default (matches 1h candles), per-iteration exception containment with full traceback, heartbeat.txt overwritten each cycle; RAZERDEMO only
- Project folder reorganization: backtesting/, multi_strategy/, testing/ packages; root keeps live runtime + strategy + ai_agent; imports and data/ paths updated; live_trading_loop.py --once sanity-checked after the move
- Fixed competition_universe.json to match officially published 50-symbol list (was previously a volume-ranked approximation, 24/50 symbols were incorrect). Note: earlier 50-symbol backtest sweep (multi_strategy_sweep_results.txt) was run against the OLD incorrect universe and should be considered indicative but not exact for the real competition set.

## Not done yet
- real AI API integrated, not yet tested (run python ai_agent.py or python testing/test_ai_api.py; only then set AI_BACKEND=real)
- Monitoring/alerting on heartbeat staleness (heartbeat.txt is written; nothing reads it yet)
- Real competition API keys (still waiting on LTP, expected before July 20)
- Final pre-launch checklist before July 20

## Next step
- Run the AI connectivity test manually, confirm response, then decide when to set AI_BACKEND=real for the live loop
