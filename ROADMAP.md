# RAZER RESEARCH - Status

## Done
- Project setup, credentials secured
- Connected to LTP test API (api.ltp-contest.com)
- Getting live market prices (BTC, working)
- Order placement and cancellation confirmed working (tested via LIMIT BUY far below market price, safely never filled, then fully cancelled - final orderState confirmed CANCELLED)

## Not done yet
- Trading strategy logic
- Risk/safety limits (stop trading if losing too much)
- Real competition keys (still waiting on LTP)

## Next step
- Design the trading strategy and safety limits, then build the AI Agent piece required by competition rules
