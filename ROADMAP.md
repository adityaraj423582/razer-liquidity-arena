# RAZER RESEARCH - Liquidity Arena 2026 Progress

## Competition Key Facts
- Track A - Logic Frontier, Phase 1: July 20 - Aug 21, 2026
- Sandbox API host: https://api.ltp-contest.com (NOT the documented production api.liquiditytech.com)
- Signing quirk: no-params requests need "&" + nonce as the message, not just the bare nonce
- Disqualification rule: equity < 800 USDT (20% drawdown from 1000 USDT start) → auto stop-loss + disqualified
- Uptime/heartbeat rule: REMOVED (confirmed by LTP staff, not just the bot) - no longer a disqualification factor
- AI Agent use is MANDATORY, $10/day AI Token budget via Organizer-provided API only
- WebSocket market data: wss://mds.ltp-contest.com/marketdata/v2/public; subscribe via `{"event":"subscribe","arg":[{"channel":"...","sym":"..."}]}`
- Test portfolio RAZERDEMO funded with 1,000 USDT mock

## ✅ Completed
- 2026-07-15: Repo scaffold — `.gitignore` (includes `.env`), `.env.example`, `requirements.txt`
- 2026-07-15: Credentials secured in local `.env` (never committed); `.env.example` restored to placeholders
- 2026-07-16: Step 1 REST connectivity — `test_connection.py` HMAC-SHA256 signing confirmed against `GET /api/v1/trading/account` → HTTP 200, `code=200000`, `message=Success` on `https://api.ltp-contest.com`
- 2026-07-16: Signing bugfix — no-params message must be `"&" + nonce` (bare nonce caused `API verification failed`)
- 2026-07-16: Step 2 market-data script created — `test_marketdata.py` + `websockets` dependency; public WS connects (no credentials); gzip binary frames decoded
- 2026-07-16: Market-data subscribe confirmed — real format is `{"event":"subscribe","arg":[{"channel":"...","sym":"..."}]}` (not `op`/`args`). Live BBO + TICKER for `BINANCE_PERP_BTC_USDT`. BBO fields: `bid`, `ask`, `bidqty`, `askqty`, `ts`. TICKER fields: `last`, `chg`, `high24h`, `low24h`, `open24h`, `vol24h`, `vol24v`, `ts`

## 🔄 In Progress / Just Finished
- (none)

## ⏳ Waiting On
- Real Phase 1 competition API keys / portfolio from LTP (sandbox RAZERDEMO is what we have now)
- Organizer-provided AI Token API access details ($10/day budget)

## 📋 Next Steps
- Build read-only order-book / position endpoint checks next
- Move toward Place Order (preview if applicable) only after reviewing that spec closely
- Define risk rails early (hard stop before 800 USDT equity DQ line)
- Design minimal trading loop only after market data + account reads are solid

## ⚠️ Open Questions / Risks
- Does the uptime/heartbeat rule removal apply specifically to Phase 1, or only later phases? (staff said removed; still worth pinning to Phase 1 written rules)
- Confirm whether sandbox equity/DQ rules mirror live Phase 1 exactly
- AI Token API endpoint, auth, and metering still unspecified in-repo
