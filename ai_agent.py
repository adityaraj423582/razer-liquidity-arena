# AI Agent regime monitor (mandatory competition component).
# get_regime_assessment() is the ONLY public entry point - the backend behind it is
# swappable: AI_BACKEND=mock (default, rule-based) or AI_BACKEND=real (LTP AI API).

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import pstdev
from typing import Any

from anthropic import Anthropic, APIConnectionError, APIStatusError, APITimeoutError
from dotenv import load_dotenv

from backtesting.backtest_funding_signal import FundingAsOf, percentile
from strategy import VOL_AVG_PERIOD, Candle

load_dotenv(Path(__file__).resolve().parent / ".env")

VOL_WINDOW = 24
VOL_PERCENTILE_CUTOFF = 90.0
FUNDING_LOW_PCTL = 5.0
FUNDING_HIGH_PCTL = 95.0
FUNDING_LOOKBACK = timedelta(days=30)
MIN_VOL_WINDOWS = 20
MIN_FUNDING_SAMPLES = 20
DECISIONS_LOG_FILE = "ai_decisions.log"
AUDIT_DIR = Path(__file__).resolve().parent / "audit"

# Real LTP-provided AI API (Anthropic-compatible). Uses the SEPARATE AI key
# (LTP_AI_API_KEY) - never the trading credentials (LTP_ACCESS_KEY/LTP_SECRET_KEY).
AI_API_BASE_URL = "https://ai.ltp-contest.com"
AI_MODEL = "MiniMax-M3"
AI_TIMEOUT_S = 300.0  # per LTP's notes
AI_MAX_RETRIES = 3  # exponential backoff on 429/5xx, per LTP's instructions
AI_RETRY_BASE_DELAY_S = 2.0
KLINE_TAIL_BARS = 24  # bars sent verbatim to the model (plus summary stats)

_decision_logger: logging.Logger | None = None
_ai_client: Anthropic | None = None


def _active_backend() -> str:
    """'real' only when explicitly requested; anything else falls back to 'mock'."""
    return "real" if os.getenv("AI_BACKEND", "mock").strip().lower() == "real" else "mock"


def _get_decision_logger() -> logging.Logger:
    global _decision_logger
    if _decision_logger is None:
        log = logging.getLogger("ai_decisions")
        log.setLevel(logging.INFO)
        log.propagate = False
        handler = logging.FileHandler(DECISIONS_LOG_FILE, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
        _decision_logger = log
    return _decision_logger


def _rolling_vols(closes: list[float]) -> list[float]:
    returns = [
        (closes[i] - closes[i - 1]) / closes[i - 1]
        for i in range(1, len(closes))
        if closes[i - 1] != 0
    ]
    vols = []
    for i in range(VOL_WINDOW, len(returns) + 1):
        vols.append(pstdev(returns[i - VOL_WINDOW : i]))
    return vols


# MOCK_BACKEND - temporary rule-based stand-in for the real LTP AI API.
# To be replaced with the organizer-provided AI endpoint once credentials arrive.
# This is NOT a real AI call and must not be presented as one.
def _no_reason_warning() -> dict[str, Any]:
    return {"warning": False, "bad_pct_claims": []}


def _mock_backend(
    symbol: str,
    recent_klines: list[Candle],
    recent_funding: FundingAsOf,
) -> tuple[str, str, dict[str, Any]]:
    vols = _rolling_vols([c.close for c in recent_klines])
    if len(vols) < MIN_VOL_WINDOWS:
        return (
            "PAUSE",
            (
                f"insufficient history for regime assessment "
                f"({len(vols)} vol windows < {MIN_VOL_WINDOWS}) [MOCK_BACKEND]"
            ),
            _no_reason_warning(),
        )

    current_vol = vols[-1]
    vol_cutoff = percentile(vols, VOL_PERCENTILE_CUTOFF)
    if current_vol >= vol_cutoff:
        return (
            "PAUSE",
            (
                f"realized 24h vol {current_vol:.5f} >= {VOL_PERCENTILE_CUTOFF:.0f}th pctl "
                f"{vol_cutoff:.5f} of recent history - unusually volatile regime [MOCK_BACKEND]"
            ),
            _no_reason_warning(),
        )

    now = recent_klines[-1].ts
    rate = recent_funding.rate_at(now)
    hist = recent_funding.window_rates(now, FUNDING_LOOKBACK) if rate is not None else []
    if rate is None or len(hist) < MIN_FUNDING_SAMPLES:
        return (
            "PAUSE",
            (
                f"insufficient funding history ({len(hist)} samples < {MIN_FUNDING_SAMPLES}) "
                "[MOCK_BACKEND]"
            ),
            _no_reason_warning(),
        )
    low = percentile(hist, FUNDING_LOW_PCTL)
    high = percentile(hist, FUNDING_HIGH_PCTL)
    if rate <= low:
        return (
            "PAUSE",
            (
                f"funding {rate:.6f} at/below {FUNDING_LOW_PCTL:.0f}th pctl {low:.6f} - "
                "extreme crowded-short regime [MOCK_BACKEND]"
            ),
            _no_reason_warning(),
        )
    if rate >= high:
        return (
            "PAUSE",
            (
                f"funding {rate:.6f} at/above {FUNDING_HIGH_PCTL:.0f}th pctl {high:.6f} - "
                "extreme crowded-long regime [MOCK_BACKEND]"
            ),
            _no_reason_warning(),
        )

    return (
        "NORMAL",
        (
            f"vol {current_vol:.5f} below {VOL_PERCENTILE_CUTOFF:.0f}th pctl and funding "
            f"{rate:.6f} within [{low:.6f}, {high:.6f}] [MOCK_BACKEND]"
        ),
        _no_reason_warning(),
    )


def _get_ai_client() -> Anthropic:
    global _ai_client
    if _ai_client is None:
        api_key = os.getenv("LTP_AI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("LTP_AI_API_KEY missing from environment/.env")
        # max_retries=0: we own the backoff policy below (429/5xx only, max 3)
        _ai_client = Anthropic(
            base_url=AI_API_BASE_URL,
            api_key=api_key,
            timeout=AI_TIMEOUT_S,
            max_retries=0,
        )
    return _ai_client


def _compute_regime_facts(recent_klines: list[Candle]) -> dict[str, Any]:
    """
    Deterministic numeric facts for the real-AI prompt (never left for the model
    to estimate). Volume ratio reuses strategy.VOL_AVG_PERIOD window ending at
    the latest bar (same averaging window as check_entry_signal).
    """
    if len(recent_klines) < 2:
        raise ValueError("need at least 2 candles to compute regime facts")

    latest = recent_klines[-1]
    prev = recent_klines[-2]
    if prev.close == 0:
        raise ValueError("previous close is zero")
    pct_move = (latest.close - prev.close) / prev.close * 100.0

    idx = len(recent_klines) - 1
    if idx < VOL_AVG_PERIOD:
        raise ValueError(
            f"need >= {VOL_AVG_PERIOD} candles for volume ratio "
            f"(have {len(recent_klines)})"
        )
    # Same window shape as strategy.check_entry_signal for the bar at idx
    vol_window = [
        c.volume for c in recent_klines[idx - VOL_AVG_PERIOD + 1 : idx + 1]
    ]
    vol_avg = sum(vol_window) / VOL_AVG_PERIOD
    if vol_avg <= 0:
        raise ValueError("non-positive average volume in VOL_AVG_PERIOD window")
    volume_ratio = latest.volume / vol_avg

    vols = _rolling_vols([c.close for c in recent_klines])
    if len(vols) < 2:
        raise ValueError("insufficient vol history for p90")
    current_vol = vols[-1]
    p90_vol = percentile(vols, VOL_PERCENTILE_CUTOFF)
    if p90_vol <= 0:
        raise ValueError("non-positive p90 vol")
    vol_vs_pctl_ratio = current_vol / p90_vol

    return {
        "pct_move_last_candle": round(pct_move, 2),
        "pct_move_last_candle_str": f"{pct_move:.2f}",
        "volume_ratio": volume_ratio,
        "volume_ratio_str": f"{volume_ratio:.2f}",
        "vol_vs_pctl_ratio": vol_vs_pctl_ratio,
        "vol_vs_pctl_ratio_str": f"{vol_vs_pctl_ratio:.2f}",
        "current_vol": current_vol,
        "current_vol_str": f"{current_vol:.5f}",
        "p90_vol": p90_vol,
        "p90_vol_str": f"{p90_vol:.5f}",
        "vol_avg_period": VOL_AVG_PERIOD,
    }


def build_regime_prompt(
    symbol: str,
    recent_klines: list[Candle],
    recent_funding: FundingAsOf,
) -> str:
    """
    Build the exact user-message string that _real_ai_backend would send.
    Public so tests can inspect the prompt with zero API calls / zero quota use.
    """
    return _build_regime_prompt(symbol, recent_klines, recent_funding)


def _build_regime_prompt(
    symbol: str,
    recent_klines: list[Candle],
    recent_funding: FundingAsOf,
) -> str:
    facts = _compute_regime_facts(recent_klines)
    now = recent_klines[-1].ts
    rate = recent_funding.rate_at(now)
    funding_hist = (
        recent_funding.window_rates(now, FUNDING_LOOKBACK) if rate is not None else []
    )

    tail = recent_klines[-KLINE_TAIL_BARS:]
    kline_lines = "\n".join(
        f"{c.ts.isoformat()} o={c.open:g} h={c.high:g} l={c.low:g} "
        f"c={c.close:g} v={c.volume:g}"
        for c in tail
    )
    funding_tail = ", ".join(f"{r:.6f}" for r in funding_hist[-15:]) or "none"

    return (
        f"You are the risk-regime monitor for a conservative crypto perpetual-futures "
        f"trading agent (long-only, tight stops, capital preservation first).\n\n"
        f"Symbol: {symbol}\n"
        f"Last {len(tail)} hourly candles (of {len(recent_klines)} provided):\n"
        f"{kline_lines}\n\n"
        f"COMPUTED FACTS (deterministic — do not re-estimate these):\n"
        f"Last candle move: {facts['pct_move_last_candle_str']}% | "
        f"Volume ratio vs recent average: {facts['volume_ratio_str']}x | "
        f"Current vol is {facts['vol_vs_pctl_ratio_str']}x the 90th percentile threshold\n"
        f"(volume average uses prior {facts['vol_avg_period']}-bar window ending at latest bar, "
        f"same N as strategy.py VOL_AVG_PERIOD)\n"
        f"Realized 24h vol (stdev of hourly returns): current={facts['current_vol_str']}, "
        f"90th percentile of recent history={facts['p90_vol_str']}\n"
        f"Current funding rate: {rate if rate is not None else 'unknown'}\n"
        f"Recent funding prints (oldest->newest, last 15 of {len(funding_hist)}): "
        f"{funding_tail}\n\n"
        f"Task: classify the current market regime for this symbol.\n"
        f"- NORMAL: conditions are ordinary enough that the agent may consider new entries.\n"
        f"- PAUSE: elevated risk (volatility spike, disorderly moves, extreme/crowded "
        f"funding, or insufficient data) - the agent must skip new entries this cycle.\n"
        f"When uncertain, prefer PAUSE.\n\n"
        f"Reasoning rules:\n"
        f"- Base your reasoning ONLY on the numeric values provided above. Do not estimate, "
        f"round, or describe magnitudes in your own words (e.g. do not say '~3% drop' or "
        f"'elevated' — cite the exact provided numbers or ratios instead).\n"
        f"- If your stated reason contains any number, it must match a number given to you "
        f"in this prompt exactly.\n\n"
        f'Respond with ONLY a JSON object, no other text: '
        f'{{"decision": "NORMAL" or "PAUSE", "reason": "<one concise sentence>"}}'
    )


def _validate_reason_against_facts(
    symbol: str, reason: str, facts: dict[str, Any]
) -> dict[str, Any]:
    """
    Lightweight drift detector: warn (do not fail) if the model's reason text
    looks like it invents percentages/ratios instead of citing our computed facts.
    Returns {"warning": bool, "bad_pct_claims": list[str]} for persistence on
    the decision log line and audit jsonl record.
    """
    log = _get_decision_logger()
    reason_l = reason.lower()
    anchors = [
        facts["pct_move_last_candle_str"],
        facts["volume_ratio_str"],
        facts["vol_vs_pctl_ratio_str"],
        facts["current_vol_str"],
        facts["p90_vol_str"],
    ]
    cites_anchor = any(a in reason for a in anchors if a)

    pct_claims = re.findall(r"(-?\d+(?:\.\d+)?)\s*%", reason)
    bad_pcts: list[str] = []
    expected_pct = facts["pct_move_last_candle_str"]
    for claim in pct_claims:
        # Allow exact match or same value with optional trailing zeros stripped
        try:
            if abs(float(claim) - float(expected_pct)) > 1e-9:
                bad_pcts.append(claim + "%")
        except ValueError:
            bad_pcts.append(claim + "%")

    vague = any(
        phrase in reason_l
        for phrase in ("elevated", "approximately", "~", "about ", "roughly", "around ")
    )

    warn_info: dict[str, Any] = {"warning": False, "bad_pct_claims": []}
    if bad_pcts or (not cites_anchor and (pct_claims or vague)):
        warn_info = {"warning": True, "bad_pct_claims": list(bad_pcts)}
        log.warning(
            f"{symbol}: AI reason may misstate provided facts | "
            f"expected anchors pct_move={expected_pct}% vol_ratio={facts['volume_ratio_str']}x "
            f"vol_vs_p90={facts['vol_vs_pctl_ratio_str']}x | "
            f"bad_pct_claims={bad_pcts or 'none'} cites_anchor={cites_anchor} "
            f"vague_language={vague} | reason={reason!r}"
        )
    return warn_info


def _parse_ai_response(text: str) -> tuple[str, str]:
    """Extract {'decision','reason'} from the model reply. Raises on anything malformed."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object in AI response: {text[:200]!r}")
    payload = json.loads(cleaned[start : end + 1])
    decision = str(payload.get("decision", "")).strip().upper()
    if decision not in ("NORMAL", "PAUSE"):
        raise ValueError(f"invalid decision {payload.get('decision')!r}")
    reason = str(payload.get("reason", "")).strip() or "no reason given by model"
    return decision, reason


def _real_ai_backend(
    symbol: str,
    recent_klines: list[Candle],
    recent_funding: FundingAsOf,
) -> tuple[str, str, dict[str, Any]]:
    """
    Real LTP AI API call. Fail-safe: any failure (retries exhausted, timeout,
    parse error, unexpected exception) returns PAUSE - never fails open to NORMAL.
    Uses LTP_AI_API_KEY only (never trading LTP_ACCESS_KEY / LTP_SECRET_KEY).
    """
    log = _get_decision_logger()
    try:
        facts = _compute_regime_facts(recent_klines)
    except Exception as exc:
        log.warning(f"{symbol}: real AI could not compute regime facts - {exc}")
        return (
            "PAUSE",
            f"real AI backend unavailable, fail-safe PAUSE: fact compute failed: "
            f"{exc} [REAL_AI:{AI_MODEL}]",
            _no_reason_warning(),
        )
    prompt = _build_regime_prompt(symbol, recent_klines, recent_funding)

    last_error = "unknown error"
    for attempt in range(1, AI_MAX_RETRIES + 1):
        try:
            client = _get_ai_client()
            response = client.messages.create(
                model=AI_MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            decision, reason = _parse_ai_response(text)
            warn_info = _validate_reason_against_facts(symbol, reason, facts)
            return decision, f"{reason} [REAL_AI:{AI_MODEL}]", warn_info
        except APIStatusError as exc:
            last_error = f"HTTP {exc.status_code}: {str(exc)[:200]}"
            retryable = exc.status_code == 429 or exc.status_code >= 500
            if not retryable:
                log.warning(f"{symbol}: real AI non-retryable error - {last_error}")
                break
        except (APITimeoutError, APIConnectionError) as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
        except (ValueError, json.JSONDecodeError) as exc:
            # Parse failure: retrying won't help a malformed contract; fail safe now
            last_error = f"unparseable AI response: {str(exc)[:200]}"
            log.warning(f"{symbol}: real AI parse failure - {last_error}")
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            log.warning(f"{symbol}: real AI unexpected error - {last_error}")
            break

        if attempt < AI_MAX_RETRIES:
            delay = AI_RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
            log.warning(
                f"{symbol}: real AI attempt {attempt}/{AI_MAX_RETRIES} failed "
                f"({last_error}); retrying in {delay:.0f}s"
            )
            time.sleep(delay)

    log.warning(f"{symbol}: real AI backend failed - failing safe to PAUSE ({last_error})")
    return (
        "PAUSE",
        f"real AI backend unavailable, fail-safe PAUSE: {last_error} [REAL_AI:{AI_MODEL}]",
        _no_reason_warning(),
    )


def _append_audit_record(
    timestamp: str,
    symbol: str,
    decision: str,
    reason: str,
    backend: str,
    warn_info: dict[str, Any] | None = None,
) -> None:
    # Durable git-trackable trail; must never raise to the caller
    try:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        date_stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
        path = AUDIT_DIR / f"ai_decisions_{date_stamp}.jsonl"
        info = warn_info if warn_info is not None else _no_reason_warning()
        warning = bool(info.get("warning"))
        record: dict[str, Any] = {
            "timestamp": timestamp,
            "symbol": symbol,
            "decision": decision,
            "reason": reason,
            "backend": backend,
            "warning": warning,
        }
        if warning:
            record["bad_pct_claims"] = list(info.get("bad_pct_claims") or [])
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logging.getLogger("ai_decisions").warning(
            f"audit write failed (decision still returned): {exc}"
        )


def get_regime_assessment(
    symbol: str,
    recent_klines: list[Candle],
    recent_funding: FundingAsOf,
) -> dict[str, Any]:
    """
    Sole public entry point. Signature and return shape are stable:
      {"decision": "NORMAL"|"PAUSE", "reason": str, "timestamp": ISO-8601 UTC}
    Backend selected by AI_BACKEND env (default mock). Failures -> PAUSE.
    """
    timestamp = datetime.now(tz=timezone.utc).isoformat()
    backend = _active_backend()
    log = _get_decision_logger()
    warn_info = _no_reason_warning()
    try:
        if backend == "real":
            decision, reason, warn_info = _real_ai_backend(
                symbol, recent_klines, recent_funding
            )
        else:
            decision, reason, warn_info = _mock_backend(
                symbol, recent_klines, recent_funding
            )
    except Exception as exc:
        # Outer fail-safe: never fail open to NORMAL
        decision, reason = "PAUSE", f"assessment error: {exc} [{backend}]"
        warn_info = _no_reason_warning()
        log.warning(f"{symbol}: get_regime_assessment outer fail-safe PAUSE - {exc}")
    warning = bool(warn_info.get("warning"))
    if warning:
        claims = list(warn_info.get("bad_pct_claims") or [])
        log.info(
            f"{timestamp} | {symbol} | {decision} | backend={backend} | "
            f"warning=true | bad_pct_claims={claims} | {reason}"
        )
    else:
        log.info(
            f"{timestamp} | {symbol} | {decision} | backend={backend} | "
            f"warning=false | {reason}"
        )
    _append_audit_record(timestamp, symbol, decision, reason, backend, warn_info)
    return {"decision": decision, "reason": reason, "timestamp": timestamp}


def connectivity_test() -> int:
    """
    One minimal request to the real LTP AI API; prints the raw response.
    Run manually (python ai_agent.py) BEFORE enabling AI_BACKEND=real.
    Spends one tiny request of quota - only run when you decide to.
    """
    api_key = os.getenv("LTP_AI_API_KEY", "").strip()
    if not api_key:
        print("FAIL: LTP_AI_API_KEY not set in .env")
        return 1
    print(f"Endpoint: {AI_API_BASE_URL}")
    print(f"Model:    {AI_MODEL}")
    print(f"Key:      set ({len(api_key)} chars, not printed)")
    print("Sending one minimal message...")
    try:
        client = Anthropic(
            base_url=AI_API_BASE_URL,
            api_key=api_key,
            timeout=AI_TIMEOUT_S,
            max_retries=0,
        )
        response = client.messages.create(
            model=AI_MODEL,
            max_tokens=50,
            messages=[
                {"role": "user", "content": "Reply with exactly: CONNECTIVITY OK"}
            ],
        )
        print("\n--- raw response object ---")
        print(response.model_dump_json(indent=2))
        text = "".join(b.text for b in response.content if hasattr(b, "text"))
        print("\n--- extracted text ---")
        print(text)
        print("\nRESULT: request/response completed")
        return 0
    except Exception as exc:
        print(f"\nRESULT: FAILED - {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(connectivity_test())
