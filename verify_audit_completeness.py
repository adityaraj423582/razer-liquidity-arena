"""
One-off / periodic check: every decision line in ai_decisions.log should have a
matching audit/*.jsonl record with required fields.

Usage:
  python verify_audit_completeness.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "ai_decisions.log"
AUDIT_DIR = ROOT / "audit"

REQUIRED_FIELDS = ("timestamp", "symbol", "decision", "reason", "backend", "warning")

# Example log line:
# 2026-07-17T14:37:49.738622+00:00 | BTCUSDT | NORMAL | backend=real | warning=false | ...
LOG_RE = re.compile(
    r"^(?P<timestamp>\S+)\s*\|\s*(?P<symbol>[A-Z0-9]+)\s*\|\s*"
    r"(?P<decision>NORMAL|PAUSE)\s*\|\s*backend=(?P<backend>\w+)\s*\|"
)


def load_audit_index() -> dict[tuple[str, str, str], list[dict]]:
    """Index audit records by (timestamp, symbol, decision)."""
    index: dict[tuple[str, str, str], list[dict]] = {}
    if not AUDIT_DIR.is_dir():
        return index
    for path in sorted(AUDIT_DIR.glob("ai_decisions_*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARN {path.name}:{line_no}: invalid JSON ({exc})")
                continue
            if not isinstance(rec, dict):
                continue
            key = (
                str(rec.get("timestamp", "")),
                str(rec.get("symbol", "")),
                str(rec.get("decision", "")).upper(),
            )
            index.setdefault(key, []).append(rec)
    return index


def main() -> int:
    if not LOG_PATH.exists():
        print(f"FAIL: missing {LOG_PATH}")
        return 1

    audit = load_audit_index()
    print(f"Loaded {sum(len(v) for v in audit.values())} audit records from {AUDIT_DIR}")

    decisions = 0
    gaps = 0
    field_gaps = 0
    for line_no, line in enumerate(LOG_PATH.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or "backend=" not in line:
            continue
        # Skip free-text WARNING lines that are not decision records
        if line.startswith("WARNING") or "AI reason may misstate" in line:
            continue
        m = LOG_RE.match(line)
        if not m:
            continue
        decisions += 1
        ts = m.group("timestamp")
        symbol = m.group("symbol")
        decision = m.group("decision")
        key = (ts, symbol, decision)
        matches = audit.get(key, [])
        if not matches:
            gaps += 1
            print(
                f"GAP log:{line_no}: no audit match for "
                f"ts={ts} symbol={symbol} decision={decision}"
            )
            continue
        rec = matches[0]
        missing = [f for f in REQUIRED_FIELDS if f not in rec]
        if missing:
            field_gaps += 1
            print(
                f"FIELD GAP log:{line_no} audit ts={ts} {symbol}: "
                f"missing fields {missing}"
            )

    print("---")
    print(f"Decision lines scanned: {decisions}")
    print(f"Missing audit records:  {gaps}")
    print(f"Incomplete field sets:  {field_gaps}")
    if gaps == 0 and field_gaps == 0:
        print("OK: audit trail complete for all logged decisions.")
        return 0
    print("FAIL: gaps found (see above).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
