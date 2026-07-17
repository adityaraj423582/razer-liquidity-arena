"""
Connectivity smoke test for LTP RapidX API (Liquidity Arena 2026 Track A).

Verifies HMAC-SHA256 request signing against GET /api/v1/trading/account.
Does not place orders or touch risk logic.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hashlib
import hmac
import os
import sys
import time
from typing import Mapping
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv


def sign_request(secret_key: str, params: Mapping[str, str], nonce: str) -> str:
    """
    Build the LTP RapidX HMAC-SHA256 signature.

    Algorithm:
      1. Sort request params alphabetically by key
      2. Join as key1=val1&key2=val2 (no URL-encoding of the signing string)
      3. Append "&" + nonce -> message
      4. signature = HMAC-SHA256(secret, message).hexdigest()

    For a request with no params, the message is "&" + nonce
    (empty param string + "&" + nonce).
    """
    sorted_items = sorted(params.items(), key=lambda item: item[0])
    param_string = "&".join(f"{key}={value}" for key, value in sorted_items)
    message = f"{param_string}&{nonce}"
    return hmac.new(
        secret_key.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def main() -> int:
    load_dotenv(_ROOT / ".env")

    access_key = os.getenv("LTP_ACCESS_KEY", "").strip()
    secret_key = os.getenv("LTP_SECRET_KEY", "").strip()
    api_host = os.getenv("LTP_API_HOST", "").strip().rstrip("/")

    missing = [
        name
        for name, value in (
            ("LTP_ACCESS_KEY", access_key),
            ("LTP_SECRET_KEY", secret_key),
            ("LTP_API_HOST", api_host),
        )
        if not value
    ]
    if missing:
        print(f"Missing required .env values: {', '.join(missing)}")
        print("Copy .env.example to .env and fill in your credentials.")
        return 1

    nonce = str(int(time.time()))
    params: dict[str, str] = {}
    signature = sign_request(secret_key, params, nonce)

    url = urljoin(f"{api_host}/", "api/v1/trading/account")
    headers = {
        "Content-Type": "application/json",
        "nonce": nonce,
        "signature": signature,
        "X-MBX-APIKEY": access_key,
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
    except requests.exceptions.ConnectionError as exc:
        print(f"Connection error: unable to reach {api_host}")
        print(f"Details: {exc}")
        return 1
    except requests.exceptions.Timeout as exc:
        print(f"Connection error: request timed out contacting {api_host}")
        print(f"Details: {exc}")
        return 1
    except requests.exceptions.RequestException as exc:
        print(f"Request error: {exc}")
        return 1

    if response.status_code in (401, 403):
        print(f"Auth error: HTTP {response.status_code}")
        print("Check LTP_ACCESS_KEY, LTP_SECRET_KEY, and that the signature algorithm matches.")
        return 1

    if response.status_code != 200:
        print(f"Unexpected status code: HTTP {response.status_code}")
        # Do not print response body (may contain account details).
        return 1

    try:
        payload = response.json()
    except ValueError:
        print("Unexpected response: HTTP 200 but body is not valid JSON")
        return 1

    code = payload.get("code") if isinstance(payload, dict) else None
    message = payload.get("message") if isinstance(payload, dict) else None
    print(f"HTTP status: {response.status_code}")
    print(f"code: {code}")
    print(f"message: {message}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
