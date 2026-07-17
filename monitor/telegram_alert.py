"""Reusable Telegram Bot API helper for external monitors."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

logger = logging.getLogger("telegram_alert")

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
REQUEST_TIMEOUT_S = 30


def send_telegram_alert(message: str) -> bool:
    """
    Post ``message`` to Telegram via Bot API sendMessage.
    Returns True on success, False on any failure (never raises).
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        logger.warning(
            "Telegram alert skipped: TELEGRAM_BOT_TOKEN and/or TELEGRAM_CHAT_ID "
            "missing from environment/.env"
        )
        return False

    url = TELEGRAM_API_URL.format(token=token)
    try:
        response = requests.post(
            url,
            json={"chat_id": chat_id, "text": message},
            timeout=REQUEST_TIMEOUT_S,
        )
        if response.status_code != 200:
            logger.warning(
                "Telegram sendMessage failed: HTTP %s body=%s",
                response.status_code,
                response.text[:300],
            )
            return False
        payload = response.json()
        if not payload.get("ok"):
            logger.warning("Telegram sendMessage rejected: %s", str(payload)[:300])
            return False
        return True
    except requests.RequestException as exc:
        logger.warning("Telegram sendMessage request error: %s", exc)
        return False
    except Exception as exc:
        logger.warning("Telegram sendMessage unexpected error: %s", exc)
        return False
