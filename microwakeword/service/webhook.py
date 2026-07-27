"""Webhook delivery with retries and optional HMAC signature."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any, Optional

import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


def _sign(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def deliver_webhook(
    url: str,
    payload: dict[str, Any],
    *,
    secret: Optional[str] = None,
    max_retries: int = 3,
    timeout_s: float = 15.0,
) -> bool:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "User-Agent": "microwakeword-service/1.0",
    }
    if secret:
        headers["X-MWW-Signature"] = _sign(body, secret)

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                status = getattr(resp, "status", 200)
                if 200 <= int(status) < 300:
                    logger.info("webhook delivered to %s (attempt %d)", url, attempt)
                    return True
                logger.warning(
                    "webhook %s returned status %s (attempt %d)", url, status, attempt
                )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning(
                "webhook delivery failed to %s (attempt %d): %s", url, attempt, exc
            )

        if attempt < max_retries:
            time.sleep(2 ** (attempt - 1))

    logger.error("webhook delivery exhausted retries for %s", url)
    return False
