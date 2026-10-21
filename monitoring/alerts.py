"""Alerts: console, alerts.log, email (optional) and webhook (optional). Failures here never raise.

Each event type is sent at most once per `rate_limit_minutes` (default 15);
suppressed repeats are counted and reported with the next alert of that type.
"""
from __future__ import annotations

import json
import smtplib
import sys
import threading
import time
import urllib.request
from email.message import EmailMessage
from typing import Callable, Dict, Optional

from monitoring.logger import get_logger

log = get_logger(__name__)

# Trigger names used across the system
REGIME_CHANGE = "regime_change"
CIRCUIT_BREAKER = "circuit_breaker"
LARGE_PNL = "large_pnl"
DATA_FEED_DOWN = "data_feed_down"
API_LOST = "api_lost"
HMM_RETRAINED = "hmm_retrained"
FLICKER_EXCEEDED = "flicker_exceeded"
ORDER_REJECTED = "order_rejected"
ERROR = "error"


class AlertManager:
    def __init__(self, settings: Dict, credentials: Optional[Dict] = None, clock: Callable[[], float] = time.time,
                 console: bool = True):
        cfg = settings.get("monitoring", {}).get("alerts", {})
        self.email_enabled = cfg.get("email_enabled", False)
        self.webhook_enabled = cfg.get("webhook_enabled", False)
        self.events = set(cfg.get("on", []))
        self.rate_limit = cfg.get("rate_limit_minutes", 15) * 60
        self.console = console and cfg.get("console", True)
        creds = credentials or {}
        self.email_cfg = creds.get("email", {})
        self.webhook_url = creds.get("webhook", {}).get("url")
        self.clock = clock
        self._last_sent: Dict[str, float] = {}
        self._suppressed: Dict[str, int] = {}
        self._lock = threading.Lock()
        self.sent: list = []           # (event, message) this session, for the summary

    def send(self, event: str, message: str, details: Optional[Dict] = None, key: Optional[str] = None) -> bool:
        """Deliver an alert unless one with the same key (default: the event type) was sent within
        the rate-limit window. Returns True if sent. Use `key` to rate-limit sub-types separately,
        e.g. a breaker halt must not be swallowed by the size-reduction alert before it."""
        if self.events and event not in self.events:
            log.info("alert (not enabled)", extra={"event": event, "alert_msg": message})
            return False
        key = key or event
        now = self.clock()
        with self._lock:
            last = self._last_sent.get(key)
            if last is not None and now - last < self.rate_limit:
                self._suppressed[key] = self._suppressed.get(key, 0) + 1
                log.info("alert rate-limited", extra={"event": event, "alert_msg": message})
                return False
            self._last_sent[key] = now
            suppressed = self._suppressed.pop(key, 0)
        if suppressed:
            message += f" (+{suppressed} similar in the last {self.rate_limit / 60:.0f} min)"
        self.sent.append((event, message))
        log.warning("alert", extra={"event": event, "alert_msg": message, **(details or {})})
        if self.console:
            print(f"🚨 ALERT [{event}] {message}", file=sys.stderr, flush=True)
        subject = f"[Quartely-trader] {event}: {message}"
        body = message + ("\n\n" + json.dumps(details, indent=2, default=str) if details else "")
        if self.email_enabled:
            self._email(subject, body)
        if self.webhook_enabled:
            self._webhook(subject, details or {})
        return True

    def _email(self, subject: str, body: str) -> None:
        cfg = self.email_cfg
        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = cfg["from_addr"]
            msg["To"] = ", ".join(cfg["to_addrs"])
            msg.set_content(body)
            with smtplib.SMTP(cfg["smtp_host"], cfg.get("smtp_port", 587), timeout=15) as smtp:
                smtp.starttls()
                smtp.login(cfg["username"], cfg["password"])
                smtp.send_message(msg)
        except Exception as exc:
            log.error("email alert failed", extra={"error": str(exc)})

    def _webhook(self, text: str, details: Dict) -> None:
        if not self.webhook_url:
            log.error("webhook alert enabled but no url configured")
            return
        # "text" is understood by Slack; "content" by Discord.
        payload = json.dumps({"text": text, "content": text, "details": details}, default=str).encode()
        req = urllib.request.Request(
            self.webhook_url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(req, timeout=10).close()
        except Exception as exc:
            log.error("webhook alert failed", extra={"error": str(exc)})
