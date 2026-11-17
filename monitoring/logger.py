"""Structured logging: JSON lines to rotating files, human-readable to the console.

Files (in `monitoring.log_dir`), each rotated at 10 MB with backups kept 30 days:
    main.log    everything
    trades.log  orders, fills, positions (order executor, position tracker)
    alerts.log  alerts (monitoring.alerts)
    regime.log  regime detection and strategy mapping
    quarterly.log  the Quarterly Theory strategy (decisions, orders)

Every JSON entry carries the shared trading context (regime, probability,
equity, positions, daily_pnl) set by the main loop with `set_context(...)`.
Pass per-message fields with `extra={...}`.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

_RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}
_configured = False

CONTEXT_FIELDS = ("regime", "probability", "equity", "positions", "daily_pnl")
_context: Dict[str, Any] = {k: None for k in CONTEXT_FIELDS}
_context_lock = threading.Lock()

CHANNELS: Dict[str, Sequence[str]] = {
    "trades.log": ("broker.order_executor", "broker.position_tracker"),
    "alerts.log": ("monitoring.alerts",),
    "regime.log": ("core.hmm_engine", "core.regime_strategies"),
    "quarterly.log": ("quarterly",),
}


def set_context(**fields: Any) -> None:
    """Update the trading context attached to every log entry."""
    with _context_lock:
        _context.update(fields)


def get_context() -> Dict[str, Any]:
    with _context_lock:
        return dict(_context)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            **get_context(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        fields = {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        if fields:
            base += "  " + " ".join(f"{k}={v}" for k, v in fields.items())
        return base


class PrefixFilter(logging.Filter):
    def __init__(self, prefixes: Sequence[str]):
        super().__init__()
        self.prefixes = tuple(prefixes)

    def filter(self, record: logging.LogRecord) -> bool:
        return any(record.name == p or record.name.startswith(p + ".") for p in self.prefixes)


class SizeAndAgeRotatingHandler(RotatingFileHandler):
    """Rotate at `max_bytes`; delete rotated files older than `max_age_days`."""

    def __init__(self, path: Path, max_bytes: int = 10_000_000, max_age_days: int = 30):
        super().__init__(path, maxBytes=max_bytes, backupCount=10_000, encoding="utf-8")
        self.max_age = max_age_days * 86400

    def rotation_filename(self, default_name: str) -> str:
        # main.log.20261021-153000 instead of main.log.1: names never shift
        return f"{self.baseFilename}.{datetime.now():%Y%m%d-%H%M%S-%f}"

    def doRollover(self) -> None:
        if self.stream:
            self.stream.close()
            self.stream = None
        Path(self.baseFilename).rename(self.rotation_filename(self.baseFilename))
        self.prune()
        if not self.delay:
            self.stream = self._open()

    def prune(self, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        base = Path(self.baseFilename)
        for f in base.parent.glob(base.name + ".*"):
            if now - f.stat().st_mtime > self.max_age:
                f.unlink(missing_ok=True)


def setup_logging(cfg: Optional[Dict] = None, project_root: Optional[Path] = None) -> None:
    global _configured
    cfg = cfg or {}
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    root.setLevel(cfg.get("log_level", "INFO"))

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(ConsoleFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    root.addHandler(console)

    if project_root is not None:
        log_dir = project_root / cfg.get("log_dir", "logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        max_bytes = int(cfg.get("log_max_bytes", 10_000_000))
        max_age = int(cfg.get("log_retention_days", 30))
        fmt = JsonFormatter() if cfg.get("json_logs", True) else console.formatter
        for name, prefixes in [("main.log", None), *CHANNELS.items()]:
            handler = SizeAndAgeRotatingHandler(log_dir / name, max_bytes, max_age)
            handler.setFormatter(fmt)
            if prefixes:
                handler.addFilter(PrefixFilter(prefixes))
            handler.prune()
            root.addHandler(handler)

    for noisy in ("urllib3", "hmmlearn", "websockets", "alpaca"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    if not _configured and not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)
