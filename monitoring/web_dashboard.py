"""Browser dashboard on localhost for a running instance.

    python main.py --web-dashboard            # http://127.0.0.1:8050

Like the terminal viewer, it only reads `monitoring.status_file` (written by
the trading process every bar), so it can be started or stopped at any time
without touching trading. Read-only, standard library only, and bound to
127.0.0.1 by default so it is not reachable from the network.

    GET /             the dashboard page (polls /api/status every 5 s)
    GET /api/status   latest status + equity samples collected while this server runs
"""
from __future__ import annotations

import json
import threading
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

from monitoring.dashboard import mark_stale, read_status
from monitoring.logger import get_logger

log = get_logger(__name__)

PAGE = Path(__file__).with_name("web") / "dashboard.html"
HISTORY_POINTS = 2000      # ~7 days of 5-minute bars


class StatusFeed:
    """Reads the status file and keeps an equity history for the chart."""

    def __init__(self, status_file: Path, refresh_seconds: float = 5.0):
        self.status_file = status_file
        self.refresh_seconds = refresh_seconds
        self.history: Deque[Tuple[str, float]] = deque(maxlen=HISTORY_POINTS)
        self._lock = threading.Lock()

    def snapshot(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        status = mark_stale(read_status(self.status_file), now=now)
        equity = (status.get("portfolio") or {}).get("equity")
        stamp = (status.get("updated") or "").split(" ")[0]
        with self._lock:
            if equity is not None and stamp and (not self.history or self.history[-1][0] != stamp):
                self.history.append((stamp, float(equity)))
            history = list(self.history)
        return {"status": status, "history": history, "refresh_seconds": self.refresh_seconds,
                "server_time": (now or datetime.now()).isoformat(timespec="seconds")}


def make_handler(feed: StatusFeed):
    page = PAGE.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "QuartelyDashboard/1.0"

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
            elif path == "/api/status":
                body = json.dumps(feed.snapshot(), default=str).encode()
                self._send(HTTPStatus.OK, body, "application/json")
            else:
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def log_message(self, fmt: str, *args) -> None:   # keep the console quiet
            log.debug("http " + fmt % args)

    return Handler


def make_server(status_file: Path, host: str = "127.0.0.1", port: int = 8050,
                refresh_seconds: float = 5.0) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(StatusFeed(status_file, refresh_seconds)))


def serve(status_file: Path, host: str = "127.0.0.1", port: int = 8050, refresh_seconds: float = 5.0) -> None:
    server = make_server(status_file, host, port, refresh_seconds)
    url = f"http://{host}:{server.server_address[1]}"
    if host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("web dashboard reachable from the network", extra={"host": host})
    log.info("web dashboard running", extra={"url": url, "status_file": str(status_file)})
    print(f"Dashboard: {url}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
