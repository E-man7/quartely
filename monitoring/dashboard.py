"""Terminal dashboard (rich) for a running instance.

The trading process writes its status to `monitoring.status_file` on every bar;
`python main.py --dashboard` re-reads that file every
`dashboard_refresh_seconds` and renders it, so the viewer never touches the
trading loop and can be opened or closed at any time.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

BAR_WIDTH = 20
STALE_AFTER_SECONDS = 15 * 60       # no status update for this long: the instance is probably down


def _money(x: Optional[float], signed: bool = False) -> str:
    if x is None:
        return "—"
    sign = "-" if x < 0 else "+" if signed else ""
    return f"{sign}${abs(x):,.0f}" if abs(x) >= 1000 else f"{sign}${abs(x):,.2f}"


def _pct(x: Optional[float], signed: bool = False, digits: int = 2) -> str:
    return "—" if x is None else f"{x:+.{digits}%}" if signed else f"{x:.{digits}%}"


def _ok(flag: Optional[bool]) -> str:
    return "❔" if flag is None else "✅" if flag else "❌"


def risk_bar(used: Optional[float], limit: float) -> Text:
    """Colour-coded fill of `used` (a positive drawdown) toward `limit`."""
    if used is None or not limit:
        return Text("—")
    frac = max(0.0, min(used / limit, 1.0))
    style = "green" if frac < 0.5 else "yellow" if frac < 0.8 else "red"
    filled = int(round(frac * BAR_WIDTH))
    icon = "✅" if frac < 0.5 else "⚠️" if frac < 1.0 else "❌"
    return Text.assemble(("█" * filled, style), ("░" * (BAR_WIDTH - filled), "dim"), f" {used:.1%}/{limit:.0%} {icon}")


def _section(title: str, body) -> Panel:
    return Panel(body, title=f"[bold]{title}", title_align="left", border_style="cyan", padding=(0, 1))


def render(status: Dict[str, Any]) -> Group:
    if not status:
        return Group(_section("SYSTEM", Text("Waiting for a running instance…", style="yellow")))
    r, p, risk, sysi = (status.get(k) or {} for k in ("regime", "portfolio", "risk", "system"))

    dash = lambda v: "—" if v is None else v  # noqa: E731
    regime = Text.assemble(
        (f"{r.get('label') or 'no signal yet'} ({_pct(r.get('probability'), digits=0)})", "bold"),
        f" | Stability: {dash(r.get('stability_bars'))} bars",
        f" | Flicker: {dash(r.get('flicker_rate'))}/{dash(r.get('flicker_window'))}",
        (f" | {r['strategy']}" if r.get("strategy") else "", "dim"),
    )
    if r.get("flickering"):
        regime.append("  UNCERTAIN", style="bold yellow")
    elif r.get("confirmed") is False:
        regime.append("  (pending confirmation)", style="yellow")

    pnl = p.get("daily_pnl")
    portfolio = Text.assemble(
        f"Equity: {_money(p.get('equity'))} | Daily: ",
        (f"{_money(pnl, signed=True)} ({_pct(p.get('daily_pnl_pct'), signed=True)})",
         "green" if (pnl or 0) >= 0 else "red"),
        f"\nAllocation: {_pct(p.get('allocation'), digits=0)} | Leverage: {p.get('leverage') or 0:.2f}x",
    )

    pos = Table.grid(padding=(0, 1))
    for row in status.get("positions") or []:
        pnl_pct = row.get("pnl_pct")
        pos.add_row(row["symbol"], row.get("side", "LONG"), _money(row.get("price")),
                    Text(_pct(pnl_pct, signed=True, digits=1), style="green" if (pnl_pct or 0) >= 0 else "red"),
                    f"Stop: {_money(row['stop']) if row.get('stop') else 'none'}", row.get("holding") or "")
    if not status.get("positions"):
        pos.add_row(Text("flat", style="dim"))

    sigs = Table.grid(padding=(0, 1))
    for s in (status.get("signals") or [])[:5]:
        sigs.add_row(s.get("time", ""), s.get("symbol", ""), s.get("action", ""), Text(s.get("reason", ""), style="dim"))
    if not status.get("signals"):
        sigs.add_row(Text("none this session", style="dim"))

    rk = Table.grid(padding=(0, 1))
    rk.add_row("Daily DD", risk_bar(risk.get("daily_dd"), risk.get("daily_limit", 0.03)))
    rk.add_row("Weekly DD", risk_bar(risk.get("weekly_dd"), risk.get("weekly_limit", 0.07)))
    rk.add_row("From peak", risk_bar(risk.get("peak_dd"), risk.get("peak_limit", 0.10)))
    breaker = risk.get("breaker", "normal")
    rk.add_row("Breaker", Text(breaker.upper(), style="green" if breaker == "normal" else "bold red"))

    latency, hmm_age, mode = sysi.get("api_latency_ms"), sysi.get("hmm_age_days"), sysi.get("mode", "—")
    system = Text.assemble(
        f"Data: {_ok(sysi.get('data_ok'))} | API: {_ok(sysi.get('api_ok'))}",
        f" {latency:.0f}ms" if latency is not None else "",
        f" | HMM: {'—' if hmm_age is None else f'{hmm_age:.0f}d ago'} | ",
        (mode, "bold red" if mode == "LIVE" else "bold green"),
        (f"\nUpdated {status.get('updated', '—')}", "dim"),
    )
    return Group(_section("REGIME", regime), _section("PORTFOLIO", portfolio), _section("POSITIONS", pos),
                 _section("RECENT SIGNALS", sigs), _section("RISK STATUS", rk), _section("SYSTEM", system))


def read_status(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_status(path: Path, status: Dict[str, Any]) -> None:
    """Atomic write so the viewer never reads a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, default=str))
    tmp.replace(path)


def mark_stale(status: Dict[str, Any], now: Optional[datetime] = None) -> Dict[str, Any]:
    """Flag the instance as down when its status file stopped updating."""
    updated = status.get("updated")
    if updated:
        age = ((now or datetime.now()) - datetime.fromisoformat(updated)).total_seconds()
        if age > STALE_AFTER_SECONDS:
            status.setdefault("system", {}).update(data_ok=False, api_ok=None)
            status["updated"] = f"{updated} (stale: {age / 60:.0f} min ago, instance not running?)"
    return status


def run_viewer(status_file: Path, refresh_seconds: float = 5.0, console: Optional[Console] = None) -> None:
    """Attach to a running instance: re-render its status file until Ctrl-C."""
    console = console or Console()
    with Live(render(read_status(status_file)), console=console, refresh_per_second=4, screen=False) as live:
        try:
            while True:
                live.update(render(mark_stale(read_status(status_file))))
                time.sleep(refresh_seconds)
        except KeyboardInterrupt:
            pass
