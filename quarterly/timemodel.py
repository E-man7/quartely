"""NY session clock (America/New_York, DST-aware via zoneinfo).

Session: 6 hours from `session_start_hour`, four 90-minute quarters.
    Q2 start      = start + 90 min      (NY Session True Open)
    Q3 start      = start + 180 min
    entry window  = Q3 start + 45 min, for 22.5 min   (10:45:00-11:07:30 at 07:00)
    session end   = start + 360 min
The trading day for date D runs from 18:00 NY on D-1.

Bars are timestamped at their open (Alpaca convention) and act at their close.
A bar is "in" a period when its open time is inside it; e.g. the entry window
holds the bars 10:45 ... 11:07 and the last signal can fire at 11:08.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")
DAY_START_HOUR = 18


def _at(d: date, hour: int, minute: int = 0, second: int = 0, micro: int = 0) -> pd.Timestamp:
    """Wall-clock NY time on date d; zoneinfo picks the right UTC offset (EST/EDT)."""
    return pd.Timestamp(datetime(d.year, d.month, d.day, hour, minute, second, micro, tzinfo=NY))


def to_ny(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC").tz_convert(NY) if ts.tzinfo is None else ts.tz_convert(NY)


def trading_day(ts) -> date:
    """The trading day a timestamp belongs to: from 18:00 NY it is already the next day."""
    t = to_ny(ts)
    return (t + timedelta(days=1)).date() if t.hour >= DAY_START_HOUR else t.date()


@dataclass(frozen=True)
class SessionTimes:
    day: date
    day_start: pd.Timestamp        # 18:00 NY on the previous calendar day
    midnight: pd.Timestamp         # 00:00 NY (True Day Open)
    session_start: pd.Timestamp
    q2_start: pd.Timestamp         # NY Session True Open
    q3_start: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp       # exclusive
    session_end: pd.Timestamp

    def in_window(self, bar_open) -> bool:
        return self.window_start <= to_ny(bar_open) < self.window_end

    def in_session(self, bar_open) -> bool:
        return self.session_start <= to_ny(bar_open) < self.session_end


def session_times(day: date, start_hour: int = 7) -> SessionTimes:
    if start_hour not in (6, 7):
        raise ValueError("session start hour must be 6 or 7")
    start = _at(day, start_hour)
    # Quarters are fixed wall-clock offsets inside one morning (no DST change happens 06:00-13:00 NY)
    q = timedelta(minutes=90)
    q3 = start + 2 * q
    window_start = q3 + timedelta(minutes=45)
    return SessionTimes(
        day=day,
        day_start=_at(day - timedelta(days=1), DAY_START_HOUR),
        midnight=_at(day, 0),
        session_start=start,
        q2_start=start + q,
        q3_start=q3,
        window_start=window_start,
        window_end=window_start + timedelta(minutes=22, seconds=30),
        session_end=start + 4 * q,
    )


def clock(day: date, hhmm: "tuple[int, int]") -> pd.Timestamp:
    return _at(day, hhmm[0], hhmm[1])


def is_overnight(ts, start: "tuple[int, int]" = (20, 0), end: "tuple[int, int]" = (4, 0)) -> bool:
    """True for bars in the overnight session (default 20:00-04:00 NY)."""
    t = to_ny(ts).time()
    return t >= time(*start) or t < time(*end)
