"""
India Standard Time, safely.

On Windows the IANA timezone database is not bundled with Python, so
ZoneInfo("Asia/Kolkata") raises ZoneInfoNotFoundError unless the `tzdata`
package is installed. We try the real zone first and fall back to a fixed
UTC+5:30 offset so the assistant always starts.
"""
from __future__ import annotations

from datetime import timedelta, timezone

try:  # preferred: real IANA zone (handles any historical/DST edge cases)
    from zoneinfo import ZoneInfo

    IST = ZoneInfo("Asia/Kolkata")
except Exception:  # noqa: BLE001  (ZoneInfoNotFoundError on Windows without tzdata)
    IST = timezone(timedelta(hours=5, minutes=30), "IST")
