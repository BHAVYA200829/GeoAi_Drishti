"""
Deterministic time-expression parsing.

Turning "tomorrow evening" or "Saturday morning" into concrete forecast hours
is a deterministic job -- it must NOT be left to the small model. The SLM only
extracts the raw phrase; this module resolves it against the live hourly series.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from .tz import IST

WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6, "mon": 0, "tue": 1,
            "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}

PARTS = {
    "morning": (5, 11), "afternoon": (12, 16), "evening": (16, 20),
    "subah": (5, 11), "dopahar": (12, 16), "shaam": (16, 20), "sham": (16, 20),
    "night": (20, 23), "tonight": (20, 23), "dawn": (5, 7), "dusk": (17, 19),
}


def _parse_hour_clock(text: str) -> int | None:
    m = re.search(r"\b(?:at|around|by|from|after|before|~)\s+(\d{1,2})\s*(?::(\d{2}))?\s*(am|pm)?\b", text)
    if m and m.group(3):
        h = int(m.group(1)) % 12
        if m.group(3) == "pm":
            h += 12
        return h
    m = re.search(r"\b(\d{1,2})\s*(?::(\d{2}))?\s*(am|pm)\b", text)
    if m:
        h = int(m.group(1)) % 12
        if m.group(3) == "pm":
            h += 12
        return h
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if m:
        return int(m.group(1))
    return None


def parse_time_window(expr: str | None, forecast_hours: list[str],
                      now: datetime | None = None) -> dict:
    """
    Return {'hours': [...], 'description': str, 'matched': bool}.
    `forecast_hours` are Open-Meteo local hour strings 'YYYY-MM-DDTHH:00'.
    """
    now = now or datetime.now(IST)
    today = now.date()
    text = (expr or "").lower()

    target_dates: set = set()
    hours_allowed: set = set()
    desc = "the next 24 hours"

    if not text:
        hours_allowed = set(range(0, 24))
    else:
        # --- day part ---
        if "parso" in text:
            target_dates.add(today + timedelta(days=2)); desc = "the day after tomorrow"
        elif "kal" in text:
            target_dates.add(today + timedelta(days=1)); desc = "tomorrow"
        elif "aaj" in text or "abhi" in text:
            target_dates.add(today); desc = "today"
        elif "day after tomorrow" in text:
            target_dates.add(today + timedelta(days=2)); desc = "the day after tomorrow"
        elif "tomorrow" in text:
            target_dates.add(today + timedelta(days=1)); desc = "tomorrow"
        elif "today" in text or "tonight" in text:
            target_dates.add(today); desc = "today"

        # --- weekend ---
        if "weekend" in text:
            for d in range(0, 7):
                dd = today + timedelta(days=d)
                if dd.weekday() >= 5:
                    target_dates.add(dd)
            desc = "this weekend"

        # --- weekday names ---
        for name, idx in WEEKDAYS.items():
            if re.search(rf"\b{name}\b", text):
                delta = (idx - today.weekday()) % 7
                if delta == 0 and "next" in text:
                    delta = 7
                target_dates.add(today + timedelta(days=delta))
                desc = name.capitalize()

        # --- next N days ---
        m = re.search(r"next\s+(\d+)\s+day", text)
        if m:
            n = int(m.group(1))
            for d in range(n):
                target_dates.add(today + timedelta(days=d))
            desc = f"the next {n} days"

        # --- part of day ---
        for part, (a, b) in PARTS.items():
            if part in text:
                hours_allowed |= set(range(a, b + 1))
                if desc.startswith("the next"):
                    desc = f"this {part}"
                else:
                    desc = f"{desc} {part}"
                break

        # --- explicit clock ---
        hc = _parse_hour_clock(text)
        if hc is not None and not hours_allowed:
            hours_allowed = {hc - 1, hc, hc + 1}
            desc = f"around {hc:02d}:00"

    if not target_dates:
        target_dates = {today}
    if not hours_allowed:
        hours_allowed = set(range(0, 24))

    matched = []
    for h in forecast_hours:
        try:
            dt = datetime.fromisoformat(h)
        except ValueError:
            continue
        if dt.date() in target_dates and dt.hour in hours_allowed:
            matched.append(h)

    # fall back to the next 24h if the phrase matched nothing
    if not matched:
        matched = forecast_hours[:24]
        desc = "the next 24 hours"
        return {"hours": matched, "description": desc, "matched": False}

    return {"hours": matched, "description": desc, "matched": True}
