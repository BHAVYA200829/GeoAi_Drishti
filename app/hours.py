# -*- coding: utf-8 -*-
"""Opening hours.

OpenStreetMap tags venues with an `opening_hours` string in its own small
grammar. Recommending a ground that is locked at the hour you suggest is a real
mistake, so the common forms are parsed here and a closed venue is pushed down.

Anything the parser does not understand returns None, which means "unknown" --
we never punish a place just because its hours are written in a way we cannot
read.
"""
import re
from datetime import datetime

DAYS = {"mo": 0, "tu": 1, "we": 2, "th": 3, "fr": 4, "sa": 5, "su": 6}
DAY_RE = re.compile(r"^(mo|tu|we|th|fr|sa|su)$")
TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


def _mins(token: str) -> int | None:
    m = TIME_RE.match(token.strip())
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 24 or mi > 59:
        return None
    return h * 60 + mi


def _day_span(spec: str) -> set[int] | None:
    """'Mo-Fr' / 'Sa' / 'Mo,We,Fr' -> the set of weekdays."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            if not (DAY_RE.match(a) and DAY_RE.match(b)):
                return None
            i, j = DAYS[a], DAYS[b]
            while True:
                out.add(i)
                if i == j:
                    break
                i = (i + 1) % 7
        elif DAY_RE.match(part):
            out.add(DAYS[part])
        else:
            return None
    return out or None


def _rule_hours(spec: str) -> list[tuple[int, int]] | None:
    """'06:00-22:00' or '06:00-09:00,17:00-21:00' -> minute ranges."""
    spans = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part.lower() in ("off", "closed"):
            continue
        a, sep, b = part.partition("-")
        if not sep:
            return None
        x, y = _mins(a), _mins(b)
        if x is None or y is None:
            return None
        if y <= x:                     # crosses midnight
            spans.append((x, 24 * 60))
            spans.append((0, y))
        else:
            spans.append((x, y))
    return spans or None


def is_open(tag: str | None, hour_iso: str) -> tuple[bool | None, str | None]:
    """
    Is the venue open at this ISO hour? Returns (open, note).
    open is True / False / None (unknown).
    """
    if not tag or not hour_iso:
        return None, None
    t = " ".join(tag.split()).strip()
    low = t.lower()
    if low in ("24/7", "24x7", "always", "open"):
        return True, None
    if low in ("sunrise-sunset", "daylight"):
        return True, "open through daylight hours"

    try:
        when = datetime.fromisoformat(hour_iso)
    except ValueError:
        return None, None
    weekday = when.weekday()
    minutes = when.hour * 60 + when.minute

    for rule in t.split(";"):
        rule = rule.strip()
        if not rule:
            continue
        rl = rule.lower()
        if rl in ("24/7", "sunrise-sunset", "daylight"):
            return True, None
        # "Mo-Fr 06:00-21:00" | "06:00-21:00" | "Mo-Fr off"
        m = re.match(r"^(?P<days>[A-Za-z,\-]+)?\s*(?P<rest>.+)$", rule)
        if not m:
            continue
        days_spec = (m.group("days") or "").strip()
        rest = m.group("rest").strip()
        days = _day_span(days_spec) if days_spec else set(range(7))
        if days is None:
            continue
        if rest.lower() in ("off", "closed"):
            if weekday in days:
                return False, "closed today"
            continue
        spans = _rule_hours(rest)
        if spans is None:
            continue
        if weekday not in days:
            continue
        if any(a <= minutes < b for a, b in spans):
            return True, None
        # it is a day it opens, but not at this hour
        opens = min(a for a, _b in spans)
        return False, "opens at %02d:%02d" % (opens // 60, opens % 60)
    return None, None
