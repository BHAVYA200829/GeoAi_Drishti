# -*- coding: utf-8 -*-
"""The SLM's reasoning layer, and the guard around it.

The small model is not asked to recall anything. The deterministic engine fetches
the weather, the places and the geospatial context, and scores every (place, hour)
pair into a *comfort surface*. That surface is handed to the SLM as FACTS, and the
SLM decides:

  * which place and hour to go for,
  * why that one, and why the runner-up lost,
  * what to carry and what to watch for.

Everything it returns is then validated against the surface. A place or an hour
that is not in the facts is rejected, and a sentence that quotes a number the
facts do not contain is dropped. If the decision does not survive, the
deterministic pick stands -- the model can only ever choose between real options,
never invent one.

This module is pure: no network, no model, so it is fully testable on its own.
"""
from __future__ import annotations

import re

MAX_PLACES = 5          # how many candidates the model sees
MAX_HOURS = 3           # and how many hours for each
ALLOWED_CARRY = {       # what the model may tell the user to bring
    "water", "hydration", "a hat", "sun cream", "sunglasses", "an umbrella",
    "a rain jacket", "a mask", "a warm layer", "a jacket", "insect repellent",
    "a torch", "sturdy shoes", "a charged phone", "a first-aid kit", "snacks",
}
ALLOWED_WATCH = {       # and what it may warn about
    "heat", "cold", "rain", "air quality", "wind", "uv", "pollen", "darkness",
    "crowds", "closed", "the trip", "uneven ground", "insects",
}


def _hhmm(hour: str) -> str:
    return (hour or "")[-5:]


def build_facts(rows: list[dict], *, activity: str, condition: str | None,
                window: str | None, now_iso: str | None = None,
                budget_minutes: float | None = None) -> dict:
    """
    Compress the comfort surface into a small, unambiguous payload: the best few
    places, each with its best few hours and the numbers behind them.
    """
    by_place: dict[str, list[dict]] = {}
    for r in rows or []:
        by_place.setdefault(r["place"], []).append(r)

    ranked = sorted(by_place.items(),
                    key=lambda kv: max(x["score"] for x in kv[1]), reverse=True)

    places = []
    for name, rs in ranked[:MAX_PLACES]:
        rs = sorted(rs, key=lambda x: x["score"], reverse=True)[:MAX_HOURS]
        hours = []
        for r in rs:
            f = r.get("facts") or {}
            hours.append({
                "hour": _hhmm(r["hour"]),
                "score": r["score"],
                "temp_c": f.get("temp"),
                "feels_like_c": f.get("heat_index"),
                "rain_pct": f.get("rain_prob"),
                "aqi": f.get("aqi"),
                "wind_kmh": f.get("wind"),
                "uv": f.get("uv"),
                "travel_min": r.get("travel_minutes"),
                "flags": [x for x in (r.get("reasons") or []) if x],
            })
        best = hours[0] if hours else {}
        places.append({
            "name": name,
            "best_hour": best.get("hour"),
            "best_score": best.get("score"),
            "travel_min": best.get("travel_min"),
            "shaded_by": (rs[0].get("facts") or {}).get("shade_by"),
            "near_water": (rs[0].get("facts") or {}).get("water_by"),
            "sun_azimuth": (rs[0].get("facts") or {}).get("sun_azimuth"),
            "hours": hours,
        })

    return {
        "task": activity,
        "person": condition or "none",
        "when": window or "the next few hours",
        "now": now_iso,
        "travel_budget_minutes": budget_minutes,
        "places": places,
    }


# ---------------------------------------------------------------- the guard
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def match_place(name: str, rows: list[dict]) -> str | None:
    """Find the real place name the model meant, allowing for a loose match."""
    want = _norm(name)
    if not want:
        return None
    names = []
    for r in rows or []:
        if r["place"] not in names:
            names.append(r["place"])
    for n in names:                       # exact
        if _norm(n) == want:
            return n
    for n in names:                       # the model dropped or added a word
        a, b = _norm(n), want
        if a.startswith(b) or b.startswith(a) or b in a or a in b:
            return n
    return None


def match_row(place: str, hour: str, rows: list[dict]) -> dict | None:
    """The exact (place, hour) row the model picked."""
    want = (hour or "").strip()[-5:]
    for r in rows or []:
        if r["place"] == place and _hhmm(r["hour"]) == want:
            return r
    return None


def _numbers_in(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def grounded(text: str, facts: dict) -> str | None:
    """
    Keep the model's prose only where it does not quote a figure the facts do not
    contain. A sentence with an unverifiable number is dropped, not shown.
    """
    if not text:
        return None
    allowed = _numbers_in(str(facts))
    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        nums = _numbers_in(sentence)
        if nums and not nums <= allowed:
            continue
        kept.append(sentence.strip())
    out = " ".join(kept).strip()
    return out or None


def _clean_list(items, allowed: set[str]) -> list[str]:
    out = []
    for x in (items or []):
        if not isinstance(x, str):
            continue
        x = x.strip().rstrip(".")
        if not x or len(x) > 80:
            continue
        low = x.lower()
        if low in allowed or any(low.startswith(a) or a in low for a in allowed):
            out.append(x)
    return out[:4]


def validate_decision(decision: dict | None, rows: list[dict], facts: dict) -> dict | None:
    """
    Turn the model's answer into something usable, or None if it does not hold up.
    Never returns a place or an hour that is not in the surface.
    """
    if not isinstance(decision, dict) or not rows:
        return None

    place = match_place(str(decision.get("best_place") or ""), rows)
    if not place:
        return None
    hour = str(decision.get("best_hour") or "").strip()
    row = match_row(place, hour, rows)
    if row is None:
        # the model named a place but a bad hour: fall back to that place's best
        best = max((r for r in rows if r["place"] == place),
                   key=lambda r: r["score"], default=None)
        if best is None:
            return None
        row = best

    runner = match_place(str(decision.get("runner_up") or ""), rows)
    if runner == place:
        runner = None

    return {
        "row": row,
        "place": place,
        "hour": _hhmm(row["hour"]),
        "why": grounded(decision.get("why"), facts),
        "runner_up": runner,
        "why_not_runner_up": grounded(decision.get("why_not_runner_up"), facts),
        "carry": _clean_list(decision.get("carry"), ALLOWED_CARRY),
        "watch": _clean_list(decision.get("watch"), ALLOWED_WATCH),
        "confidence": (str(decision.get("confidence") or "").lower()
                       if str(decision.get("confidence") or "").lower() in
                       ("high", "medium", "low") else "medium"),
    }
