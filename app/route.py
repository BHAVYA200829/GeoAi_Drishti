# -*- coding: utf-8 -*-
"""Scoring a journey, not a place.

"Cycle from Powai to BKC at 8 AM" is not a question about a venue, so no single
comfort score answers it. Instead the route is walked, a point is taken every few
hundred metres, and each stretch is scored on its own surroundings: how close it
runs to industry and to a major road, and how much green or water is beside it.

The weather is the same across a city at one hour, so it is fetched once for the
route and the *place* variation comes from what is beside each stretch. That is
both honest and cheap.

The verdict rests on the worst stretch, because a journey is only as good as its
worst part -- but the average is reported alongside, so a single bad 300 m does
not damn a good ride.
"""
from __future__ import annotations

from .scorer import angular_diff, bearing, haversine_m

SAMPLE_EVERY_M = 600.0
INDUSTRY_M, INDUSTRY_PEN = 200.0, 26.0
MAJOR_ROAD_M, MAJOR_ROAD_PEN = 140.0, 18.0
GREEN_M, GREEN_BONUS = 250.0, 7.0
WATER_M, WATER_BONUS = 350.0, 6.0

GOOD, CAUTION = 60.0, 38.0


def sample_along(coords: list, every_m: float = SAMPLE_EVERY_M) -> list[dict]:
    """Walk a polyline and pick a point every `every_m` metres, ends included."""
    pts = [(c[0], c[1]) for c in (coords or []) if c and len(c) >= 2]
    if not pts:
        return []
    if len(pts) == 1:
        return [{"lat": pts[0][0], "lon": pts[0][1], "at_m": 0.0}]
    out = [{"lat": pts[0][0], "lon": pts[0][1], "at_m": 0.0}]
    walked = 0.0
    since = 0.0
    for (a_lat, a_lon), (b_lat, b_lon) in zip(pts, pts[1:]):
        seg = haversine_m(a_lat, a_lon, b_lat, b_lon)
        if seg <= 0:
            continue
        steps = int(seg // max(1.0, every_m - since)) if since else int(seg // every_m)
        if steps:
            for k in range(1, steps + 1):
                f = k / (steps + 1)
                out.append({"lat": a_lat + (b_lat - a_lat) * f,
                            "lon": a_lon + (b_lon - a_lon) * f,
                            "at_m": round(walked + seg * f)})
        walked += seg
        since = (since + seg) % every_m
    out.append({"lat": pts[-1][0], "lon": pts[-1][1], "at_m": round(walked)})
    return out


def _nearest(lat: float, lon: float, items: list | None):
    best = None
    for s in (items or []):
        if s.get("lat") is None or s.get("lon") is None:
            continue
        d = haversine_m(lat, lon, s["lat"], s["lon"])
        if best is None or d < best[0]:
            best = (d, s.get("name") or "unnamed")
    return best


def stretch_modifier(lat: float, lon: float, ctx: dict | None) -> tuple[float, list[str]]:
    """How much the surroundings of one stretch help or hurt it."""
    delta, flags = 0.0, []
    if not ctx:
        return 0.0, flags
    ind = _nearest(lat, lon, ctx.get("industrial"))
    if ind and ind[0] <= INDUSTRY_M:
        delta -= INDUSTRY_PEN
        flags.append(f"past {ind[1]}, an industrial area")
    road = _nearest(lat, lon, ctx.get("roads"))
    if road and road[0] <= MAJOR_ROAD_M:
        delta -= MAJOR_ROAD_PEN
        flags.append(f"alongside a major road near {road[1]}")
    green = _nearest(lat, lon, ctx.get("green"))
    if green and green[0] <= GREEN_M:
        delta += GREEN_BONUS
        flags.append(f"green cover near {green[1]}")
    water = _nearest(lat, lon, ctx.get("water"))
    if water and water[0] <= WATER_M:
        delta += WATER_BONUS
        flags.append(f"beside {water[1]}")
    return delta, flags


def road_for(lat: float, lon: float, steps: list | None) -> str | None:
    """The named road this point is on, from OSRM's steps."""
    best = None
    for s in (steps or []):
        if s.get("lat") is None or s.get("lon") is None or not s.get("name"):
            continue
        d = haversine_m(lat, lon, s["lat"], s["lon"])
        if best is None or d < best[0]:
            best = (d, s["name"])
    if best and best[0] <= 1500:
        return best[1]
    return None


def plan(samples: list[dict], ctx: dict | None, steps: list | None,
         base_by_hour: dict[str, float], hours: list[str]) -> dict:
    """
    Score every stretch at every hour and read the result.

    base_by_hour is the weather-only score for the route at each hour, so the
    difference between stretches is entirely local.
    """
    if not samples or not hours:
        return {"verdict": None, "stretches": [], "by_hour": []}

    scored = []
    for s in samples:
        delta, flags = stretch_modifier(s["lat"], s["lon"], ctx)
        road = road_for(s["lat"], s["lon"], steps)
        scored.append({"lat": s["lat"], "lon": s["lon"], "at_m": s["at_m"],
                       "delta": round(delta, 1), "flags": flags, "road": road})

    by_hour = []
    for h in hours:
        base = base_by_hour.get(h)
        if base is None:
            continue
        vals = [max(0.0, min(100.0, base + s["delta"])) for s in scored]
        by_hour.append({
            "hour": h,
            "average": round(sum(vals) / len(vals), 1),
            "worst": round(min(vals), 1),
        })
    if not by_hour:
        return {"verdict": None, "stretches": scored, "by_hour": []}

    best_hour = max(by_hour, key=lambda x: x["worst"])
    base = base_by_hour[best_hour["hour"]]
    for s in scored:
        s["score"] = round(max(0.0, min(100.0, base + s["delta"])), 1)
    worst = min(scored, key=lambda s: s["score"])

    verdict = ("go" if best_hour["worst"] >= GOOD else
               "caution" if best_hour["worst"] >= CAUTION else "avoid")
    return {
        "verdict": verdict,
        "best_hour": best_hour["hour"],
        "best_worst": best_hour["worst"],
        "best_average": best_hour["average"],
        "worst": worst,
        "stretches": scored,
        "by_hour": by_hour,
        "flags": worst.get("flags") or [],
        "road": worst.get("road"),
        "length_m": samples[-1]["at_m"] if samples else None,
    }
