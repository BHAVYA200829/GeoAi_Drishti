"""
Drishti scoring engine.

This is the heart of the assistant and the headline USP: an explicit
PLACE x TIME COMFORT SURFACE. For every candidate place and every upcoming
hour it computes a 0-100 suitability score, then reads the surface to find:

  * the best (place, hour),
  * the "golden window" (contiguous good hours),
  * the "avoid" window,
  * FLIP POINTS -- hours where the verdict changes (e.g. a ground downwind of
    an expressway at 8 AM that becomes pleasant by evening when the wind shifts).

The SLM never computes any of this. Everything here is deterministic.
"""
from __future__ import annotations

import math
from datetime import datetime

from .config import AVOID_SCORE, GOOD_SCORE
from .profiles import FACTOR_LABELS, PERSONA_RULES, get_profile, normalise_condition


# ----------------------------------------------------------------------
# small numeric helpers
# ----------------------------------------------------------------------
def interp(points: list[list[float]], x: float | None) -> float:
    """Piecewise-linear comfort in [0,1] over ascending (x, y) points."""
    if x is None:
        return 0.5  # neutral when data is missing
    pts = sorted(points, key=lambda p: p[0])
    if x <= pts[0][0]:
        return float(pts[0][1])
    if x >= pts[-1][0]:
        return float(pts[-1][1])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return float(y1)
            t = (x - x0) / (x1 - x0)
            return float(y0 + t * (y1 - y0))
    return 0.5


def heat_index_c(temp: float | None, humidity: float | None) -> float | None:
    """Simple apparent-heat approximation (deg C)."""
    if temp is None or humidity is None:
        return None
    if temp < 27:
        return temp
    t, r = temp, humidity
    hi = (-8.784695 + 1.61139411 * t + 2.338549 * r - 0.14611605 * t * r
          - 0.012308094 * t * t - 0.016424828 * r * r
          + 0.002211732 * t * t * r + 0.00072546 * t * r * r
          - 0.000003582 * t * t * r * r)
    return round(hi, 1)


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing in degrees from point 1 to point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def angular_diff(a: float, b: float) -> float:
    d = abs((a - b + 180) % 360 - 180)
    return d


# ----------------------------------------------------------------------
# hourly series helpers (Open-Meteo returns parallel arrays)
# ----------------------------------------------------------------------
def hourly_map(block: dict | None, var: str) -> dict[str, float]:
    if not block:
        return {}
    h = block.get("hourly", {})
    times, vals = h.get("time", []), h.get(var, [])
    return {t: v for t, v in zip(times, vals) if v is not None}


# ----------------------------------------------------------------------
# geospatial penalty: is the place DOWNWIND of a pollution source?
# ----------------------------------------------------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import asin, cos, radians, sin, sqrt
    r = 6371000.0
    p1, p2 = radians(lat1), radians(lat2)
    dp, dl = p2 - p1, radians(lon2 - lon1)
    a = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return 2 * r * asin(sqrt(a))


def nearest(place: dict, items: list | None) -> tuple[float, str] | None:
    """Distance in metres to the closest item, with its name."""
    best = None
    for s in (items or []):
        if s.get("lat") is None or s.get("lon") is None:
            continue
        d = haversine_m(place["lat"], place["lon"], s["lat"], s["lon"])
        if best is None or d < best[0]:
            best = (d, s.get("name") or "a green space")
    return best


# how far green cover still shades a spot, and how far water still cools it
SHADE_M, SHADE_DROP = 450.0, 2.5     # trees / park next door
WATER_M, WATER_DROP = 700.0, 2.0     # a lake or river close by
COOL_FROM_C = 27.0                   # below this it is pleasant anyway


def shade_and_cooling(place: dict, ctx: dict | None, temp: float | None) -> tuple[float, dict]:
    """
    Green cover shades a place and water cools the air around it. Both only matter
    when it is actually hot, so the benefit scales in as the temperature climbs.
    Returns the temperature drop and the facts worth mentioning.
    """
    facts: dict = {"shade_by": None, "shade_m": None, "water_by": None, "water_m": None}
    if not ctx:
        return 0.0, facts
    g = nearest(place, ctx.get("green"))
    w = nearest(place, ctx.get("water"))
    if g and g[0] <= SHADE_M:
        facts["shade_by"], facts["shade_m"] = g[1], round(g[0])
    if w and w[0] <= WATER_M:
        facts["water_by"], facts["water_m"] = w[1], round(w[0])

    drop = (SHADE_DROP if facts["shade_by"] else 0.0) + (WATER_DROP if facts["water_by"] else 0.0)
    if not drop:
        return 0.0, facts
    if temp is None or temp <= COOL_FROM_C:
        return 0.0, facts          # pleasant already: note it, do not score it
    scale = min(1.0, (temp - COOL_FROM_C) / 10.0)
    return round(drop * scale, 2), facts


# ----------------------------------------------------------------------
# the sun
# ----------------------------------------------------------------------
def _hms(text: str | None) -> int | None:
    """'2026-10-23T18:12' -> minutes past midnight."""
    if not text or "T" not in text:
        return None
    try:
        hh, mm = text.split("T")[1][:5].split(":")
        return int(hh) * 60 + int(mm)
    except Exception:  # noqa: BLE001
        return None


def sun_position(lat: float, lon: float, when_iso: str) -> tuple[float, float] | None:
    """
    Solar elevation and azimuth in degrees, from the standard low-precision
    algorithm (accurate to a fraction of a degree, which is plenty here).
    Azimuth is compass-style: 0 = north, 90 = east, 180 = south, 270 = west.
    """
    from datetime import datetime, timezone
    from math import acos, asin, cos, degrees, radians, sin, tan
    try:
        dt = datetime.fromisoformat(when_iso)
    except ValueError:
        return None
    if dt.tzinfo is None:
        from .tz import IST          # every hour in this app is India Standard Time
        dt = dt.replace(tzinfo=IST)
    # days since J2000
    n = (dt - datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds() / 86400.0
    L = (280.460 + 0.9856474 * n) % 360.0            # mean longitude
    g = radians((357.528 + 0.9856003 * n) % 360.0)   # mean anomaly
    lam = radians(L + 1.915 * sin(g) + 0.020 * sin(2 * g))
    eps = radians(23.439 - 0.0000004 * n)
    dec = asin(sin(eps) * sin(lam))
    ra = (degrees(__import__("math").atan2(cos(eps) * sin(lam), cos(lam)))) % 360.0

    gmst = (18.697374558 + 24.06570982441908 * n) % 24.0
    lst = (gmst * 15.0 + lon) % 360.0
    ha = radians((lst - ra + 180.0) % 360.0 - 180.0)
    latr = radians(lat)
    elev = asin(sin(latr) * sin(dec) + cos(latr) * cos(dec) * cos(ha))
    cos_az = (sin(dec) - sin(elev) * sin(latr)) / (cos(elev) * cos(latr) + 1e-9)
    az = degrees(acos(max(-1.0, min(1.0, cos_az))))
    if sin(ha) > 0:
        az = 360.0 - az
    return round(degrees(elev), 1), round(az % 360.0, 1)


def golden_score(hour: str, sunrise: str | None, sunset: str | None) -> float:
    """How close the middle of this hour sits to sunrise or sunset."""
    hm = _hms(hour)
    if hm is None:
        return 0.0
    mid = hm + 30
    best = 0.0
    for ev in (sunrise, sunset):
        em = _hms(ev)
        if em is None:
            continue
        d = abs(mid - em)
        if d <= 45:
            best = max(best, 1.0)        # inside the golden hour
        elif d <= 90:
            best = max(best, 0.6)        # the shoulder of it
        elif d <= 150:
            best = max(best, 0.2)        # blue hour
    return best


def open_toward(place: dict, ctx: dict | None, azimuth: float | None,
                window: float = 60.0) -> str | None:
    """Is there water or open green in the direction the sun will be?"""
    if ctx is None or azimuth is None:
        return None
    best = None
    for kind, items in (("water", ctx.get("water")), ("green", ctx.get("green"))):
        for s in (items or []):
            if s.get("lat") is None or s.get("lon") is None:
                continue
            b = bearing(place["lat"], place["lon"], s["lat"], s["lon"])
            d = angular_diff(b, azimuth)
            if d <= window:
                near = haversine_m(place["lat"], place["lon"], s["lat"], s["lon"])
                if best is None or near < best[0]:
                    best = (near, s.get("name") or kind, kind)
    if best and best[0] <= 3000:
        return best[1]
    return None


def closed_at(place: dict, hour: str) -> tuple[bool, str | None]:
    """A venue that is shut at this hour is not a recommendation."""
    from .hours import is_open
    tags = place.get("tags") or {}
    tag = tags.get("opening_hours")
    if not tag:
        return False, None
    open_, note = is_open(tag, hour)
    if open_ is False:
        return True, (f"closed at this hour ({note})" if note else "closed at this hour")
    return False, None


def upwind_penalty(place: dict, ctx: dict | None, wind_from_deg: float | None) -> tuple[float, str | None]:
    """
    If a pollution source (industry / major road) sits upwind of the place,
    air is carried toward it. Meteorological wind direction = direction the
    wind blows FROM. A source at bearing B from the place is upwind when
    B ~= wind_from_deg.
    """
    if ctx is None or wind_from_deg is None:
        return 0.0, None
    sources = (ctx.get("industrial", []) or []) + (ctx.get("roads", []) or [])
    worst, worst_name = 0.0, None
    for s in sources:
        if s.get("lat") is None or s.get("lon") is None:
            continue
        b = bearing(place["lat"], place["lon"], s["lat"], s["lon"])
        diff = angular_diff(b, wind_from_deg)
        if diff <= 50:  # source is roughly upwind
            strength = 1.0 - diff / 50.0
            pen = 30.0 * strength
            if pen > worst:
                worst, worst_name = pen, s.get("name", "a nearby source")
    if worst > 0:
        return worst, f"downwind of {worst_name} while the wind blows from there"
    return 0.0, None


# ----------------------------------------------------------------------
# score a single (place, hour)
# ----------------------------------------------------------------------
def score_slot(place: dict, hour: str, wx: dict, aq: dict, ctx: dict | None,
               condition: str | None, sunrise: str | None, sunset: str | None) -> dict:
    profile = get_profile(place.get("activity"))
    tmap = {v: hourly_map(wx, v) for v in
            ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
             "wind_direction_10m", "precipitation_probability", "uv_index", "is_day"]}
    amap = {v: hourly_map(aq, v) for v in ["us_aqi", "pm2_5", "grass_pollen", "birch_pollen", "alder_pollen"]}

    def val(m, k):
        return m.get(k, {}).get(hour)

    temp = val(tmap, "temperature_2m")
    hum = val(tmap, "relative_humidity_2m")
    wind = val(tmap, "wind_speed_10m")
    wdir = val(tmap, "wind_direction_10m")
    rain = val(tmap, "precipitation_probability")
    uv = val(tmap, "uv_index")
    is_day = val(tmap, "is_day")
    aqi = val(amap, "us_aqi")
    pollen = max([x for x in [val(amap, "grass_pollen"), val(amap, "birch_pollen"),
                              val(amap, "alder_pollen")] if x is not None], default=None)

    # green cover and water: shade and cooling, before anything is scored
    cooling, cool_facts = shade_and_cooling(place, ctx, temp)
    temp_eff = (temp - cooling) if (temp is not None and cooling) else temp

    facts = {
        "temp": temp, "humidity": hum, "wind": wind, "rain_prob": rain, "uv": uv,
        "aqi": aqi, "pollen": pollen,
        "heat_index": heat_index_c(temp_eff, hum),
        "slope": place.get("slope"),
        "temp_effective": temp_eff,
        "cooling": cooling or None,
        **cool_facts,
    }

    subs: dict[str, float] = {
        "temp": interp(profile["temp"], temp_eff),
        "wind": interp(profile["wind"], wind),
        "humidity": interp(profile["humidity"], hum),
        "rain_prob": interp(profile["rain_prob"], rain),
        "aqi": interp(profile["aqi"], aqi),
        "uv": interp(profile["uv"], uv),
    }

    # golden hour for photography: the real distance to sunrise or sunset
    sun_elev = sun_az = None
    if profile.get("golden_hour"):
        subs["golden"] = golden_score(hour, sunrise, sunset)
        sun = sun_position(place["lat"], place["lon"], hour)
        if sun:
            sun_elev, sun_az = sun
            facts["sun_elevation"] = sun_elev
            facts["sun_azimuth"] = sun_az
            facing = open_toward(place, ctx, sun_az)
            facts["open_toward_sun"] = facing
            if facing and subs["golden"] >= 0.6:
                subs["golden"] = min(1.0, subs["golden"] + 0.25)

    weights = profile["weights"]
    score = sum(weights[k] * subs.get(k, 0.5) for k in weights) * 100.0
    reasons: list[str] = []

    # daylight
    if profile.get("needs_daylight"):
        if is_day == 0:
            score = min(score, 5.0)
            reasons.append("outside daylight hours")
        elif is_day is None and sunrise and sunset:
            if hour < sunrise or hour > sunset:
                score = min(score, 5.0)
                reasons.append("outside daylight hours")

    # activity hard limits
    hr = profile.get("hard_rain_prob")
    if hr and rain is not None and rain >= hr:
        score = min(score, 15.0)
        reasons.append(f"rain very likely ({rain:.0f}%)")

    # is the place even open at this hour?
    shut, shut_why = closed_at(place, hour)
    if shut:
        score = min(score, 8.0)
        reasons.append(shut_why)

    # geospatial: downwind of a source
    pen, why = upwind_penalty(place, ctx, wdir)
    if pen:
        score -= pen
        reasons.append(why)

    # persona hard overrides
    cond = condition
    if cond in PERSONA_RULES:
        for rule in PERSONA_RULES[cond]:
            v = facts.get(rule["factor"])
            if v is None:
                continue
            hit = (rule["op"] == ">" and v > rule["value"]) or (rule["op"] == "<" and v < rule["value"])
            if hit:
                if rule["cap"] is None:
                    score = 0.0
                    reasons.append(f"not suitable: {rule['why']}")
                else:
                    score = min(score, float(rule["cap"]))
                    reasons.append(f"caution: {rule['why']}")

    # a couple of positive reasons
    if not reasons and score >= GOOD_SCORE:
        reasons.append("conditions are comfortable for this activity")

    if profile.get("golden_hour") and subs.get("golden", 0) >= 0.6 and sun_az is not None:
        where = ("the west" if 240 <= sun_az <= 300 else
                 "the east" if 60 <= sun_az <= 120 else
                 "the north" if sun_az < 60 or sun_az > 300 else "the south")
        facing = facts.get("open_toward_sun")
        line = f"the sun is low and to {where} at this hour"
        if facing:
            line += f", and {facing} lies that way, so you get a clear horizon"
        reasons.append(line)

    if cooling:
        notes = []
        if cool_facts.get("shade_by"):
            notes.append(f"shaded by {cool_facts['shade_by']} next door")
        if cool_facts.get("water_by"):
            notes.append(f"close to {cool_facts['water_by']}, which cools the air")
        if notes:
            reasons.append("; ".join(notes))

    return {
        "score": round(max(0.0, min(100.0, score)), 1),
        "subs": {k: round(v, 2) for k, v in subs.items()},
        "facts": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in facts.items()},
        "reasons": reasons,
    }


# ----------------------------------------------------------------------
# travel cost: is the place "worth the trip"?
# ----------------------------------------------------------------------
def travel_penalty(minutes: float | None) -> tuple[float, str | None]:
    if minutes is None or minutes <= 15:
        return 0.0, None
    pen = min(30.0, (minutes - 15) * 0.7)
    return pen, f"about {minutes:.0f} min away"


# ----------------------------------------------------------------------
# build the full surface and analyse it
# ----------------------------------------------------------------------
def build_surface(places: list[dict], wx: dict, aq: dict, ctx_by_place: dict,
                  condition: str | None, hours: list[str],
                  travel_by_place: dict | None = None,
                  sunrise: str | None = None, sunset: str | None = None) -> list[dict]:
    rows: list[dict] = []
    for p in places:
        ctx = (ctx_by_place or {}).get(p["name"])
        for h in hours:
            r = score_slot(p, h, wx, aq, ctx, condition, sunrise, sunset)
            tpen, _ = travel_penalty((travel_by_place or {}).get(p["name"]))
            rows.append({
                "place": p["name"], "lat": p["lat"], "lon": p["lon"], "hour": h,
                "score": round(max(0.0, r["score"] - tpen), 1),
                "raw_score": r["score"], "travel_penalty": round(tpen, 1),
                "subs": r["subs"], "facts": r["facts"], "reasons": r["reasons"],
                "travel_minutes": (travel_by_place or {}).get(p["name"]),
            })
    return rows


def _fmt(hour: str) -> str:
    try:
        return datetime.fromisoformat(hour).strftime("%a %d %b, %I:%M %p")
    except Exception:
        return hour


def analyse(rows: list[dict], prefer: dict | None = None) -> dict:
    """Read the surface: best, golden window, avoid window, flip points.

    `prefer` is the (place, hour) the SLM chose. It wins if it is genuinely one of
    the scored rows; otherwise the highest score wins, as always.
    """
    if not rows:
        return {"best": None, "golden_window": None, "avoid": None, "flips": [], "alternatives": []}

    by_place: dict[str, list[dict]] = {}
    for r in rows:
        by_place.setdefault(r["place"], []).append(r)
    for v in by_place.values():
        v.sort(key=lambda r: r["hour"])

    best = max(rows, key=lambda r: r["score"])
    if prefer is not None:
        same = next((r for r in rows
                     if r["place"] == prefer.get("place") and r["hour"] == prefer.get("hour")),
                    None)
        if same is not None:
            best = same

    # golden window: longest contiguous run >= GOOD_SCORE, across all places
    golden, avoid = None, None
    for name, series in by_place.items():
        run = []
        for r in series:
            if r["score"] >= GOOD_SCORE:
                run.append(r)
            else:
                if len(run) >= 2 and (golden is None or len(run) > golden["_len"]):
                    golden = {"place": name, "from": run[0]["hour"], "to": run[-1]["hour"],
                              "avg_score": round(sum(x["score"] for x in run) / len(run), 1),
                              "_len": len(run)}
                run = []
        if len(run) >= 2 and (golden is None or len(run) > golden["_len"]):
            golden = {"place": name, "from": run[0]["hour"], "to": run[-1]["hour"],
                      "avg_score": round(sum(x["score"] for x in run) / len(run), 1),
                      "_len": len(run)}

    worst = min(rows, key=lambda r: r["score"])
    if worst["score"] <= AVOID_SCORE:
        avoid = {"place": worst["place"], "hour": worst["hour"], "score": worst["score"],
                 "reasons": worst["reasons"]}

    # flip points: for each place, hours where the verdict crosses GOOD_SCORE
    flips: list[dict] = []
    for name, series in by_place.items():
        for a, b in zip(series, series[1:]):
            crossed_up = a["score"] < GOOD_SCORE <= b["score"]
            crossed_down = a["score"] >= GOOD_SCORE > b["score"]
            if crossed_up or crossed_down:
                # for an improvement, the earlier hour explains why it was poor;
                # for a worsening, the later hour explains why it turned bad
                src = (a["reasons"] if crossed_up else b["reasons"]) or \
                      (b["reasons"] if crossed_up else a["reasons"]) or ["conditions shift"]
                flips.append({
                    "place": name, "at": b["hour"],
                    "direction": "improves" if crossed_up else "worsens",
                    "from_score": a["score"], "to_score": b["score"],
                    "reason": src[0],
                })
    # also: place-ranking flip (top place changes between consecutive hours)
    hours = sorted({r["hour"] for r in rows})
    rank_flips = []
    prev_top = None
    for h in hours:
        cand = [r for r in rows if r["hour"] == h]
        top = max(cand, key=lambda r: r["score"])
        if prev_top and top["place"] != prev_top:
            rank_flips.append({"at": h, "from": prev_top, "to": top["place"]})
        prev_top = top["place"]

    # alternatives: best hour per other place
    alts = []
    for name, series in by_place.items():
        if name == best["place"]:
            continue
        b = max(series, key=lambda r: r["score"])
        alts.append(b)
    alts.sort(key=lambda r: r["score"], reverse=True)

    if golden:
        golden.pop("_len", None)

    return {
        "best": best, "golden_window": golden, "avoid": avoid,
        "flips": flips, "rank_flips": rank_flips, "alternatives": alts[:12],
        "hours": hours,
    }
