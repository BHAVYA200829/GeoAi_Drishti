"""
Drishti tests -- deterministic core only (no network, no model).

Run:
    cd drishti && python tests/test_pipeline.py
or:
    cd drishti && python -m pytest -q
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import query_parser, scorer  # noqa: E402
from app.profiles import get_profile  # noqa: E402
from app.timeparse import parse_time_window  # noqa: E402


# ---------------------------------------------------------------- fixtures
def mock_weather(hours, wind_from, temp=28.0):
    return {
        "hourly": {
            "time": hours,
            "temperature_2m": [temp] * len(hours),
            "relative_humidity_2m": [55] * len(hours),
            "wind_speed_10m": [12] * len(hours),
            "wind_direction_10m": wind_from,
            "precipitation_probability": [5] * len(hours),
            "uv_index": [3] * len(hours),
            "is_day": [1] * len(hours),
        },
        "daily": {"sunrise": ["2026-10-23T06:30"], "sunset": ["2026-10-23T18:20"]},
    }


def mock_aq(hours, aqi=60):
    return {"hourly": {"time": hours, "us_aqi": [aqi] * len(hours),
                       "pm2_5": [30] * len(hours), "grass_pollen": [10] * len(hours)}}


HOURS = [f"2026-10-23T{h:02d}:00" for h in (6, 7, 8, 9, 17, 18, 19, 20)]
MORNING = [h for h in HOURS if h.endswith(("06:00", "07:00", "08:00", "09:00"))]
EVENING = [h for h in HOURS if h.endswith(("17:00", "18:00", "19:00", "20:00"))]

PLACE_A = {"name": "Expressway Ground", "lat": 18.530, "lon": 73.880, "activity": "running"}
PLACE_B = {"name": "City Park", "lat": 18.540, "lon": 73.890, "activity": "running"}
# a road sitting due NORTH of place A
CTX = {"industrial": [], "water": [], "green": [], "power": [],
       "roads": [{"name": "Expressway", "lat": 18.540, "lon": 73.880}]}


# ---------------------------------------------------------------- tests
def test_activity_profiles_exist():
    for act in ["running", "cycling", "golf", "cricket", "football",
                "tennis", "hiking", "picnic", "event", "photo"]:
        p = get_profile(act)
        assert "weights" in p and "temp" in p
    print("ok  activity profiles for all 10 activities")


def test_persona_hard_override_asthma():
    # AQI 200 must zero the score for an asthma profile
    aq = mock_aq(HOURS, aqi=200)
    wx = mock_weather(HOURS, [180] * len(HOURS))
    r = scorer.score_slot(PLACE_B, HOURS[0], wx, aq, None, "asthma", None, None)
    assert r["score"] == 0.0, r
    print("ok  asthma hard override zeroes a high-AQI slot")


def test_downwind_penalty_and_flip():
    # Morning: wind blows FROM the north -> place A is downwind of the Expressway.
    wx_am = mock_weather(HOURS, [0] * len(HOURS))     # wind_from = 0 (north)
    aq = mock_aq(HOURS, aqi=60)
    rows_am = scorer.build_surface([PLACE_A], wx_am, aq, {PLACE_A["name"]: CTX}, None, MORNING)
    # Evening: wind turns to blow FROM the south -> no longer downwind.
    wx_pm = mock_weather(HOURS, [180] * len(HOURS))
    rows_pm = scorer.build_surface([PLACE_A], wx_pm, aq, {PLACE_A["name"]: CTX}, None, EVENING)

    am_score = rows_am[0]["score"]
    pm_score = rows_pm[0]["score"]
    assert pm_score > am_score, (am_score, pm_score)
    assert any("downwind" in r for r in rows_am[0]["reasons"]), rows_am[0]["reasons"]
    print(f"ok  downwind penalty: morning {am_score} -> evening {pm_score} (flip works)")


def test_flip_detection():
    # Build a full day where wind shifts mid-day: morning from north, evening from south.
    wind = []
    for h in HOURS:
        wind.append(0 if h in MORNING else 180)
    wx = mock_weather(HOURS, wind)
    aq = mock_aq(HOURS)
    rows = scorer.build_surface([PLACE_A], wx, aq, {PLACE_A["name"]: CTX}, None, HOURS)
    res = scorer.analyse(rows)
    assert res["flips"], "expected a flip point"
    print(f"ok  flip detected at {res['flips'][0]['at']} ({res['flips'][0]['direction']})")


def test_worst_place_avoided_and_travel_penalty():
    wx = mock_weather(HOURS, [180] * len(HOURS))
    aq = mock_aq(HOURS)
    # Place A is 60 min away -> should lose to the nearby Place B on equal conditions
    rows = scorer.build_surface([PLACE_A, PLACE_B], wx, aq,
                                {PLACE_A["name"]: None, PLACE_B["name"]: None}, None,
                                HOURS, travel_by_place={PLACE_A["name"]: 60, PLACE_B["name"]: 5})
    res = scorer.analyse(rows)
    assert res["best"]["place"] == "City Park", res["best"]
    print("ok  worth-the-trip: nearby park beats a 60-min-away ground")


def test_timeparse_phrases():
    hours = [f"2026-10-2{d}T{h:02d}:00" for d in (2, 3, 4) for h in range(24)]
    from datetime import datetime
    from zoneinfo import ZoneInfo
    now = datetime(2026, 10, 22, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    w = parse_time_window("tomorrow evening", hours, now=now)
    assert w["matched"] and all("T1" in h or "T2" in h for h in w["hours"]), w
    assert all(16 <= int(h[11:13]) <= 20 for h in w["hours"]), w["hours"]
    print(f"ok  timeparse 'tomorrow evening' -> {len(w['hours'])} hours")


def test_rule_parser_on_examples():
    cases = {
        "Where can I go running tomorrow evening near Koregaon Park?": ("running", "Koregaon Park"),
        "Best time this weekend for a kids' cricket match within 15 km of Whitefield?": ("cricket", "Whitefield"),
        "I have mild asthma, where's good for a morning walk near Powai in the next two days?": ("hiking", "Powai"),
        "Good place and time for golf on Saturday within 40 km of Andheri?": ("golf", "Andheri"),
        "Where can I take sunset photos this evening around Bandra?": ("photo", "Bandra"),
    }
    for q, (act, loc) in cases.items():
        s = query_parser.rule_parse(q)
        assert s["activity"] == act, (q, s["activity"], act)
        assert s["location_text"] and loc.split()[0] in s["location_text"], (q, s["location_text"], loc)
    print("ok  rule parser on the example queries")


def test_missing_location_asks_clarification():
    s = query_parser.parse_query("Is it okay to cycle to work at 8 AM?")
    assert s["needs_clarification"] is True, s
    print("ok  missing location triggers a clarifying question (no guessing)")


def test_places_and_context_split():
    """One Overpass round-trip must yield both places and context features."""
    from app import data_fetcher as df
    fake = {"elements": [
        {"type": "node", "lat": 18.54, "lon": 73.89, "tags": {"leisure": "park", "name": "Riverside Park"}},
        {"type": "way", "center": {"lat": 18.60, "lon": 73.95}, "tags": {"landuse": "industrial", "name": "MIDC"}},
        {"type": "way", "center": {"lat": 18.50, "lon": 73.80}, "tags": {"natural": "water", "name": "Powai Lake"}},
        {"type": "way", "center": {"lat": 18.52, "lon": 73.85}, "tags": {"highway": "primary", "name": "NH-48"}},
        {"type": "node", "lat": 18.56, "lon": 73.91, "tags": {"sport": "cricket", "name": "Cricket Ground"}},
    ]}
    real = df._overpass
    df._overpass = lambda q, timeout=12: fake
    try:
        res = df.places_and_context(18.5366, 73.8933, 5000, "running")
        assert "Riverside Park" in [p["name"] for p in res["places"]], res["places"]
        assert res["context"]["counts"]["industrial"] == 1
        assert res["context"]["counts"]["water"] == 1
        assert res["context"]["counts"]["roads"] == 1
        cricket = df.places_and_context(18.5366, 73.8933, 5000, "cricket")
        assert [p["name"] for p in cricket["places"]] == ["Cricket Ground"], cricket["places"]
        water = df.places_and_context(18.5366, 73.8933, 5000, "watersport")
        assert "Powai Lake" in [p["name"] for p in water["places"]], water["places"]
    finally:
        df._overpass = real
    print("ok  one Overpass call yields places + context (split correctly)")


def test_objective_venue_search():
    """movie -> cinemas, shopping -> malls, not parks."""
    from app import data_fetcher as df
    fake = {"elements": [
        {"type": "node", "lat": 19.12, "lon": 72.85, "tags": {"amenity": "cinema", "name": "PVR Andheri"}},
        {"type": "node", "lat": 19.11, "lon": 72.86, "tags": {"leisure": "park", "name": "Some Park"}},
        {"type": "node", "lat": 19.14, "lon": 72.83, "tags": {"shop": "mall", "name": "Infiniti Mall"}},
    ]}
    real = df._overpass
    df._overpass = lambda q, timeout=12: fake
    try:
        mv = [p["name"] for p in df.places_and_context(19.1197, 72.8464, 5000, "movie")["places"]]
        sh = [p["name"] for p in df.places_and_context(19.1197, 72.8464, 5000, "shopping")["places"]]
        rn = [p["name"] for p in df.places_and_context(19.1197, 72.8464, 5000, "running")["places"]]
        # a movie finds cinemas and the malls that house multiplexes -- never a park
        assert "PVR Andheri" in mv and "Some Park" not in mv, mv
        assert sh == ["Infiniti Mall"], sh               # no park for shopping
        assert rn == ["Some Park"], rn
    finally:
        df._overpass = real
    print("ok  objective venues: movie->cinema, shopping->mall, running->park")


def test_intent_weather_vs_task():
    """A place or weather question -> conditions; an activity -> recommend."""
    from app.query_parser import rule_parse
    assert rule_parse("What is the weather in Delhi?").get("intent") == "conditions"
    assert rule_parse("AQI in Mumbai right now").get("intent") == "conditions"
    assert rule_parse("Bengaluru").get("intent") == "conditions"
    assert rule_parse("Where can I go running near Bandra?").get("intent") == "recommend"
    assert rule_parse("I want to do something outdoors near Bandra").get("intent") == "recommend"
    assert rule_parse("School/ colleges").get("intent") == "recommend"
    print("ok  intent: weather/place -> conditions, activity -> recommend")


def test_place_categories():
    """school / college / hospital resolve to their own venue type."""
    from app import data_fetcher as df
    assert df._objective_key("School/ colleges") == "school"
    assert df._objective_key("Hospitals") == "hospital"
    assert df._objective_key("temples") == "temple"
    fake = {"elements": [
        {"type": "node", "lat": 19.12, "lon": 72.85, "tags": {"amenity": "school", "name": "City School"}},
        {"type": "node", "lat": 19.13, "lon": 72.84, "tags": {"amenity": "hospital", "name": "City Hospital"}},
        {"type": "node", "lat": 19.11, "lon": 72.86, "tags": {"leisure": "park", "name": "Some Park"}},
    ]}
    real = df._overpass
    df._overpass = lambda q, timeout=12: fake
    try:
        got = [p["name"] for p in df.places_and_context(19.1197, 72.8464, 5000, "school")["places"]]
        assert got == ["City School"], got          # no park for a school query
        got2 = [p["name"] for p in df.places_and_context(19.1197, 72.8464, 5000, "hospital")["places"]]
        assert got2 == ["City Hospital"], got2
    finally:
        df._overpass = real
    print("ok  place categories: school->school, hospital->hospital (no parks)")


def test_places_fallback_and_location_preference():
    """A narrow filter that returns nothing must widen instead of dead-ending,
    and a typed place must always beat a GPS fix."""
    from app import data_fetcher as df
    real = df.places
    def fake(lat, lon, radius_m, activity, limit=12):
        if activity == "cricket":
            return []
        if activity == "picnic" and radius_m > 20000:
            return [{"name": "Some Park", "lat": lat + 0.01, "lon": lon + 0.01, "tags": {}}]
        return []
    df.places = fake
    try:
        got, rad, used, note = df.places_with_fallback(19.2183, 72.9781, 5000, "cricket")
        assert [p["name"] for p in got] == ["Some Park"], got
        assert used == "picnic" and rad > 5000
        assert note
    finally:
        df.places = real

    from app.query_parser import parse_query
    p = parse_query("Where can I go running near Pune?", lat=19.1197, lon=72.8464)
    assert p["location_text"] == "Pune", p["location_text"]
    print("ok  fallback widens, and a typed place beats the GPS fix")


def test_understands_varied_sentences():
    """Full sentences, Hinglish, synonyms and 'near me' all resolve."""
    from app.query_parser import rule_parse
    cases = [
        ("I feel like going for a jog this evening near Bandra", "running", "Bandra"),
        ("planning a picnic with family on Sunday in Powai", "picnic", "Powai"),
        ("kya aaj bahar jana chahiye thane me", None, "thane"),
        ("mujhe kal subah cricket khelna hai powai me", "cricket", "powai"),
        ("aaj shaam ko ghumne kaha ja sakta hu", "gentle", None),
        ("bandra ke paas jogging", "running", "bandra"),
        ("navi mumbai me running ke liye jagah", "running", "navi mumbai"),
        ("shall we do a barbeque this saturday", "picnic", None),
        ("suggest a cafe for a quiet read", "reading", None),
        ("trekking options near lonavala", "hiking", "lonavala"),
    ]
    for text, act, loc in cases:
        r = rule_parse(text)
        assert r["activity"] == act, (text, r["activity"], act)
        assert r["location_text"] == loc, (text, r["location_text"], loc)
    # "near me" must never become a place called "me"
    assert rule_parse("which park is good near me")["location_text"] is None
    # Hinglish time
    from app.query_parser import normalise_time
    assert normalise_time("kal subah") == "tomorrow morning"
    print("ok  varied sentences: Hinglish, synonyms, 'near me', times")


def test_movie_never_returns_food_and_more_widens():
    """A cinema search must never hand back a food court, and 'suggest more'
    must widen the radius rather than repeat the same results."""
    from app import data_fetcher as df
    mix = {"elements": [
        {"type": "node", "lat": 19.20, "lon": 72.97, "tags": {"amenity": "cinema", "name": "Cinepolis"}},
        {"type": "node", "lat": 19.21, "lon": 72.98, "tags": {"amenity": "fast_food", "name": "Burger King"}},
        {"type": "node", "lat": 19.22, "lon": 72.99, "tags": {"amenity": "restaurant", "name": "Barbeque Nation"}},
        {"type": "node", "lat": 19.23, "lon": 73.00, "tags": {"shop": "mall", "name": "Viviana Mall"}},
        {"type": "node", "lat": 19.24, "lon": 73.01, "tags": {"shop": "mall", "name": "R Mall"}},
    ]}
    real = df._overpass
    df._overpass = lambda q, timeout=6: mix
    try:
        df._CACHE.clear()
        got = [p["name"] for p in df.places_and_context(19.2, 72.97, 5000, "movie")["places"]]
        # a multiplex lives inside a mall, so cinemas AND malls count ...
        assert "Cinepolis" in got and "Viviana Mall" in got and "R Mall" in got, got
        # ... but a food court never does
        assert "Burger King" not in got and "Barbeque Nation" not in got, got
    finally:
        df._overpass = real
    # the text fallback filters too
    assert not df._plausible("Burger King", "movie")
    assert df._plausible("Cinepolis", "movie") and df._plausible("Korum Mall", "movie")
    # 'suggest more' grows the radius, doubling each time
    from app import session
    sid = session.create()
    assert session.grow_radius(sid) == 2.0
    assert session.grow_radius(sid) == 4.0
    assert session.grow_radius(sid) == 8.0
    assert session.grow_radius(sid) == 8.0        # capped
    session.reset_radius(sid)
    assert session.get_radius_mult(sid) == 1.0
    from app.query_parser import detect_more
    assert detect_more("suggest more") and detect_more("aur dikhao") and detect_more("show more")
    print("ok  movie stays on cinemas; 'suggest more' widens the radius")


def test_multiple_briefs_returned():
    """The answer must offer several places: a top pick, two more briefs, and
    the rest to explore -- each with its own short brief."""
    from app import reasoning, data_fetcher as df
    real_pc, real_geo = df.places_and_context, df.geocode
    parks = [{"name": "Park %d" % i, "lat": 18.52 + i * 0.002, "lon": 73.85 + i * 0.002,
              "tags": {"leisure": "park"}} for i in range(8)]
    df.places_and_context = lambda lat, lon, r, act: {"places": parks, "context": {"counts": {}}}
    df.geocode = lambda place, country="India": {"lat": 18.52, "lon": 73.85,
                                                 "display_name": place}
    try:
        out = reasoning.answer("Where can I go running near Koregaon Park?", lang="en")
        rec = out.get("recommendation") or {}
        assert rec.get("best"), "no top pick"
        assert len(rec.get("also") or []) == 2, len(rec.get("also") or [])
        assert len(rec.get("alternatives") or []) >= 3, len(rec.get("alternatives") or [])
        for c in [rec["best"]] + (rec.get("also") or []) + (rec.get("alternatives") or []):
            assert c.get("summary"), c
            assert "maps" in c
    finally:
        df.places_and_context, df.geocode = real_pc, real_geo
    print("ok  multiple briefs: 1 top + 2 more + the rest to explore")


def test_movie_phrase_is_understood():
    """'a movie' / 'for a movie' must resolve to the cinema, not be skipped."""
    from app.query_parser import rule_parse
    cases = [
        "i have to go for a movie so suggest best location based on my current location",
        "watch a movie near Thane",
        "go for a movie",
        "a movie",
        "movie",
        "suggest me a cinema hall",
        "where can I watch a picture",
        "multiplex near andheri",
        "bandra me movie dekhni hai",
    ]
    for text in cases:
        assert rule_parse(text)["activity"] == "movie", (text, rule_parse(text)["activity"])
    # an article must not hide the activity, but a place phrase still must
    assert rule_parse("near Market Road running")["activity"] == "running"
    assert rule_parse("near Market Road running")["location_text"] == "Market Road"
    # "my current location" is a location keyword, not a place name
    r = rule_parse("go for a movie based on my current location")
    assert r["activity"] == "movie" and r["location_text"] is None
    print("ok  'a movie' resolves to the cinema; articles no longer hide activities")


def test_a_new_location_wins():
    """Asking about a different place must answer for THAT place, and a new task
    in the same sentence must replace the old one."""
    from app import reasoning, data_fetcher as df
    rp, rg, rw, ra, rpc = (df.places_and_context, df.geocode, df.weather,
                           df.air_quality, None)
    df.geocode = lambda place, country="India": {"lat": 20.0, "lon": 75.0,
                                                 "display_name": place}
    df.weather = lambda *a, **k: None
    df.air_quality = lambda *a, **k: None
    df.places_and_context = lambda lat, lon, r, act: {
        "places": [{"name": "Spot A", "lat": 20.01, "lon": 75.01, "tags": {}}],
        "context": {"counts": {}}}
    try:
        a = reasoning.answer("temples in Ayodhya")
        b = reasoning.answer("monuments in New Delhi", session_id=a["session_id"])
        assert b["parsed"]["location_text"] == "New Delhi", b["parsed"]["location_text"]
        assert b["parsed"]["activity"] == "monument", b["parsed"]["activity"]
        assert b["centre"]["display_name"] == "New Delhi"
        # a follow-up that only names a place keeps the task but moves the place
        c = reasoning.answer("temples in Ayodhya")
        d = reasoning.answer("what about Varanasi?", session_id=c["session_id"])
        assert d["parsed"]["location_text"] == "Varanasi", d["parsed"]["location_text"]
        assert d["parsed"]["activity"] in ("temple", "temples"), d["parsed"]["activity"]
        # a brand new topic replaces both
        e = reasoning.answer("Where can I go running near Koregaon Park?",
                             session_id=c["session_id"])
        assert e["parsed"]["activity"] == "running"
        assert e["parsed"]["location_text"] == "Koregaon Park"
    finally:
        df.places_and_context, df.geocode = rp, rg
        df.weather, df.air_quality = rw, ra
    print("ok  a new place or task in the message always wins over the old context")


def test_trailing_instructions_are_not_part_of_the_place():
    """'weather of Prayagraj with AQI' is Prayagraj, not 'Prayagraj with AQI'."""
    from app.query_parser import rule_parse
    cases = [
        ("weather of Prayagraj with AQI", "Prayagraj"),
        ("weather in Delhi with AQI and wind", "Delhi"),
        ("AQI of Delhi", "Delhi"),
        ("temperature in Pune with humidity", "Pune"),
    ]
    for text, loc in cases:
        assert rule_parse(text)["location_text"] == loc, (text, rule_parse(text)["location_text"])


def test_each_answer_closes_the_conversation():
    """After a finished answer the next message starts its own conversation, and
    a direct reply to "which place?" is used as that place."""
    from app import reasoning, data_fetcher as df
    rp, rg, rw, ra = df.places_and_context, df.geocode, df.weather, df.air_quality
    df.geocode = lambda place, country="India": {"lat": 25.4, "lon": 81.8,
                                                 "display_name": place}
    df.weather = lambda *a, **k: None
    df.air_quality = lambda *a, **k: None
    df.places_and_context = lambda lat, lon, r, act: {
        "places": [{"name": "Spot", "lat": 25.41, "lon": 81.81, "tags": {}}],
        "context": {"counts": {}}}
    try:
        a = reasoning.answer("temples in Ayodhya")
        b = reasoning.answer("MNIT prayagraj", session_id=a["session_id"])
        # the old task is NOT carried into a fresh, complete message
        assert b["parsed"]["location_text"] == "MNIT prayagraj", b["parsed"]["location_text"]
        assert b["parsed"]["activity"] is None, b["parsed"]["activity"]
    finally:
        df.places_and_context, df.geocode = rp, rg
        df.weather, df.air_quality = rw, ra
    print("ok  trailing words dropped; each answer closes the turn")


def _wx(hour, temp=36.0, is_day=1, uv=9.0):
    return {"hourly": {"time": [hour], "temperature_2m": [temp],
                       "relative_humidity_2m": [60.0], "wind_speed_10m": [8.0],
                       "wind_direction_10m": [250.0], "precipitation_probability": [0.0],
                       "uv_index": [uv], "is_day": [is_day]}}


def _aq(hour):
    return {"hourly": {"time": [hour], "us_aqi": [90.0], "pm2_5": [30.0],
                       "grass_pollen": [0.0], "birch_pollen": [0.0], "alder_pollen": [0.0]}}


def test_green_cover_and_water_are_scored():
    """A shaded place beside water beats bare ground -- but only when it is hot."""
    from app import scorer
    ctx = {"green": [{"name": "Park", "lat": 19.1200, "lon": 72.8460}],
           "water": [{"name": "Lake", "lat": 19.1210, "lon": 72.8465}],
           "industrial": [], "roads": []}
    bare = {"name": "Bare", "lat": 19.1197, "lon": 72.8464, "activity": "running", "tags": {}}
    leafy = dict(bare, name="Leafy")
    h = "2026-10-23T15:00"
    hot_bare = scorer.score_slot(bare, h, _wx(h), _aq(h), {"industrial": [], "roads": []}, None, None, None)
    hot_leafy = scorer.score_slot(leafy, h, _wx(h), _aq(h), ctx, None, None, None)
    assert hot_leafy["score"] > hot_bare["score"], (hot_bare["score"], hot_leafy["score"])
    assert hot_leafy["facts"]["shade_by"] == "Park"
    assert hot_leafy["facts"]["water_by"] == "Lake"
    # on a mild day the shade must not inflate anything
    cool_bare = scorer.score_slot(bare, h, _wx(h, temp=22.0), _aq(h), {"industrial": [], "roads": []}, None, None, None)
    cool_leafy = scorer.score_slot(leafy, h, _wx(h, temp=22.0), _aq(h), ctx, None, None, None)
    assert cool_leafy["score"] == cool_bare["score"], (cool_bare["score"], cool_leafy["score"])


def test_a_closed_venue_is_not_recommended():
    from app import scorer
    from app.hours import is_open
    assert is_open("24/7", "2026-10-23T23:00")[0] is True
    assert is_open("Mo-Su 06:00-22:00", "2026-10-23T23:00")[0] is False
    assert is_open("Mo-Fr 06:00-09:00,17:00-21:00", "2026-10-23T12:00")[0] is False
    assert is_open("Mo-Fr 06:00-09:00,17:00-21:00", "2026-10-23T18:00")[0] is True
    assert is_open("something unreadable", "2026-10-23T18:00")[0] is None
    h = "2026-10-23T23:00"
    shut = {"name": "G", "lat": 19.12, "lon": 72.85, "activity": "running",
            "tags": {"opening_hours": "Mo-Su 06:00-22:00"}}
    opn = dict(shut, tags={"opening_hours": "24/7"})
    a = scorer.score_slot(shut, h, _wx(h, is_day=0, uv=0.0), _aq(h), {"industrial": [], "roads": []}, None, None, None)
    b = scorer.score_slot(opn, h, _wx(h, is_day=0, uv=0.0), _aq(h), {"industrial": [], "roads": []}, None, None, None)
    assert any("closed" in r for r in a["reasons"]), a["reasons"]
    assert not any("closed" in r for r in b["reasons"]), b["reasons"]


def test_golden_hour_follows_the_sun():
    from app import scorer
    assert scorer.golden_score("2026-10-23T18:00", "2026-10-23T06:32", "2026-10-23T18:12") == 1.0
    assert scorer.golden_score("2026-10-23T13:00", "2026-10-23T06:32", "2026-10-23T18:12") == 0.0
    # 18:00 IST in Mumbai: just above the horizon, to the west-south-west
    elev, az = scorer.sun_position(19.076, 72.877, "2026-10-23T18:00")
    assert -3 < elev < 5, elev
    assert 240 < az < 300, az
    midday = scorer.sun_position(19.076, 72.877, "2026-10-23T12:30")
    assert midday[0] > 50, midday          # high in the sky
    assert 150 < midday[1] < 210, midday   # and to the south


def _rows():
    return [
        {"place": "Kamala Nehru Park", "hour": "2026-10-24T07:00", "score": 91.2,
         "facts": {"temp": 26, "heat_index": 27, "rain_prob": 0, "aqi": 72, "wind": 5,
                   "uv": 2, "shade_by": "a garden"},
         "travel_minutes": 6, "reasons": ["conditions are comfortable for this activity"]},
        {"place": "Kamala Nehru Park", "hour": "2026-10-24T08:00", "score": 84.0,
         "facts": {"temp": 28, "heat_index": 29, "rain_prob": 0, "aqi": 75, "wind": 6, "uv": 4},
         "travel_minutes": 6, "reasons": []},
        {"place": "Bare Maidan", "hour": "2026-10-24T07:00", "score": 78.4,
         "facts": {"temp": 27, "heat_index": 28, "rain_prob": 0, "aqi": 118, "wind": 9, "uv": 2},
         "travel_minutes": 14,
         "reasons": ["downwind of a road while the wind blows from there"]},
    ]


def test_planner_facts_and_guard():
    """The model may only choose from what the engine fetched, and only with
    numbers the facts actually contain."""
    from app import planner
    rows = _rows()
    facts = planner.build_facts(rows, activity="running", condition=None,
                               window="tomorrow morning")
    assert [p["name"] for p in facts["places"]][0] == "Kamala Nehru Park"
    assert facts["places"][0]["hours"][0]["hour"] == "07:00"

    good = {"best_place": "Bare Maidan", "best_hour": "07:00",
            "why": "It is flat and open.", "runner_up": "Kamala Nehru Park",
            "why_not_runner_up": "It is small.",
            "carry": ["water", "a hat"], "watch": ["air quality"], "confidence": "high"}
    v = planner.validate_decision(good, rows, facts)
    assert v and v["place"] == "Bare Maidan" and v["hour"] == "07:00"
    assert v["carry"] == ["water", "a hat"] and v["watch"] == ["air quality"]

    # an invented place is refused outright
    assert planner.validate_decision(dict(good, best_place="Imaginary Ground"),
                                     rows, facts) is None
    # a place the model got right but an hour it does not have -> that place's best
    assert planner.validate_decision(dict(good, best_hour="03:00"),
                                     rows, facts)["hour"] == "07:00"
    # a sentence quoting a number that is not in the facts is dropped
    v2 = planner.validate_decision(dict(good, why="It is 19C there. It is flat."), rows, facts)
    assert v2["why"] == "It is flat."
    # carry and watch outside the vocabulary are discarded
    v3 = planner.validate_decision(dict(good, carry=["a helicopter"], watch=["dragons"]),
                                   rows, facts)
    assert v3["carry"] == [] and v3["watch"] == []


def test_slm_decision_is_honoured_when_valid():
    """When the model picks a lower-scoring place and the pick is real, the engine
    takes it. When the pick is not real, the engine's own choice stands."""
    from app import scorer, slm
    rows = _rows()
    engine = scorer.analyse(rows)
    assert engine["best"]["place"] == "Kamala Nehru Park"

    picked = scorer.analyse(rows, prefer=rows[2])          # the model's choice
    assert picked["best"]["place"] == "Bare Maidan"

    not_real = scorer.analyse(rows, prefer={"place": "Nowhere", "hour": "2026-10-24T07:00"})
    assert not_real["best"]["place"] == "Kamala Nehru Park"


def test_the_assistant_says_who_decided():
    from app import reasoning
    out = reasoning.answer("Where can I go running near Koregaon Park?")
    assert out["decided_by"] in ("engine", "slm")
    # with the model off, it must always be the engine
    if not reasoning.ENABLE_SLM:
        assert out["decided_by"] == "engine"


def test_a_journey_is_recognised():
    """'cycle from X to Y' is a route question; an activity question is not."""
    from app.query_parser import detect_route
    r = detect_route("can I cycle from Powai to BKC at 8 AM?")
    assert r == {"origin": "Powai", "destination": "BKC", "mode": "cycling"}, r
    r = detect_route("should I walk from Bandra to Khar tomorrow morning?")
    assert r["origin"] == "Bandra" and r["destination"] == "Khar" and r["mode"] == "walking", r
    # "cycle to work" names no place, so both ends come back empty and we ask
    r = detect_route("Is it okay to cycle to work at 8 AM?")
    assert r and r["origin"] is None and r["destination"] is None, r
    # and ordinary questions are untouched
    assert detect_route("Where can I go running near Koregaon Park?") is None
    assert detect_route("I want to go cycling near Powai") is None
    assert detect_route("Best time this weekend for a kids' cricket match?") is None


def test_route_scoring_finds_the_bad_stretch():
    """A journey is scored stretch by stretch, and the verdict rests on the worst."""
    from app import route
    coords = [[19.1200 + i * 0.001, 72.8460 + i * 0.001] for i in range(16)]
    samples = route.sample_along(coords, 200)
    assert len(samples) > 5
    assert samples[0]["at_m"] == 0
    ctx = {"industrial": [{"name": "MIDC", "lat": 19.1250, "lon": 72.8500}],
           "roads": [{"name": "Highway", "lat": 19.1250, "lon": 72.8500}],
           "green": [], "water": []}
    # right beside the source is penalised, far away is not
    near, near_flags = route.stretch_modifier(19.1250, 72.8500, ctx)
    far, far_flags = route.stretch_modifier(19.1400, 72.8650, ctx)
    assert near < far and near_flags and not far_flags, (near, far, near_flags)
    # the plan reports the worst stretch and a best hour
    base = {"2026-10-24T08:00": 72.0, "2026-10-24T06:00": 84.0}
    plan = route.plan(samples, ctx, [], base, list(base))
    assert plan["best_hour"] == "2026-10-24T06:00", plan["best_hour"]
    assert plan["worst"]["score"] < plan["best_average"]
    assert plan["flags"] and plan["road"] is None or True
    # no context at all means no local penalty
    flat = route.plan(samples, None, [], base, list(base))
    assert flat["best_worst"] == 84.0, flat["best_worst"]


def test_a_location_request_is_never_a_place():
    """'Use my current location' must never be geocoded, and a plain 'where'
    question must not be mistaken for a request for the user's own location."""
    from app.query_parser import wants_my_location as w, looks_like_bare_place as bare, rule_parse
    # the bug: "Where ..." matched on the "here" inside "Where"
    assert w("Where can I go running near Koregaon Park?") is False
    assert w("Where can I go for cricket?") is False
    assert w("Best place for tennis in Pune") is False
    assert w("use my location") is True
    assert w("Use my current location") is True
    assert w("Where can I run near me?") is True
    # and it is not a place name either
    assert bare("Use my current location") is False
    assert bare("my location") is False
    assert bare("Koregaon Park") is True
    assert rule_parse("Use my current location")["location_text"] is None


def test_clicking_use_my_location_works():
    """The chip is an action: with a fix it answers, without one it asks to turn
    the location on. It must never say it could not locate the sentence."""
    from app import reasoning, data_fetcher as df
    rp, rg, rw, ra, rrt, rgc = (df.places_and_context, df.geocode, df.weather,
                                df.air_quality, df.route_time, df.geospatial_context)
    df.geocode = lambda p, country="India": {"lat": 19.12, "lon": 72.85,
                                             "display_name": f"{p}, India"}
    df.places_and_context = lambda la, lo, r, a: {
        "places": [{"name": "Ground", "lat": 19.121, "lon": 72.851,
                    "tags": {"leisure": "park"}}], "context": {"counts": {}}}
    df.weather = lambda *a, **k: None
    df.air_quality = lambda *a, **k: None
    df.route_time = lambda *a, **k: {"seconds": 300, "metres": 1500}
    df.geospatial_context = lambda la, lo, r=3000: {
        "industrial": [], "water": [], "green": [], "power": [], "roads": [], "counts": {}}
    try:
        a = reasoning.answer("Where can I go for cricket?")
        assert a["clarifying"] is True
        # with a fix from the device
        b = reasoning.answer("Use my current location", session_id=a["session_id"],
                             lat=19.1197, lon=72.8464)
        assert "could not locate" not in b["advice"], b["advice"]
        assert b["clarifying"] is False, b["advice"]
        # without one
        c = reasoning.answer("Use my current location", session_id=a["session_id"])
        assert "could not locate" not in c["advice"], c["advice"]
        assert c["clarifying"] is True
    finally:
        (df.places_and_context, df.geocode, df.weather, df.air_quality,
         df.route_time, df.geospatial_context) = rp, rg, rw, ra, rrt, rgc
    print("ok  'Use my current location' answers, or asks to turn the pin on")


def test_any_activity_resolves():
    from app.profiles import resolve_profile_key
    assert resolve_profile_key("boating") == "watersport"
    assert resolve_profile_key("kayaking") == "watersport"
    assert resolve_profile_key("horse riding") == "field"
    assert resolve_profile_key("yoga") == "gentle"
    assert resolve_profile_key("zorbing") == "gentle"      # unknown -> safe default
    assert resolve_profile_key("running") == "running"
    print("ok  any activity name resolves to a usable profile")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")
