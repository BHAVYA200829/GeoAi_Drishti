"""
Drishti orchestrator.

Wires the whole pipeline together:

    question (possibly across turns)
      -> session merge                           [chat memory]
      -> parse (SLM + rules)                     [NLU]
      -> geocode centre                          [Nominatim]
      -> fetch weather, AQ, places, context      [parallel]
      -> travel time for the top places          [OSRM, parallel]
      -> build place x time comfort surface       [scorer]
      -> analyse (best / golden window / flips)  [scorer]
      -> narrate advice + provenance ledger      [SLM or deterministic]
      -> attach Google Maps links + how-to-reach  [maps]

Nothing here invents a fact. Missing data becomes a recorded data gap.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from . import data_fetcher as df
from . import maps, session
from .config import AVOID_SCORE, ENABLE_SLM, GOOD_SCORE, MAX_CANDIDATES
from .i18n import t as _t
from .l10n import localize, localize_label
from .slm import model_name
from .narration import (best_day_advice, build_advice, carry_and_watch,
                        route_advice,
                        conditions_advice, friendly_summary, go_out_advice)
from .query_parser import (detect_route,
                          activity_label_from_text, clean_activity, clean_place,
                          detect_keywords, detect_more, looks_like_bare_place,
                          normalise_time,
                          wants_my_location,
                          ends_in_ing, is_lexicon_activity, looks_like_activity,
                          looks_like_place, parse_query, rule_parse)
from .scorer import analyse, build_surface, score_slot
from .timeparse import parse_time_window
from .tz import IST

QUICK_LOCATION_REPLIES = ["Use my current location", "Mumbai", "Bengaluru", "Pune"]

GREETING = ("Namaste! I'm Drishti. Tell me an activity and a place in India - for example "
            "\"boating near Powai\" or \"running in Bengaluru tomorrow evening\" - and I'll find "
            "the best spot and hour, with directions and what to carry.")

_SMALL_WORDS = {"hi", "hii", "hiii", "hey", "helo", "hello", "yo", "namaste", "thanks",
                "thank", "you", "ok", "okay", "cool", "bye", "sup", "help", "there",
                "good", "morning", "evening", "afternoon", "sir", "please", "plz", "?"}


def _classify_bare(text: str) -> tuple[str, str]:
    """A standalone short message: is it an objective or a place name?

    Objective words are recognised from the lexicon (fast, no network). Anything
    else is geocoded: if it resolves to a place whose name matches, it is a
    location; otherwise we treat it as an objective so nothing is ever ignored.
    """
    if is_lexicon_activity(text) or ends_in_ing(text):
        return "activity", clean_activity(text)
    # a short message with no activity word and no weather word is a place name,
    # even if geocoding is unavailable right now
    if looks_like_bare_place(text):
        return "location", clean_place(text)
    g = df.geocode(text)
    if g:
        first = (g.get("display_name") or "").split(",")[0].strip().lower()
        q = re.sub(r"[^a-z ]", " ", text.lower()).strip()
        if first and (first == q or first.startswith(q + " ") or q.startswith(first)):
            return "location", clean_place(text)
    return "activity", clean_activity(text)


def _is_smalltalk(text: str) -> bool:
    """A greeting / filler message that carries no task or place."""
    t = re.sub(r"[^a-z ]", " ", text.lower())
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return True
    words = t.split()
    if len(words) <= 3 and all(w in _SMALL_WORDS for w in words):
        return True
    return False
from .profiles import FEATURED_ACTIVITIES  # noqa: E402

QUICK_ACTIVITY_REPLIES = [label for _k, label in FEATURED_ACTIVITIES]


def _now_iso() -> str:
    return datetime.now(IST).isoformat(timespec="seconds")


def _go_no_go(score: float) -> str:
    if score >= GOOD_SCORE:
        return "GO"
    if score > AVOID_SCORE:
        return "CAUTION"
    return "AVOID"


def _out(question, parsed, sid, *, recommendation=None, advice="", surface=None,
         evidence=None, provenance=None, gaps=None, model="", clarifying=False,
         quick_replies=None, centre=None, best_day=None, keywords=None, lang="en",
         more_available=False, decided_by="engine", slm_confidence=None,
         route=None, plan=None):
    return {
        "assistant": "Drishti",
        "query": question,
        "session_id": sid,
        "parsed": parsed,
        "recommendation": recommendation,
        "advice": advice,
        "surface": surface,
        "evidence": evidence or {},
        "provenance": provenance or [],
        "sources": df.SOURCES,
        "data_gaps": gaps or [],
        "model": model,
        "clarifying": clarifying,
        "quick_replies": quick_replies or [],
        "centre": centre or {},
        "best_day": best_day,
        "keywords": keywords or [],
        "lang": lang,
        "more_available": more_available,
        "decided_by": decided_by,
        "slm_confidence": slm_confidence,
        "route": route,
        "route_plan": plan,
    }


def _cand(r: dict, detail: dict, centre: dict, activity: str,
          condition: str | None = None, lang: str = "en") -> dict:
    d = detail.get(r["place"]) or {}
    m = maps.how_to_reach(activity, d.get("metres"), d.get("minutes"),
                          centre["lat"], centre["lon"], r["lat"], r["lon"])
    reasons = [x for x in (r.get("reasons") or []) if x]
    carry, watch = carry_and_watch(r, activity, condition, lang)
    return {
        "name": r["place"], "lat": r["lat"], "lon": r["lon"], "score": r["score"],
        "hour": r["hour"], "travel_minutes": r.get("travel_minutes"),
        "reasons": reasons, "facts": r.get("facts", {}), "maps": m,
        # a short brief, so "more locations" can open in place
        "summary": friendly_summary(r.get("facts", {}) or {}, reasons, activity),
        "carry": carry, "watch": watch,
    }


def _recommendation_block(res: dict, centre: dict, activity: str, detail: dict,
                          condition: str | None = None, more: bool = False,
                          lang: str = "en") -> dict:
    best = res.get("best")
    if not best:
        return {"best": None}

    flips = res.get("flips", [])
    improving = [f for f in flips if f["direction"] == "improves"]
    flip_note = None
    if improving:
        f = improving[0]
        flip_note = (f"{f['place']} is poor early but improves around {f['at'][11:16]} "
                     f"({f['reason']}).")
    elif flips:
        f = flips[0]
        flip_note = f"{f['place']} {f['direction']} around {f['at'][11:16]} ({f['reason']})."

    # a warm, human one-liner
    reasons = [x for x in (best.get("reasons") or []) if x]
    summary = friendly_summary(best.get("facts", {}) or {}, reasons, activity)

    n_alt = 14 if more else 10
    carry, watch = carry_and_watch(best, activity, condition, lang)
    others = [_cand(x, detail, centre, activity, condition, lang)
              for x in res.get("alternatives", [])[:n_alt]]
    # two more get a full brief, the rest are there to explore
    also, rest = others[:2], others[2:]
    return {
        "best": _cand(best, detail, centre, activity, condition, lang),
        "also": also,
        "golden_window": res.get("golden_window"),
        "avoid": res.get("avoid"),
        "alternatives": rest,
        "flips": flips,
        "flip_note": flip_note,
        "go_no_go": _go_no_go(best["score"]),
        "summary": summary,
        "carry": carry,
        "watch": watch,
    }


def _provenance(best: dict, places: list, centre: dict, gaps: list) -> list[dict]:
    prov = []
    now = _now_iso()
    f = best.get("facts", {})
    for claim, src, val in [
        (f"temperature at {best['hour'][11:16]}", "Open-Meteo", f"{f['temp']}°C" if f.get("temp") is not None else None),
        ("chance of rain", "Open-Meteo", f"{f['rain_prob']}%" if f.get("rain_prob") is not None else None),
        ("wind", "Open-Meteo", f"{f['wind']} km/h" if f.get("wind") is not None else None),
        ("air quality (US AQI)", "Open-Meteo Air-Quality", f"{f['aqi']}" if f.get("aqi") is not None else None),
    ]:
        if val is not None:
            prov.append({"claim": claim, "source": src, "value": val, "fetched_at": now})
    prov.append({"claim": f"{len(places)} candidate places", "source": "OpenStreetMap via Overpass",
                 "value": [p["name"] for p in places][:6], "fetched_at": now})
    if centre.get("display_name"):
        prov.append({"claim": "geocoded search centre", "source": "Nominatim",
                     "value": centre["display_name"], "fetched_at": now})
    if best.get("travel_minutes") is not None:
        prov.append({"claim": "travel time to best place", "source": "OSRM",
                     "value": f"{best['travel_minutes']:.0f} min", "fetched_at": now})
    for g in gaps:
        prov.append({"claim": "data gap", "source": "system", "value": g, "fetched_at": now})
    return prov


ROUTE_PROFILE = {"cycling": "cycling", "walking": "gentle", "driving": "event"}
ROUTE_OSRM = {"cycling": "cycling", "walking": "walking", "driving": "driving"}


def _route_answer(route: dict, question: str, sid: str,
                  lat: float | None, lon: float | None, lang: str) -> dict:
    """
    Answer a journey: route it, walk along it, score every stretch.

    The weather is fetched once for the route, because it does not change across a
    city within the hour; what changes from stretch to stretch is what is beside
    the road, and that comes from OpenStreetMap.
    """
    from . import route as R

    mode = route.get("mode") or "cycling"
    parsed = {"intent": "route", "activity": ROUTE_PROFILE.get(mode),
              "location_text": None, "radius_km": None, "time_expression": None,
              "persona": {"age": None, "condition": None},
              "needs_clarification": False, "clarification_question": None,
              "parser": "rules", "route": {"mode": mode}}

    dest_text = route.get("destination")
    origin_text = route.get("origin")

    # nothing to find yet: ask, rather than guess
    if not dest_text:
        parsed["needs_clarification"] = True
        parsed["clarification_question"] = (
            f"Where are you {mode} to? Give me the start and the end, "
            f"for example 'cycle from Powai to BKC'.")
        return _out(question, parsed, sid,
                    advice=parsed["clarification_question"], clarifying=True,
                    quick_replies=["Powai to BKC", "Bandra to Khar",
                                   "Use my current location"],
                    decided_by="engine")

    if not origin_text and lat is None:
        parsed["needs_clarification"] = True
        parsed["clarification_question"] = (
            "Where are you starting from? Name it, or turn on your location "
            "and I will start from where you are.")
        return _out(question, parsed, sid,
                    advice=parsed["clarification_question"], clarifying=True,
                    quick_replies=["Powai", "Bandra", "Use my current location"],
                    decided_by="engine")

    gaps: list[str] = []
    if origin_text:
        o = df.geocode(origin_text)
        if not o:
            gaps.append(f"I could not locate '{origin_text}'")
            parsed["needs_clarification"] = True
            parsed["clarification_question"] = (
                f"I could not locate '{origin_text}'. Could you give a nearby "
                f"landmark, city or district?")
            return _out(question, parsed, sid, advice=parsed["clarification_question"],
                        clarifying=True, decided_by="engine")
        olat, olon = o["lat"], o["lon"]
        origin_name = o.get("display_name") or origin_text
    else:
        olat, olon = lat, lon
        origin_name = "your location"

    d = df.geocode(dest_text)
    if not d:
        parsed["needs_clarification"] = True
        parsed["clarification_question"] = (
            f"I could not locate '{dest_text}'. Could you give a nearby landmark, "
            f"city or district?")
        return _out(question, parsed, sid, advice=parsed["clarification_question"],
                    clarifying=True, decided_by="engine")
    dlat, dlon = d["lat"], d["lon"]
    dest_name = d.get("display_name") or dest_text

    geo = df.route_geometry(olat, olon, dlat, dlon, profile=ROUTE_OSRM.get(mode, "driving"))
    if not geo or not geo.get("main"):
        gaps.append("the routing service did not return a route")
        parsed["needs_clarification"] = True
        parsed["clarification_question"] = (
            "I could not work out a route between those two places right now. "
            "Please try again shortly.")
        return _out(question, parsed, sid, advice=parsed["clarification_question"],
                    clarifying=True, decided_by="engine")

    main = geo["main"]
    samples = R.sample_along(main.get("coords") or [])
    if not samples:
        gaps.append("the route came back without a shape")
        parsed["needs_clarification"] = True
        parsed["clarification_question"] = (
            "I could not read the route shape right now. Please try again shortly.")
        return _out(question, parsed, sid, advice=parsed["clarification_question"],
                    clarifying=True, decided_by="engine")

    mid_lat = sum(s["lat"] for s in samples) / len(samples)
    mid_lon = sum(s["lon"] for s in samples) / len(samples)
    span_m = max(1500.0, (main.get("metres") or 3000) / 2.0 + 1500.0)

    jobs = {
        "wx": lambda _x: df.weather(mid_lat, mid_lon),
        "aq": lambda _x: df.air_quality(mid_lat, mid_lon),
        "ctx": lambda _x: df.geospatial_context(mid_lat, mid_lon, int(span_m)),
    }
    got = df.parallel(lambda k: jobs[k](k), list(jobs), max_workers=3)
    wx, aq, ctx = got[0], got[1], got[2]
    ctx = ctx or {"industrial": [], "water": [], "green": [], "roads": []}
    if not wx:
        gaps.append("live weather was unavailable, so the score is places-only")
    if not aq:
        gaps.append("air-quality data was unavailable, so the score is weather-only")

    hours_all = (wx or {}).get("hourly", {}).get("time", [])
    if not hours_all:
        hours_all = [(datetime.now(IST) + timedelta(hours=h)).strftime("%Y-%m-%dT%H:00")
                     for h in range(0, 13)]
    window = parse_time_window(normalise_time(parsed.get("time_expression")), hours_all)
    hours = window["hours"] or hours_all[:12]
    hours = [h for h in hours if h >= hours_all[0]][:12] or hours_all[:12]

    daily = (wx or {}).get("daily", {})
    sunrise = (daily.get("sunrise") or [None])[0]
    sunset = (daily.get("sunset") or [None])[0]

    virtual = {"name": "the route", "lat": mid_lat, "lon": mid_lon,
               "activity": ROUTE_PROFILE.get(mode, "cycling"), "tags": {}}
    base_by_hour = {}
    for h in hours:
        try:
            sc = score_slot(virtual, h, wx or {}, aq or {}, None, None, sunrise, sunset)
            base_by_hour[h] = sc["score"]
        except Exception:  # noqa: BLE001
            continue

    plan = R.plan(samples, ctx, main.get("steps"), base_by_hour, hours)
    route_block = {
        "mode": mode,
        "origin": origin_text, "origin_name": origin_name,
        "destination": dest_text, "destination_name": dest_name,
        "lat": dlat, "lon": dlon,
        "metres": main.get("metres"), "seconds": main.get("seconds"),
        "estimated": bool(main.get("estimated")),
        "maps_url": maps.directions_url(olat, olon, dlat, dlon, ROUTE_OSRM.get(mode, "driving")),
        "alternative": bool(geo.get("alternative")),
    }
    advice = route_advice(route_block, plan, parsed, {}, gaps, lang=lang)

    surface = [{"place": "route", "hour": x["hour"], "score": x["worst"]}
               for x in plan.get("by_hour") or []]
    evidence = {
        "route": {k: route_block[k] for k in
                  ("mode", "origin_name", "destination_name", "metres", "seconds")},
        "stretches": len(plan.get("stretches") or []),
        "worst": plan.get("worst"),
        "geospatial_context_counts": (ctx or {}).get("counts", {}),
        "weather_source": "Open-Meteo",
        "air_quality_source": "Open-Meteo Air-Quality",
    }
    return _out(
        question, parsed, sid, advice=advice, surface=surface, evidence=evidence,
        gaps=gaps, lang=lang, centre={"lat": dlat, "lon": dlon, "display_name": dest_name},
        keywords=detect_keywords(question, parsed) + [{"type": "radius", "value": mode}],
        route=route_block, plan=plan,
        decided_by="engine",
    )


def _slm_decide(rows: list, question: str, activity: str, condition: str | None,
                window: str | None, travel_detail: dict) -> dict | None:
    """
    Hand the scored surface to the small model and take back a validated choice.
    Returns None whenever the model is off, unavailable, or its answer does not
    hold up against the facts -- in which case the engine's own pick stands.
    """
    if not ENABLE_SLM or not rows:
        return None
    try:
        from . import planner, slm
        facts = planner.build_facts(rows, activity=activity, condition=condition,
                                    window=window, now_iso=_now_iso())
        raw = slm.decide(facts, question=question)
        return planner.validate_decision(raw, rows, facts)
    except Exception:  # noqa: BLE001  -- the model must never break the answer
        return None


def _fold_decision(rec: dict, decision: dict) -> None:
    """Let the model's carry / watch list join the engine's own advice."""
    best = rec.get("best") or {}
    if decision.get("carry"):
        have = {x.lower() for x in (best.get("carry") or [])}
        best["carry"] = list(best.get("carry") or []) + [
            x for x in decision["carry"] if x.lower() not in have]
    if decision.get("watch"):
        have = {x.lower() for x in (best.get("watch") or [])}
        best["watch"] = list(best.get("watch") or []) + [
            x for x in decision["watch"] if x.lower() not in have]
    if decision.get("why"):
        best["reasons"] = list(best.get("reasons") or []) + [decision["why"]]


def _localize_candidate(c: dict, lang: str) -> dict:
    if not c:
        return c
    c["summary"] = localize(c.get("summary"), lang)
    c["carry"] = [localize(x, lang) for x in (c.get("carry") or [])]
    c["watch"] = [localize(x, lang) for x in (c.get("watch") or [])]
    c["reasons"] = [localize(x, lang) for x in (c.get("reasons") or [])]
    m = c.get("maps") or {}
    if m.get("text"):
        m["text"] = localize(m["text"], lang)
    return c


def _localize_result(res: dict, lang: str) -> dict:
    """Rewrite every user-facing string in the reply into `lang`."""
    if (lang or "en")[:2] == "en" or not res:
        return res
    res["advice"] = localize(res.get("advice"), lang)
    # the activity word the user typed may sit inside the sentence
    _lab = (res.get("parsed") or {}).get("activity_label")
    if _lab:
        _loc = localize_label(_lab, lang)
        if _loc and _loc != _lab:
            res["advice"] = res["advice"].replace(_lab, _loc)
    res["data_gaps"] = [localize(x, lang) for x in (res.get("data_gaps") or [])]
    rec = res.get("recommendation") or {}
    for key in ("best",):
        _localize_candidate(rec.get(key), lang)
    for key in ("also", "alternatives"):
        for c in (rec.get(key) or []):
            _localize_candidate(c, lang)
    if rec.get("flip_note"):
        rec["flip_note"] = localize(rec["flip_note"], lang)
    av = rec.get("avoid") or {}
    if av:
        av["reasons"] = [localize(x, lang) for x in (av.get("reasons") or [])]
    for pv in (res.get("provenance") or []):
        if pv.get("claim"):
            pv["claim"] = localize(pv["claim"], lang)
    for kw in (res.get("keywords") or []):
        if kw.get("type") == "activity":
            kw["value"] = localize_label(kw["value"], lang)
        elif kw.get("type") == "weather":
            kw["value"] = localize("weather and AQI", lang)
        elif kw.get("type") == "question":
            kw["value"] = localize("go out or not", lang)
        elif kw.get("type") == "time":
            kw["value"] = localize(kw["value"], lang)
        elif kw.get("type") == "location":
            kw["value"] = localize("use my location", lang)
        elif kw.get("type") == "time":
            kw["value"] = localize(kw["value"], lang)
        elif kw.get("type") == "location":
            kw["value"] = localize("use my location", lang)
    parsed = res.get("parsed") or {}
    if parsed.get("activity_label"):
        parsed["activity_label"] = localize_label(parsed["activity_label"], lang)
    if res.get("best_day"):
        for row in res["best_day"].get("ranking") or []:
            row["reasons"] = [localize(x, lang) for x in (row.get("reasons") or [])]
    return res


def _answer_impl(question: str, lat: float | None = None, lon: float | None = None,
           session_id: str | None = None, lang: str = "en") -> dict:
    # --- chat memory: remember the question, merge follow-ups ---
    sid = session.append(session_id, question)
    # a journey question ("cycle from Powai to BKC at 8 AM") is answered on the
    # route, not by picking a venue
    route_q = detect_route(question)
    if route_q:
        return _route_answer(route_q, question, sid, lat, lon, lang)

    fresh = rule_parse(question)
    pending = session.get_pending(sid)        # what we last asked the user for

    # --- a greeting carries no task: never reuse a previous location/task ---
    if (pending is None and not fresh.get("activity") and not fresh.get("location_text")
            and _is_smalltalk(question)):
        session.reset(sid, question)
        parsed = {"intent": "chat", "activity": None, "location_text": None, "radius_km": None,
                  "time_expression": None, "persona": {"age": None, "condition": None},
                  "needs_clarification": True, "clarification_question": None, "parser": "rules"}
        return _out(question, parsed, sid, advice=GREETING, clarifying=True,
                    quick_replies=QUICK_ACTIVITY_REPLIES)

    more = detect_more(question)

    # --- decide what this message is: a reply, a bare objective, or a place ---
    location_hint = activity_hint = None
    bare = False
    if not fresh.get("activity") and not fresh.get("location_text"):
        if pending == "activity" and looks_like_activity(question):
            activity_hint = clean_activity(question)
        elif (pending == "location" and not wants_my_location(question)
              and (looks_like_place(question) or looks_like_bare_place(question))):
            location_hint = clean_place(question)
        elif pending is None and looks_like_activity(question):
            bare = True
            kind, hint = _classify_bare(question)
            if kind == "activity":
                activity_hint = hint
            else:
                location_hint = hint

    # --- carry context forward only for a genuine follow-up ---
    # a message that names BOTH a task and a place is a brand new topic;
    # one that names a place (or a task) may still be a follow-up
    complete_on_its_own = (bool(fresh.get("activity"))
                           and bool(fresh.get("location_text") or lat is not None))
    use_context = ((not bare) and (bool(pending) or bool(fresh.get("activity"))
                                   or bool(fresh.get("location_text")))) or more
    if complete_on_its_own or not use_context:
        session.reset(sid, question)          # new/independent message -> start clean
    combined = session.context_text(sid) or question

    parsed = parse_query(combined, lat=lat, lon=lon,
                         location_hint=location_hint, activity_hint=activity_hint)

    # a direct answer to our own "which place?" question is the place --
    # unless the user asked us to use their own location, which is not a place
    if location_hint and not wants_my_location(question):
        parsed["location_text"] = location_hint
    # whatever the user just typed outranks anything carried over from earlier.
    # "temples in Ayodhya" then "monuments in New Delhi" must answer for New Delhi.
    if fresh.get("location_text") and not wants_my_location(question):
        parsed["location_text"] = fresh["location_text"]
    if wants_my_location(question):
        parsed["location_text"] = None      # the fix comes from the device, not a name
    if fresh.get("activity"):
        parsed["activity"] = fresh["activity"]
    if fresh.get("radius_km"):
        parsed["radius_km"] = fresh["radius_km"]
    if fresh.get("time_expression"):
        parsed["time_expression"] = fresh["time_expression"]

    # remember the resolved activity, and restore it for a genuine follow-up
    if parsed.get("activity"):
        session.set_activity(sid, parsed["activity"])
    elif use_context:
        stored_activity = session.get_activity(sid)
        if stored_activity:
            parsed["activity"] = stored_activity

    # recompute the clarification state now that the activity may be known
    has_act = bool(parsed.get("activity"))
    has_loc = bool(parsed.get("location_text")) or lat is not None
    if parsed.get("intent") == "conditions":
        parsed["needs_clarification"] = not has_loc
    else:
        parsed["needs_clarification"] = not (has_act and has_loc)

    # keep the user's own word for the activity, not our internal family name
    fresh_label = activity_hint or activity_label_from_text(question)
    if fresh_label:
        session.set_activity_label(sid, fresh_label)
    parsed["activity_label"] = (fresh_label or session.get_activity_label(sid)
                                or parsed.get("activity"))
    # never show the user an internal family name like "gentle"
    from .query_parser import FRIENDLY_FAMILY as _FF
    if parsed["activity_label"] in _FF:
        parsed["activity_label"] = _FF[parsed["activity_label"]]
    gaps: list[str] = []

    # --- ask back if a detail is genuinely missing (no guessing, no loop) ---
    if parsed.get("needs_clarification"):
        missing_activity = (not parsed.get("activity")
                            and parsed.get("intent") not in ("conditions", "go_out", "best_day"))
        missing_loc = not parsed.get("location_text") and lat is None
        if missing_activity:
            # the more fundamental detail -- ask for this first
            q = ("What would you like to do? Anything works, such as running, cycling, "
                 "boating, yoga, a picnic or photography.")
            session.set_pending(sid, "activity")
            return _out(question, parsed, sid, advice=q, clarifying=True,
                        quick_replies=QUICK_ACTIVITY_REPLIES,
                        model=parsed.get("parser") or "",
                        keywords=detect_keywords(question, parsed), lang=lang)
        if missing_loc:
            act = parsed.get("activity_label")
            if wants_my_location(question):
                q = ((f"Got it - {act}. ") if act else "") + \
                    "Your location is not on yet, so tap the pin button and I will use it. Or just name a city, neighbourhood or landmark."
            else:
                q = ((f"Got it - {act}. ") if act else "") + \
                    "Which place in India should I look at? A city, a neighbourhood or a landmark works."
            session.set_pending(sid, "location")
            return _out(question, parsed, sid, advice=q, clarifying=True,
                        quick_replies=QUICK_LOCATION_REPLIES,
                        model=parsed.get("parser") or "",
                        keywords=detect_keywords(question, parsed), lang=lang)
        # nothing is actually missing -- carry on and answer

    # --- centre: an explicitly named place ALWAYS beats a GPS fix ---
    if parsed.get("location_text"):
        centre = df.geocode(parsed["location_text"]) or {}
        if not centre:
            return _out(question, parsed, sid,
                        advice=f"I could not locate '{parsed.get('location_text')}'. "
                               "Could you give a nearby landmark, city or district?",
                        clarifying=True, quick_replies=QUICK_LOCATION_REPLIES)
    elif lat is not None and lon is not None:
        # do NOT wait on reverse geocoding here -- it is only a label, so it is
        # fetched in parallel with the weather/places calls further down
        centre = {"lat": lat, "lon": lon, "display_name": "your location",
                  "_needs_name": True}
    else:
        return _out(question, parsed, sid,
                    advice="Which place in India should I look at? "
                           "A landmark, neighbourhood, city or district all work.",
                    clarifying=True, quick_replies=QUICK_LOCATION_REPLIES)

    # --- plain weather / AQI question: no activity needed ---
    if parsed.get("intent") == "conditions":
        if centre.get("_needs_name"):
            centre["display_name"] = df.reverse_geocode(centre["lat"], centre["lon"]) or "your location"
        place_name = parsed.get("location_text") or centre.get("display_name", "this location")
        snap = df.conditions(centre["lat"], centre["lon"], label=place_name)
        advice = conditions_advice(snap, place_name, lang)
        prov = []
        if snap.get("available"):
            prov = [{"claim": "temperature", "source": "Open-Meteo",
                     "value": f"{snap.get('temperature')}°C", "fetched_at": _now_iso()},
                    {"claim": "air quality (US AQI)", "source": "Open-Meteo Air-Quality",
                     "value": snap.get("aqi"), "fetched_at": _now_iso()}]
        return _out(question, parsed, sid, advice=advice,
                    evidence={"centre": centre, "centre_conditions": snap},
                    provenance=prov, centre=centre, lang=lang,
                    keywords=detect_keywords(question, parsed),
                    model=parsed.get("parser") or "")

    # --- "should I go out?" -- a friendly go / no-go, no activity needed ---
    if parsed.get("intent") == "go_out":
        if centre.get("_needs_name"):
            centre["display_name"] = df.reverse_geocode(centre["lat"], centre["lon"]) or "your location"
        place_name = parsed.get("location_text") or centre.get("display_name", "your area")
        snap = df.conditions(centre["lat"], centre["lon"], label=place_name)
        advice = go_out_advice(snap, place_name, lang)
        prov = []
        if snap.get("available"):
            prov = [{"claim": "temperature", "source": "Open-Meteo",
                     "value": "%s C" % snap.get("temperature"), "fetched_at": _now_iso()},
                    {"claim": "air quality (US AQI)", "source": "Open-Meteo Air-Quality",
                     "value": snap.get("aqi"), "fetched_at": _now_iso()}]
        return _out(question, parsed, sid, advice=advice, centre=centre, provenance=prov,
                    evidence={"centre": centre, "centre_conditions": snap},
                    keywords=detect_keywords(question, parsed), lang=lang,
                    model=parsed.get("parser") or "")

    # --- best day this week: rank each of the next 7 days ---
    if parsed.get("intent") == "best_day":
        activity = parsed.get("activity") or "picnic"
        radius_m = int((parsed.get("radius_km") or 5) * 1000)
        if centre.get("_needs_name"):
            centre["display_name"] = df.reverse_geocode(centre["lat"], centre["lon"]) or "your location"
        place_name = parsed.get("location_text") or centre.get("display_name", "here")
        condition_bd = parsed["persona"].get("condition")

        wx7 = df.weather(centre["lat"], centre["lon"], days=7)
        if not wx7:
            return _out(question, parsed, sid,
                        advice="Live weather is unavailable right now, so I cannot compare "
                               "the days this week. Please try again shortly.",
                        centre=centre, gaps=["weather unavailable"])
        aq7 = df.air_quality(centre["lat"], centre["lon"], days=7)
        combo = df.places_and_context(centre["lat"], centre["lon"], radius_m, activity)
        cand = (combo or {}).get("places") or []
        ctx = (combo or {}).get("context") or {"counts": {}}
        for p in cand:
            p["activity"] = activity
        ctx_by_place = {p["name"]: ctx for p in cand}

        hourly = wx7.get("hourly", {})
        hours_all = hourly.get("time", [])
        isday = dict(zip(hours_all, hourly.get("is_day", []) or []))
        daily = wx7.get("daily", {})
        sunrise = (daily.get("sunrise") or [None])[0]
        sunset = (daily.get("sunset") or [None])[0]

        ranking = []
        for d in sorted({h[:10] for h in hours_all}):
            hs = [h for h in hours_all if h.startswith(d) and isday.get(h) == 1]
            if not hs:
                hs = [h for h in hours_all if h.startswith(d) and 6 <= int(h[11:13]) <= 20]
            if not hs or not cand:
                continue
            rows = build_surface(cand, wx7, aq7, ctx_by_place, condition_bd, hs,
                                 sunrise=sunrise, sunset=sunset)
            b = max(rows, key=lambda x: x["score"])
            try:
                wd = datetime.fromisoformat(d).strftime("%A")
            except Exception:
                wd = d
            ranking.append({"date": d, "weekday": wd, "place": b["place"],
                            "hour": b["hour"], "score": b["score"],
                            "reasons": b.get("reasons", [])})
        ranking.sort(key=lambda x: x["score"], reverse=True)

        if not ranking:
            return _out(question, parsed, sid,
                        advice="I could not find places or weather for that area this week. "
                               "Could you give a nearby landmark or city?",
                        centre=centre, clarifying=True, quick_replies=QUICK_LOCATION_REPLIES)

        advice = best_day_advice(ranking, parsed.get("activity_label") or activity, place_name)
        gaps = []
        if not aq7:
            gaps.append("air-quality data was unavailable, so the score is weather-only")
        prov = [{"claim": "7-day forecast", "source": "Open-Meteo", "value": "7 days",
                 "fetched_at": _now_iso()},
                {"claim": "candidate places", "source": "OpenStreetMap via Overpass",
                 "value": [p["name"] for p in cand][:6], "fetched_at": _now_iso()}]
        return _out(question, parsed, sid, advice=advice, centre=centre, gaps=gaps,
                    provenance=prov, best_day={"place": place_name, "ranking": ranking},
                    keywords=detect_keywords(question, parsed), lang=lang,
                    evidence={"centre": centre, "centre_conditions": df.conditions(centre["lat"], centre["lon"])})

    # the answer is complete, so this turn is closed: the next message starts
    # its own conversation unless it clearly refers back to this one
    session.set_pending(sid, None)

    # --- fetch weather, AQ, places, context IN PARALLEL ---
    radius_m = int((parsed.get("radius_km") or 5) * 1000)
    if more:
        # "suggest more" means: look FURTHER OUT, not just list more of the same
        mult = session.grow_radius(sid)
        radius_m = int(radius_m * mult)
        gaps.append("widened the search to about %.0f km" % (radius_m / 1000.0))
    elif not use_context:
        session.reset_radius(sid)
    activity = parsed.get("activity") or "picnic"
    condition = parsed["persona"].get("condition")

    tasks = [
        lambda _: df.weather(centre["lat"], centre["lon"]),
        lambda _: df.air_quality(centre["lat"], centre["lon"]),
        # places + geospatial context come from ONE Overpass round-trip
        lambda _: df.places_and_context(centre["lat"], centre["lon"], radius_m, activity),
        # a nice name for a GPS fix, fetched alongside everything else
        lambda _: (df.reverse_geocode(centre["lat"], centre["lon"])
                   if centre.get("_needs_name") else None),
    ]
    wx, aq, combo, rgname = df.parallel(lambda f: f(None), tasks, max_workers=4)
    if rgname:
        centre["display_name"] = rgname
    cand = (combo or {}).get("places") or []
    ctx = (combo or {}).get("context") or {"counts": {}}

    if not wx and not cand:
        gaps.append("live weather was unavailable, so no scoring was possible")
        return _out(question, parsed, sid,
                    advice="Live weather is unavailable right now, so I cannot give a grounded "
                           "recommendation. Please try again shortly.",
                    evidence={"centre": centre}, gaps=gaps, centre=centre)
    if not aq:
        gaps.append("air-quality data was unavailable, so the score is weather-only")
    # never let a missing feed crash the answer -- degrade instead
    wx = wx or {}
    aq = aq or {}

    cand = cand or []
    ctx = ctx or {"counts": {}}
    if not cand:
        # the exact venue type came back empty -- broaden and widen until we
        # have somewhere to send them, rather than dead-ending
        cand, radius_m, used, note = df.places_with_fallback(
            centre["lat"], centre["lon"], radius_m, activity)
        if note:
            gaps.append(note)
        if cand:
            try:
                more_ctx = df.geospatial_context(centre["lat"], centre["lon"], radius_m)
                if more_ctx:
                    ctx = more_ctx
            except Exception:
                pass
    if not cand:
        err = df.last_error("places")
        gaps.append("no nearby places were found")
    cand = cand[:MAX_CANDIDATES]
    for p in cand:
        p["activity"] = activity
    ctx_by_place = {p["name"]: ctx for p in cand}
    if not ctx.get("industrial") and not ctx.get("roads"):
        gaps.append("no industrial or road features were found for downwind analysis")

    hours_all = wx.get("hourly", {}).get("time", [])
    if not hours_all:
        # no forecast -- score the present moment so we still return something
        hours_all = [datetime.now().strftime("%Y-%m-%dT%H:00")]
    window = parse_time_window(normalise_time(parsed.get("time_expression")), hours_all)
    hours = window["hours"] or hours_all[:1]

    sunrise = sunset = None
    daily = wx.get("daily", {})
    if daily.get("sunrise"):
        sunrise, sunset = daily["sunrise"][0], daily["sunset"][0]

    # --- pass 1: rank places WITHOUT travel cost ---
    rows0 = build_surface(cand, wx, aq, ctx_by_place, condition, hours,
                          sunrise=sunrise, sunset=sunset) if cand else []
    best_raw: dict[str, float] = {}
    for r in rows0:
        best_raw[r["place"]] = max(best_raw.get(r["place"], -1.0), r["raw_score"])
    top_places = sorted(best_raw, key=best_raw.get, reverse=True)[:6]

    # --- travel time for the top places only, IN PARALLEL ---
    def _route(name: str):
        p = next((x for x in cand if x["name"] == name), None)
        if not p:
            return name, None
        r = df.route_time(centre["lat"], centre["lon"], p["lat"], p["lon"])
        return name, ({"minutes": r["seconds"] / 60.0, "metres": r["metres"]} if r else None)

    travel_detail = dict(df.parallel(_route, top_places, max_workers=4))
    travel = {k: v["minutes"] for k, v in travel_detail.items() if v}

    # --- pass 2: final surface with travel folded in ---
    rows = build_surface(cand, wx, aq, ctx_by_place, condition, hours,
                         travel_by_place=travel, sunrise=sunrise, sunset=sunset) if cand else []

    # --- the SLM decides, the engine validates ---
    decision = _slm_decide(rows, question, activity, condition,
                           window.get("description"), travel_detail)
    res = analyse(rows, prefer=(decision or {}).get("row")) if rows else {"best": None}

    rec = _recommendation_block(res, centre, activity, travel_detail, condition, more, lang)

    evidence = {
        "centre": {"name": centre.get("display_name"), "lat": centre["lat"], "lon": centre["lon"]},
        "centre_conditions": df.conditions(centre["lat"], centre["lon"]),
        "time_window": {"description": window["description"], "hours": hours[:24]},
        "places": [{"name": p["name"], "lat": p["lat"], "lon": p["lon"]} for p in cand],
        "geospatial_context_counts": ctx.get("counts", {}),
        "weather_source": "Open-Meteo",
        "air_quality_source": "Open-Meteo Air-Quality",
        "best": rec.get("best") or {},
    }

    if decision and res.get("best") is decision.get("row"):
        rec["slm"] = {"why": decision.get("why"),
                      "runner_up": decision.get("runner_up"),
                      "why_not_runner_up": decision.get("why_not_runner_up"),
                      "confidence": decision.get("confidence")}
        _fold_decision(rec, decision)

    advice, used_slm = build_advice(parsed, rec, evidence, gaps, use_slm=ENABLE_SLM, lang=lang)
    provenance = _provenance(res["best"], cand, centre, gaps) if cand and res.get("best") else []
    surface = [{"place": r["place"], "hour": r["hour"], "score": r["score"]} for r in rows]

    return _out(
        question, parsed, sid,
        recommendation=rec, advice=advice, surface=surface, evidence=evidence,
        provenance=provenance, gaps=gaps, centre=centre, lang=lang,
        keywords=detect_keywords(question, parsed),
        more_available=bool(more or len(cand) > 4),
        model=(("SLM: " + model_name()) if decision else (parsed.get("parser") or "")),
        decided_by=("slm" if decision else "engine"),
        slm_confidence=(decision or {}).get("confidence"),
    )


def answer(question: str, lat: float | None = None, lon: float | None = None,
           session_id: str | None = None, lang: str = "en") -> dict:
    """Answer a question, in the language the user picked."""
    res = _answer_impl(question, lat=lat, lon=lon, session_id=session_id, lang=lang)
    return _localize_result(res, lang)
