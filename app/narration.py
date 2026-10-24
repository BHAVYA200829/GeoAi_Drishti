"""
Drishti narration.

Builds the final user-facing advice. Deterministic by default (so it always
works), optionally polished by the SLM. Every factual line names the source
it came from -- this is the provenance discipline the challenge asks for.
"""
from __future__ import annotations

from datetime import datetime

from .i18n import t as _t


def _fmt(hour: str) -> str:
    try:
        return datetime.fromisoformat(hour).strftime("%a %d %b, %I:%M %p")
    except Exception:
        return hour


def go_out_verdict(snap: dict) -> tuple:
    """A friendly go / caution / wait verdict from the current conditions."""
    if not snap or not snap.get("available"):
        return "CAUTION", []
    reasons = []
    score = 100
    temp = snap.get("temperature")
    feels = snap.get("feels_like")
    rain = snap.get("rain_prob")
    aqi = snap.get("aqi")
    wind = snap.get("wind")
    uv = snap.get("uv")
    heat = feels if feels is not None else temp
    if heat is not None:
        if heat >= 40:
            score -= 55; reasons.append("very hot (feels like %.0f°C)" % heat)
        elif heat >= 35:
            score -= 30; reasons.append("quite hot (feels like %.0f°C)" % heat)
        elif heat <= 5:
            score -= 25; reasons.append("very cold (%.0f°C)" % heat)
    if rain is not None:
        if rain >= 80:
            score -= 45; reasons.append("rain is very likely (%d%%)" % rain)
        elif rain >= 50:
            score -= 22; reasons.append("rain is likely (%d%%)" % rain)
    if aqi is not None:
        if aqi > 200:
            score -= 45; reasons.append("air quality is unhealthy (AQI %d)" % aqi)
        elif aqi > 150:
            score -= 25; reasons.append("air quality is poor (AQI %d)" % aqi)
        elif aqi > 100:
            score -= 10; reasons.append("air quality is moderate (AQI %d)" % aqi)
    if wind is not None and wind >= 40:
        score -= 15; reasons.append("strong wind (%.0f km/h)" % wind)
    if uv is not None and uv >= 9:
        score -= 8; reasons.append("very high UV")
    if not snap.get("is_day"):
        reasons.append("it is dark outside")
    if score >= 70:
        return "GO", reasons
    if score >= 40:
        return "CAUTION", reasons
    return "AVOID", reasons


def go_out_advice(snap: dict, place: str, lang: str = "en") -> str:
    if not snap or not snap.get("available"):
        return "I could not fetch live conditions for %s right now. Please try again shortly." % place
    verdict, reasons = go_out_verdict(snap)
    head = _t(lang, {"GO": "go_yes", "CAUTION": "go_caution", "AVOID": "go_no"}[verdict])
    bits = []
    if snap.get("temperature") is not None:
        bits.append("%.0f°C" % snap["temperature"])
    if snap.get("feels_like") is not None:
        bits.append("feels like %.0f°C" % snap["feels_like"])
    if snap.get("humidity") is not None:
        bits.append("%.0f%% humidity" % snap["humidity"])
    if snap.get("wind") is not None:
        bits.append("wind %.0f km/h" % snap["wind"])
    if snap.get("rain_prob") is not None:
        bits.append("%.0f%% chance of rain" % snap["rain_prob"])
    if snap.get("aqi") is not None:
        bits.append("AQI %d (%s)" % (snap["aqi"], snap.get("aqi_band", "")))
    lines = ["%s - %s." % (head, place)]
    if bits:
        lines.append(", ".join(bits) + ".")
    if reasons:
        lines.append("Because: " + "; ".join(reasons) + ".")
    else:
        lines.append("Nothing is working against you right now.")
    if snap.get("sunrise") and snap.get("sunset"):
        lines.append("Sunrise %s, sunset %s." % (snap["sunrise"][11:16], snap["sunset"][11:16]))
    lines.append("Sources: Open-Meteo (weather, air quality).")
    return "\n".join(lines)


def carry_and_watch(best: dict, activity: str, condition: str | None,
                    lang: str = "en") -> tuple:
    """Shared with the API so the UI can render these as ordered steps."""
    facts = (best or {}).get("facts", {}) or {}
    watch = []
    for w in _watch_list(best or {}, [], condition):
        if w not in watch:
            watch.append(w)
    return _carry_list(facts, activity or "", lang), watch


def _carry_list(facts: dict, activity: str, lang: str = "en") -> list[str]:
    items = []
    if (facts.get("rain_prob") or 0) >= 40:
        items.append(_t(lang, "carry_rain"))
    if (facts.get("uv") or 0) >= 6:
        items.append(_t(lang, "carry_uv"))
    if (facts.get("aqi") or 0) >= 100:
        items.append(_t(lang, "carry_mask"))
    if (facts.get("temp") or 0) >= 32:
        items.append(_t(lang, "carry_water"))
    if (facts.get("temp") is not None) and facts["temp"] <= 12:
        items.append(_t(lang, "carry_warm"))
    if (facts.get("wind") or 0) >= 30:
        items.append(_t(lang, "carry_wind"))
    if activity == "photo":
        items.append(_t(lang, "carry_photo"))
    if activity in ("running", "cycling", "hiking", "watersport", "field"):
        items.append(_t(lang, "carry_phone"))
    if not items:
        items.append(_t(lang, "carry_basic"))
    return items


def _watch_list(best: dict, flips: list[dict], condition: str | None) -> list[str]:
    watch = []
    filler = ("conditions are comfortable for this activity",
              "conditions look stable through the window")
    for r in best.get("reasons", []):
        if r and r not in watch and r.lower() not in filler:
            watch.append(r)
    for f in flips[:2]:
        watch.append(f"{f['place']} {f['direction']} around {_fmt(f['at'])} ({f['reason']})")
    if condition:
        watch.append(f"your {condition} profile tightens the safe limits")
    return watch


def deterministic_advice(parsed: dict, rec: dict, evidence: dict, gaps: list[str]) -> str:
    best = rec.get("best")
    if not best:
        return ("I could not pin down a specific spot from the live map data for that "
                "activity, so here is the picture for the area instead. Try naming a "
                "landmark, a bigger neighbourhood, or a different activity.")

    activity = (parsed.get("activity_label") or parsed.get("activity") or "outing").replace("_", " ")
    facts = best.get("facts", {})
    window = rec.get("golden_window")
    avoid = rec.get("avoid")
    flips = rec.get("flips", [])

    lines = []
    lines.append(f"Best bet for {activity}: **{best.get('name','a nearby spot')}** at {_fmt(best['hour'])} "
                 f"(suitability {best['score']}/100).")
    if best.get("travel_minutes"):
        lines.append(f"That is about {best['travel_minutes']:.0f} min away.")
    if window:
        lines.append(f"The good window runs {_fmt(window['from'])} to {_fmt(window['to'])} "
                     f"at {window['place']}.")

    why = best.get("reasons") or ["comfortable conditions"]
    lines.append("Why: " + "; ".join(why) + ".")

    facts_bits = []
    if facts.get("temp") is not None:
        facts_bits.append(f"{facts['temp']:.0f}°C")
    if facts.get("rain_prob") is not None:
        facts_bits.append(f"{facts['rain_prob']:.0f}% chance of rain")
    if facts.get("wind") is not None:
        facts_bits.append(f"wind {facts['wind']:.0f} km/h")
    if facts.get("aqi") is not None:
        facts_bits.append(f"AQI {facts['aqi']:.0f}")
    if facts_bits:
        lines.append("Live conditions (Open-Meteo): " + ", ".join(facts_bits) + ".")

    lines.append("Carry: " + ", ".join(_carry_list(facts, parsed.get("activity") or "")) + ".")
    lines.append("Watch for: " + "; ".join(_watch_list(best, flips, parsed.get("persona", {}).get("condition"))) + ".")
    if avoid:
        lines.append(f"Avoid {avoid['place']} around {_fmt(avoid['hour'])} "
                     f"({'; '.join(avoid.get('reasons') or ['poor conditions'])}).")
    if gaps:
        lines.append("Note: " + "; ".join(gaps) + ".")

    lines.append("Sources: Open-Meteo (weather, air quality), OpenStreetMap via Overpass "
                 "(places and geospatial context), Nominatim (geocoding), OSRM (travel time).")
    return "\n".join(lines)


def conditions_advice(snap: dict, place_name: str, lang: str = "en") -> str:
    """Plain-language summary when the user just wants weather/AQI for a place."""
    if not snap or not snap.get("available"):
        return (f"I could not fetch live conditions for {place_name} right now. "
                "Please try again shortly.")
    bits = []
    if snap.get("temperature") is not None:
        bits.append(f"{snap['temperature']:.0f}°C")
    if snap.get("feels_like") is not None:
        bits.append(f"feels like {snap['feels_like']:.0f}°C")
    if snap.get("humidity") is not None:
        bits.append(f"{snap['humidity']:.0f}% humidity")
    if snap.get("wind") is not None:
        bits.append(f"wind {snap['wind']:.0f} km/h")
    if snap.get("rain_prob") is not None:
        bits.append(f"{snap['rain_prob']:.0f}% chance of rain")
    aqi_txt = ""
    if snap.get("aqi") is not None:
        aqi_txt = f" Air quality: AQI {snap['aqi']:.0f} ({snap.get('aqi_band','')})"
        if snap.get("pollen") is not None:
            aqi_txt += f", pollen {snap['pollen']:.0f}"
        aqi_txt += "."
    day = "It is daylight right now." if snap.get("is_day") else "It is dark right now."
    sr = (snap.get("sunrise") or "")[11:16]
    ss = (snap.get("sunset") or "")[11:16]
    sun = f" Sunrise {sr}, sunset {ss}." if sr and ss else ""
    return (_t(lang, "now_in") + " " + place_name + ": " + ", ".join(bits) + "." + aqi_txt + " " + day + sun +
            "\n\nSources: Open-Meteo (weather, air quality).")


def best_day_advice(ranking: list, activity: str | None, place_name: str) -> str:
    if not ranking:
        return ("I could not work out a day this week - live weather or nearby places "
                "were unavailable. Please try again shortly.")
    act = (activity or "outing").replace("_", " ")
    top = ranking[0]
    lines = [f"Best day this week for {act} near {place_name}: **{top['weekday']} "
             f"({top['date']})**, at {top['place']} around {top['hour'][11:16]} "
             f"(suitability {top['score']}/100)."]
    rest = ranking[1:6]
    if rest:
        lines.append("The rest of the week, best first:")
        for r in rest:
            lines.append(f"  - {r['weekday']} {r['date']}: {r['score']}/100 at {r['place']}")
    if top.get("reasons"):
        lines.append("Why: " + "; ".join(top["reasons"]) + ".")
    lines.append("Scored with the place and time comfort surface, using live weather "
                 "and air quality and places from OpenStreetMap.")
    return "\n".join(lines)


def friendly_summary(facts: dict, reasons: list, activity: str) -> str:
    """One warm sentence a person would actually say."""
    f = facts or {}
    temp = f.get("temp")
    feels = f.get("feels_like") or temp
    rain = f.get("rain_prob")
    aqi = f.get("aqi")
    wind = f.get("wind")

    if feels is not None and feels >= 38:
        lead = "It will feel hot out there"
    elif feels is not None and feels >= 32:
        lead = "It will be on the warm side"
    elif feels is not None and feels <= 8:
        lead = "It will be cold out there"
    elif rain is not None and rain >= 60:
        lead = "Rain is likely"
    elif aqi is not None and aqi > 180:
        lead = "The air is not great today"
    else:
        lead = "Conditions look pleasant"

    bits = []
    if temp is not None:
        bits.append("about %.0f°C" % temp)
    if rain is not None:
        bits.append("no rain expected" if rain < 20 else
                    ("a light chance of rain" if rain < 50 else "rain is quite likely (%d%%)" % rain))
    if aqi is not None:
        bits.append("air quality is decent (AQI %.0f)" % aqi if aqi <= 100
                    else "the air is a bit thick (AQI %.0f)" % aqi)
    if wind is not None and wind >= 25:
        bits.append("it will be breezy")
    tail = (", ".join(bits) + ".") if bits else ""

    extra = ""
    for r in (reasons or []):
        rl = (r or "").lower()
        if "comfortable" in rl or "stable" in rl:
            continue
        if rl.startswith("shaded by") or "cools the air" in rl:
            continue          # shade is a plus, not a warning
        extra = " " + r.rstrip(".").capitalize() + "."
        break

    # shade and water are worth saying when it is hot
    shade = f.get("shade_by")
    water = f.get("water_by")
    if f.get("cooling") and (shade or water):
        if shade and water:
            extra += " It is shaded by %s and close to %s, so it stays cooler than the open ground." % (shade, water)
        elif shade:
            extra += " %s next door shades it, so it stays cooler than the open ground." % shade
        else:
            extra += " %s is close by, which takes the edge off the heat." % water
    return (lead + " - " + tail + extra).strip()


def route_advice(route: dict, plan: dict, parsed: dict, centre: dict,
                 gaps: list | None = None, lang: str = "en") -> str:
    """The answer for a journey, not a place."""
    L = lambda k: _t(lang, k)
    mode = {"cycling": "cycling", "walking": "on foot", "driving": "by car"}.get(
        route.get("mode"), "travelling")
    def short(x):
        return (x or "").split(",")[0].strip() or None
    a = short(route.get("origin_name")) or route.get("origin") or L("your location")
    b = short(route.get("destination_name")) or route.get("destination") or ""
    km = (route.get("metres") or plan.get("length_m") or 0) / 1000.0
    mins = (route.get("seconds") or 0) / 60.0

    verdict = plan.get("verdict")
    head = {"go": "Yes - it is a good time to go.",
            "caution": "It is workable, but one stretch is not great.",
            "avoid": "I would not go just now."}.get(verdict, "Here is the picture.")

    out = [f"{a} to {b} {mode}: {head}"]
    out.append(f"The route is about {km:.1f} km, roughly {mins:.0f} min"
               + (" (estimated from the driving route)" if route.get("estimated") else "") + ".")

    worst = plan.get("worst") or {}
    flags = plan.get("flags") or []
    if flags:
        where = f"around {(worst.get('at_m') or 0) / 1000.0:.1f} km in"
        road = plan.get("road")
        if road:
            where += f", on {road}"
        out.append(f"The rough part is {where}: " + ", ".join(flags)
                   + f". That stretch scores {worst.get('score')}/100.")

    best = plan.get("best_hour")
    by_hour = plan.get("by_hour") or []
    if best and len(by_hour) > 1:
        hh = best[-5:]
        out.append(f"The best hour in this window is {hh} - the worst stretch "
                   f"is {plan.get('best_worst')}/100 then.")
        bad = min((x for x in by_hour if x["hour"] != best), key=lambda x: x["worst"])
        if bad["worst"] < plan.get("best_worst", 0) - 5:
            out.append(f"At {bad['hour'][-5:]} it drops to {bad['worst']}/100, so "
                       f"shift the trip if you can.")
    elif best:
        out.append(f"Go around {best[-5:]}.")

    if plan.get("verdict") != "go":
        out.append("If you can shift the trip, do; if not, the stretch above is the one to brace for.")

    if gaps:
        out.append("; ".join(gaps) + ".")
    out.append("Sources: " + ", ".join(df_sources()[:4]) + ".")
    return "\n".join(out)


def df_sources():
    try:
        from .data_fetcher import SOURCES
        return [s.split(" (")[0] for s in SOURCES]
    except Exception:  # noqa: BLE001
        return ["Open-Meteo", "OpenStreetMap via Overpass", "Nominatim", "OSRM"]


def sequential_advice(parsed: dict, rec: dict, evidence: dict, gaps: list, lang: str = "en") -> str:
    """The answer as ordered steps -- the way the UI presents it."""
    best = rec.get("best")
    if not best:
        return deterministic_advice(parsed, rec, evidence, gaps)
    L = lambda k: _t(lang, k)
    steps = []
    if rec.get("summary"):
        steps.append((L("sec_why"), rec["summary"]))
    m = best.get("maps") or {}
    if m.get("text"):
        steps.append((L("sec_reach"), m["text"]))
    if rec.get("carry"):
        steps.append((L("sec_carry"), ", ".join(rec["carry"]) + "."))
    if rec.get("watch"):
        steps.append((L("sec_watch"), "; ".join(rec["watch"]) + "."))
    if rec.get("avoid"):
        a = rec["avoid"]
        steps.append((L("sec_avoid"), "%s (%s)." % (a.get("place", ""),
                                                    "; ".join(a.get("reasons") or ["poor conditions"]))))
    if rec.get("flip_note"):
        steps.append((L("sec_watch"), rec["flip_note"]))
    alts = rec.get("alternatives") or []
    if alts:
        steps.append((L("sec_alts"), "; ".join("%s (%s/100)" % (a["name"], a["score"]) for a in alts[:4])))
    body = "\n".join("%d) %s - %s" % (i, lab, txt) for i, (lab, txt) in enumerate(steps, start=1))
    if gaps:
        body += "\n\n" + "; ".join(gaps)
    body += "\n" + L("sec_sources") + ": Open-Meteo, OpenStreetMap (Overpass), Nominatim, OSRM."
    return body


def build_advice(parsed: dict, rec: dict, evidence: dict, gaps: list[str], use_slm: bool,
                 lang: str = "en") -> tuple[str, bool]:
    """Return (advice_text, used_slm)."""
    if use_slm:
        try:
            from .slm import polish_advice
            text = polish_advice(rec, evidence)
            if text:
                return text, True
        except Exception:
            pass
    return sequential_advice(parsed, rec, evidence, gaps, lang), False
