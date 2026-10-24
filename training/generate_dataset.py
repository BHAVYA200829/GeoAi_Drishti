#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the fine-tuning set for Drishti's small language model.

Drishti's model does three jobs, and this script writes training examples for all
three. Nothing here needs a network: the places, the weather and the geospatial
flags are drawn from realistic Indian ranges so the set can be regenerated
anywhere, reproducibly.

    python training/generate_dataset.py --out data --n 1200

Outputs
    data/parse.jsonl      question  -> slots JSON          (NLU)
    data/decide.jsonl     FACTS     -> decision JSON       (the reasoning step)
    data/narrate.jsonl    evidence  -> grounded prose      (NLG)
    data/all.jsonl        all three, shuffled, chat format

Every record is in the standard chat format:

    {"messages": [{"role": "system", ...}, {"role": "user", ...},
                  {"role": "assistant", ...}]}
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------- vocabulary
ACTIVITIES = [
    ("running", ["running", "a run", "jogging", "a jog"]),
    ("cycling", ["cycling", "a cycle ride", "a bike ride"]),
    ("golf", ["golf", "a round of golf"]),
    ("cricket", ["cricket", "a cricket match", "cricket practice"]),
    ("football", ["football", "five-a-side", "a kickabout"]),
    ("tennis", ["tennis", "a game of tennis"]),
    ("hiking", ["hiking", "a hike", "a nature walk", "a walk"]),
    ("picnic", ["a picnic", "a park outing"]),
    ("event", ["an outdoor event", "a birthday party", "a fair"]),
    ("photo", ["photography", "sunset photos", "a photo walk"]),
    ("movie", ["a movie", "the cinema"]),
    ("monument", ["monuments", "a heritage walk", "a museum"]),
    ("shopping", ["shopping", "a mall"]),
    ("food", ["street food", "a restaurant"]),
]

PLACES = [
    ("Koregaon Park", "Pune"), ("Powai", "Mumbai"), ("Bandra", "Mumbai"),
    ("Whitefield", "Bengaluru"), ("Indiranagar", "Bengaluru"), ("Hinjewadi", "Pune"),
    ("Andheri", "Mumbai"), ("Lonavala", None), ("Thane", None), ("Delhi", None),
    ("New Delhi", None), ("Ayodhya", None), ("Varanasi", None), ("Chennai", None),
    ("Kolkata", None), ("Hyderabad", None), ("Jaipur", None), ("Goa", None),
    ("Coimbatore", None), ("Kochi", None), ("Nagpur", None), ("Surat", None),
]

VENUE_WORDS = ["Park", "Garden", "Maidan", "Ground", "Udyan", "Lake", "Playground",
               "Sports Complex", "Court", "Field", "Green", "Reserve"]

TIMES = [None, "this evening", "tomorrow morning", "tomorrow evening", "tonight",
         "this weekend", "on Sunday", "in the next two days", "at 8 AM", "after work",
         "around sunset", "day after tomorrow"]

PERSONAS = [None, "asthma", "senior", "child", "pregnancy"]

CARRY = ["water", "a hat", "sun cream", "sunglasses", "an umbrella", "a rain jacket",
         "a mask", "a warm layer", "a jacket", "insect repellent", "a torch",
         "sturdy shoes", "a charged phone", "snacks"]

WATCH = ["heat", "cold", "rain", "air quality", "wind", "uv", "pollen", "darkness",
         "crowds", "closed", "the trip", "uneven ground", "insects"]

FLAGS = ["conditions are comfortable for this activity",
         "downwind of a road while the wind blows from there",
         "outside daylight hours", "closed at this hour (opens at 06:00)",
         "shaded by a garden next door; close to a lake, which cools the air",
         "quite hot (feels like 39°C)", "rain is likely (70%)",
         "air quality is poor (AQI 210)"]

PARSE_SYSTEM = """You are a query parser for an outdoor-activity advisor in India.
Convert the user's free-form question into STRICT JSON. Output JSON only, no prose.
Schema:
{"activity": one of ["running","cycling","golf","cricket","football","tennis","hiking","picnic","event","photo"] or null,
 "location_text": string or null,
 "radius_km": number or null,
 "time_expression": raw time phrase or null,
 "persona": {"age": int or null, "condition": "asthma"|"senior"|"child"|"pregnancy" or null},
 "needs_clarification": bool,
 "clarification_question": string or null}
Rules: never guess a location; if none given set location_text=null and needs_clarification=true.
Keep vague times ("this weekend") as-is. Output JSON only."""

DECIDE_SYSTEM = """You are Drishti's decision layer for an outdoor-activity advisor in India.
You are given FACTS: real places, real hours, and the real weather and air-quality
numbers for each. Choose between them. You may NOT invent a place, an hour or a
number -- everything you name must appear in FACTS.

Output STRICT JSON only, no prose, with exactly these keys:
{"best_place": <one of the place names in FACTS, copied exactly>,
 "best_hour": "HH:MM" (one of that place's hours),
 "why": "one or two short sentences, no numbers",
 "runner_up": <another place name from FACTS, or null>,
 "why_not_runner_up": "one short sentence, no numbers",
 "carry": [up to 3 short items],
 "watch": [up to 3 items],
 "confidence": "high" | "medium" | "low"}

Weigh the person, the task and the hour: a shaded place beside water wins on a hot
afternoon, a place downwind of a road loses, a place that is shut at that hour is
not a choice, and a long trip only pays off if the conditions really are better.
Do not quote any number in "why" or "why_not_runner_up". Output JSON only."""

NARRATE_SYSTEM = """You are Drishti, an outdoor-activity advisor in India.
Write a short, warm recommendation using ONLY the facts in EVIDENCE.
Never invent a weather value, place or time that is not in EVIDENCE.
Name the data source for factual claims. If a value is missing, say it is unavailable.
Cover: best place and time, what to carry, what to watch for, and when to avoid going.
Keep it under 160 words."""


# ---------------------------------------------------------------- generators
def slots_for(rng, activity, place, time_expr, persona, radius):
    return {
        "activity": activity,
        "location_text": place,
        "radius_km": radius,
        "time_expression": time_expr,
        "persona": {"age": None, "condition": persona},
        "needs_clarification": place is None,
        "clarification_question": None if place else
        "Which place in India should I look at? A city, a neighbourhood or a landmark works.",
    }


def question_for(rng, act_words, place, time_expr, persona, radius):
    word = rng.choice(act_words)
    bits = []
    if persona == "asthma":
        bits.append(rng.choice(["I have mild asthma,", "My asthma plays up in bad air,"]))
    elif persona == "senior":
        bits.append(rng.choice(["For my parents,", "My father is 72,"]))
    elif persona == "child":
        bits.append(rng.choice(["For my kids,", "With a 6-year-old,"]))
    elif persona == "pregnancy":
        bits.append("My wife is pregnant,")
    if place:
        where = rng.choice([f"near {place}", f"in {place}", f"around {place}",
                            f"within {radius} km of {place}" if radius else f"near {place}"])
    else:
        where = rng.choice(["", "somewhere nice", "near me"])
    when = f" {time_expr}" if time_expr else ""
    lead = rng.choice(["Where can I go", "Where's good for", "Any good spot for",
                       "Suggest a place for", "Best place for"])
    q = f"{' '.join(bits)} {lead} {word}{when} {where}".strip()
    q = " ".join(q.split())
    if not q.endswith("?"):
        q += "?"
    return q[0].upper() + q[1:]


def facts_for(rng, places):
    out = []
    for name in places:
        hours = []
        for h in sorted(rng.sample([5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20], 3)):
            temp = rng.randint(18, 39)
            hours.append({
                "hour": f"{h:02d}:00",
                "score": round(rng.uniform(28, 96), 1),
                "temp_c": temp,
                "feels_like_c": temp + rng.randint(-1, 5),
                "rain_pct": rng.choice([0, 0, 5, 10, 30, 60, 85]),
                "aqi": rng.randint(28, 260),
                "wind_kmh": rng.randint(2, 26),
                "uv": rng.randint(0, 11),
                "travel_min": rng.randint(3, 55),
                "flags": rng.sample(FLAGS, rng.randint(0, 2)),
            })
        hours.sort(key=lambda x: -x["score"])
        out.append({"name": name, "best_hour": hours[0]["hour"],
                    "best_score": hours[0]["score"], "travel_min": hours[0]["travel_min"],
                    "shaded_by": rng.choice([None, "a garden", "a park"]),
                    "near_water": rng.choice([None, None, "a lake", "a river"]),
                    "sun_azimuth": rng.randint(60, 300),
                    "hours": hours})
    out.sort(key=lambda p: -p["best_score"])
    return out


def decide_for(rng, facts):
    good, bad = facts[0], facts[-1]
    h = good["hours"][0]
    why = []
    if h["temp_c"] >= 33:
        why.append("It is the cooler option at that hour and there is shade close by.")
    elif h["aqi"] <= 80:
        why.append("The air is cleanest there at that hour and it is close by.")
    else:
        why.append("It scores best for the hour you asked about and it is not far.")
    if good.get("near_water"):
        why.append("The water nearby keeps the air a little milder.")
    not_bad = []
    if bad["hours"][0]["aqi"] > 120:
        not_bad.append("the air there is poorer")
    if any("downwind" in f for f in bad["hours"][0]["flags"]):
        not_bad.append("it sits downwind of a road")
    if bad["travel_min"] > 30:
        not_bad.append("it is a longer trip")
    if not not_bad:
        not_bad.append("the conditions are simply not as good")
    watch = []
    if h["aqi"] > 100:
        watch.append("air quality")
    if h["temp_c"] >= 33:
        watch.append("heat")
    if h["rain_pct"] >= 50:
        watch.append("rain")
    if h["uv"] >= 8:
        watch.append("uv")
    if not watch:
        watch = ["the trip"]
    carry = ["water"]
    if h["temp_c"] >= 30:
        carry.append("a hat")
    if h["uv"] >= 6:
        carry.append("sun cream")
    if h["rain_pct"] >= 40:
        carry.append("an umbrella")
    if h["aqi"] > 120:
        carry.append("a mask")
    return {
        "best_place": good["name"],
        "best_hour": h["hour"],
        "why": " ".join(why[:2]),
        "runner_up": bad["name"] if bad["name"] != good["name"] else None,
        "why_not_runner_up": "It loses because " + " and ".join(not_bad[:2]) + ".",
        "carry": carry[:3],
        "watch": watch[:3],
        "confidence": rng.choice(["high", "high", "medium"]),
    }


def narrate_for(rng, facts, decision, activity):
    good = next(p for p in facts["places"] if p["name"] == decision["best_place"])
    h = next(x for x in good["hours"] if x["hour"] == decision["best_hour"])
    when = facts.get("when") or "the coming hours"
    lines = [f"Go to {good['name']} around {h['hour']} for {activity}.",
             f"At that hour it is about {h['temp_c']}°C, feels like {h['feels_like_c']}°C, "
             f"rain {h['rain_pct']}%, AQI {h['aqi']}, wind {h['wind_kmh']} km/h "
             f"(Open-Meteo), and it is about {h['travel_min']} min away (OSRM)."]
    if decision.get("carry"):
        lines.append("Carry " + ", ".join(decision["carry"]) + ".")
    if decision.get("watch"):
        lines.append("Watch for " + ", ".join(decision["watch"]) + ".")
    if decision.get("runner_up"):
        lines.append(f"{decision['runner_up']} is the second option; "
                     f"{decision['why_not_runner_up'].rstrip('.')}.")
    when_clean = when
    for pre in ("on ", "in ", "at "):
        if when_clean.startswith(pre):
            when_clean = when_clean[len(pre):]
    lines.append(f"Looking at {when_clean}. Places and the geospatial context come from "
                 f"OpenStreetMap via Overpass, geocoding from Nominatim.")
    return " ".join(lines)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parse_rows, decide_rows, narrate_rows = [], [], []

    for _ in range(args.n):
        activity, words = rng.choice(ACTIVITIES)
        place, _city = rng.choice(PLACES) if rng.random() < 0.85 else (None, None)
        time_expr = rng.choice(TIMES)
        persona = rng.choice(PERSONAS)
        radius = rng.choice([None, None, 5, 10, 15, 25, 40])

        # 1. parse
        q = question_for(rng, words, place, time_expr, persona, radius)
        parse_rows.append({"messages": [
            {"role": "system", "content": PARSE_SYSTEM},
            {"role": "user", "content": q},
            {"role": "assistant",
             "content": json.dumps(slots_for(rng, activity, place, time_expr, persona, radius),
                                   ensure_ascii=False)},
        ]})

        # 2. decide  +  3. narrate
        names = rng.sample(VENUE_WORDS, rng.randint(3, 5))
        names = [f"{rng.choice(['Shivaji', 'Kamala Nehru', 'Rajiv Gandhi', 'Sanjay',
                                'Nehru', 'Tilak', 'Ambedkar', 'Sardar'])} {n}"
                 if rng.random() < 0.7 else f"{n} {rng.randint(1, 9)}" for n in names]
        facts = {"task": activity, "person": persona or "none",
                 "when": time_expr or "the coming hours", "now": None,
                 "travel_budget_minutes": None, "places": facts_for(rng, names)}
        decision = decide_for(rng, facts["places"])
        decide_rows.append({"messages": [
            {"role": "system", "content": DECIDE_SYSTEM},
            {"role": "user", "content": "FACTS: " + json.dumps(facts, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)},
        ]})
        narrate_rows.append({"messages": [
            {"role": "system", "content": NARRATE_SYSTEM},
            {"role": "user", "content": "EVIDENCE: " + json.dumps(
                {"facts": facts, "decision": decision}, ensure_ascii=False)},
            {"role": "assistant", "content": narrate_for(rng, facts, decision, activity)},
        ]})

    def dump(name, rows):
        with (out / name).open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump("parse.jsonl", parse_rows)
    dump("decide.jsonl", decide_rows)
    dump("narrate.jsonl", narrate_rows)
    allrows = parse_rows + decide_rows + narrate_rows
    rng.shuffle(allrows)
    dump("all.jsonl", allrows)

    print(f"wrote {len(parse_rows)} parse / {len(decide_rows)} decide / "
          f"{len(narrate_rows)} narrate -> {out}/")
    print("example (decide):")
    print(json.dumps(decide_rows[0]["messages"][2], ensure_ascii=False)[:300])


if __name__ == "__main__":
    main()
