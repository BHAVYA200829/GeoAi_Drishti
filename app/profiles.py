"""
Activity profiles and persona rules for Drishti.

Each activity has a comfort curve per factor (piecewise-linear, x -> comfort in
[0,1]) plus weights. A single condition can suit one activity and rule out
another, which is the whole point of the problem.

Persona rules are HARD constraints: they cap or zero the score for a given
person, they are not soft advice.
"""
from __future__ import annotations

# comfort curves: list of (x_value, comfort_0_to_1), x ascending
ACTIVITY_PROFILES: dict[str, dict] = {
    "running": {
        "label": "Running / jogging",
        "temp": [[0, .15], [5, .5], [10, 1], [22, 1], [30, .55], [36, .15], [40, 0]],
        "wind": [[0, 1], [20, 1], [35, .6], [50, .3], [65, .1]],
        "humidity": [[25, 1], [60, 1], [80, .6], [95, .3]],
        "rain_prob": [[0, 1], [30, .8], [60, .3], [80, .05], [100, 0]],
        "aqi": [[0, 1], [50, 1], [100, .7], [150, .4], [250, .1], [400, 0]],
        "uv": [[0, 1], [6, .9], [8, .65], [11, .35]],
        "weights": {"temp": .30, "rain_prob": .25, "aqi": .20, "humidity": .10, "wind": .10, "uv": .05},
        "needs_daylight": True,
        "hard_rain_prob": 85,
    },
    "cycling": {
        "label": "Cycling (leisure or commute)",
        "temp": [[0, .1], [8, .6], [15, 1], [26, 1], [32, .6], [38, .15], [42, 0]],
        "wind": [[0, 1], [15, 1], [25, .75], [40, .4], [55, .15]],
        "humidity": [[25, 1], [60, 1], [85, .5], [97, .25]],
        "rain_prob": [[0, 1], [25, .85], [55, .35], [75, .05], [100, 0]],
        "aqi": [[0, 1], [50, 1], [90, .6], [140, .3], [220, .05], [400, 0]],
        "uv": [[0, 1], [6, .9], [8, .6], [11, .3]],
        "weights": {"temp": .22, "rain_prob": .20, "aqi": .25, "wind": .18, "humidity": .10, "uv": .05},
        "needs_daylight": True,
        "hard_rain_prob": 80,
    },
    "golf": {
        "label": "Golf",
        "temp": [[5, .2], [12, .7], [18, 1], [28, 1], [34, .5], [40, .05]],
        "wind": [[0, 1], [12, 1], [22, .6], [35, .25], [50, .05]],
        "humidity": [[20, 1], [70, 1], [90, .5]],
        "rain_prob": [[0, 1], [25, .8], [55, .25], [70, .02], [100, 0]],
        "aqi": [[0, 1], [60, 1], [120, .7], [200, .35], [350, .05]],
        "uv": [[0, 1], [6, .9], [8, .6], [11, .3]],
        "weights": {"temp": .25, "rain_prob": .30, "wind": .20, "aqi": .10, "humidity": .10, "uv": .05},
        "needs_daylight": True,
        "hard_rain_prob": 65,
    },
    "cricket": {
        "label": "Cricket (recreational)",
        "temp": [[5, .2], [15, .8], [20, 1], [30, 1], [35, .5], [41, .05]],
        "wind": [[0, 1], [25, 1], [40, .6], [55, .3]],
        "humidity": [[25, 1], [70, 1], [90, .5]],
        "rain_prob": [[0, 1], [20, .8], [45, .25], [60, .0], [100, 0]],
        "aqi": [[0, 1], [60, 1], [120, .7], [200, .35], [350, .05]],
        "uv": [[0, 1], [7, .9], [9, .6], [12, .3]],
        "weights": {"rain_prob": .40, "temp": .22, "aqi": .13, "humidity": .10, "wind": .10, "uv": .05},
        "needs_daylight": True,
        "hard_rain_prob": 55,
    },
    "football": {
        "label": "Outdoor football / five-a-side",
        "temp": [[2, .15], [10, .7], [16, 1], [26, 1], [32, .55], [38, .1]],
        "wind": [[0, 1], [25, 1], [40, .6], [55, .25]],
        "humidity": [[25, 1], [70, 1], [90, .5]],
        "rain_prob": [[0, 1], [30, .7], [60, .2], [75, .0]],
        "aqi": [[0, 1], [60, 1], [120, .7], [200, .3], [350, .05]],
        "uv": [[0, 1], [7, .9], [9, .6], [12, .3]],
        "weights": {"temp": .28, "rain_prob": .27, "aqi": .15, "humidity": .12, "wind": .13, "uv": .05},
        "needs_daylight": True,
        "hard_rain_prob": 70,
    },
    "tennis": {
        "label": "Outdoor tennis",
        "temp": [[5, .2], [14, .8], [20, 1], [29, 1], [34, .5], [40, .05]],
        "wind": [[0, 1], [12, 1], [20, .55], [32, .2], [45, .05]],
        "humidity": [[20, 1], [70, 1], [90, .5]],
        "rain_prob": [[0, 1], [20, .8], [45, .2], [60, .0]],
        "aqi": [[0, 1], [60, 1], [120, .7], [200, .35], [350, .05]],
        "uv": [[0, 1], [6, .9], [8, .6], [11, .3]],
        "weights": {"rain_prob": .32, "wind": .22, "temp": .22, "aqi": .12, "humidity": .07, "uv": .05},
        "needs_daylight": True,
        "hard_rain_prob": 55,
    },
    "hiking": {
        "label": "Hiking / nature walk",
        "temp": [[-2, .05], [6, .6], [14, 1], [24, 1], [30, .5], [36, .1]],
        "wind": [[0, 1], [25, 1], [40, .55], [60, .2]],
        "humidity": [[20, 1], [70, 1], [92, .45]],
        "rain_prob": [[0, 1], [30, .7], [60, .2], [80, .0]],
        "aqi": [[0, 1], [70, 1], [130, .7], [220, .3], [350, .05]],
        "uv": [[0, 1], [6, .9], [8, .6], [11, .35]],
        "weights": {"temp": .25, "rain_prob": .25, "aqi": .12, "humidity": .13, "wind": .15, "uv": .10},
        "needs_daylight": True,
        "hard_rain_prob": 80,
        "max_slope": 25,  # % slope tolerance before penalty
    },
    "picnic": {
        "label": "Picnic / park outing",
        "temp": [[5, .2], [14, .8], [22, 1], [30, 1], [34, .55], [40, .1]],
        "wind": [[0, 1], [20, 1], [35, .55], [50, .2]],
        "humidity": [[20, 1], [70, 1], [90, .5]],
        "rain_prob": [[0, 1], [25, .8], [50, .3], [70, .0]],
        "aqi": [[0, 1], [60, 1], [120, .7], [200, .35], [350, .05]],
        "uv": [[0, 1], [6, .9], [8, .6], [11, .35]],
        "weights": {"rain_prob": .32, "temp": .28, "wind": .15, "aqi": .13, "humidity": .07, "uv": .05},
        "needs_daylight": True,
        "hard_rain_prob": 70,
    },
    "event": {
        "label": "Outdoor event / gathering",
        "temp": [[5, .15], [15, .8], [22, 1], [31, 1], [36, .45], [42, .05]],
        "wind": [[0, 1], [20, 1], [35, .55], [50, .2]],
        "humidity": [[20, 1], [70, 1], [92, .45]],
        "rain_prob": [[0, 1], [20, .85], [45, .3], [60, .0]],
        "aqi": [[0, 1], [60, 1], [120, .7], [200, .35], [350, .05]],
        "uv": [[0, 1], [6, .9], [8, .6], [11, .35]],
        "weights": {"rain_prob": .40, "temp": .25, "wind": .12, "aqi": .12, "humidity": .06, "uv": .05},
        "needs_daylight": True,
        "hard_rain_prob": 55,
    },
    "photo": {
        "label": "Outdoor photography session",
        "temp": [[0, .3], [10, .8], [20, 1], [32, 1], [38, .6]],
        "wind": [[0, 1], [30, 1], [50, .6], [70, .3]],
        "humidity": [[20, 1], [80, 1], [95, .6]],
        "rain_prob": [[0, 1], [30, .85], [55, .5], [80, .2]],
        "aqi": [[0, 1], [100, 1], [200, .7], [350, .4], [500, .1]],
        "uv": [[0, 1], [8, 1], [11, .7]],
        "weights": {"temp": .15, "rain_prob": .20, "aqi": .20, "wind": .10, "humidity": .05, "uv": .05,
                    "golden": .25},
        "needs_daylight": False,
        "golden_hour": True,
        "hard_rain_prob": 90,
    },
}

# --- extra families so ANY outdoor activity gets a sensible profile ---
ACTIVITY_PROFILES["watersport"] = {
    "label": "Water-based activity (boating, kayaking, swimming, fishing)",
    "temp": [[5, .1], [15, .5], [22, 1], [32, 1], [38, .5], [42, .1]],
    "wind": [[0, 1], [15, 1], [25, .7], [38, .35], [50, .1]],   # wind is a safety factor on water
    "humidity": [[20, 1], [80, 1], [95, .6]],
    "rain_prob": [[0, 1], [25, .8], [50, .35], [70, .1], [100, 0]],
    "aqi": [[0, 1], [80, 1], [150, .7], [250, .35], [400, .05]],
    "uv": [[0, 1], [6, .9], [8, .6], [11, .3]],
    "weights": {"temp": .28, "wind": .25, "rain_prob": .22, "aqi": .10, "humidity": .10, "uv": .05},
    "needs_daylight": True,
    "hard_rain_prob": 75,
}
ACTIVITY_PROFILES["field"] = {
    "label": "Field / court activity",
    "temp": [[3, .15], [12, .7], [18, 1], [28, 1], [34, .55], [40, .1]],
    "wind": [[0, 1], [22, 1], [35, .6], [50, .25]],
    "humidity": [[25, 1], [70, 1], [90, .5]],
    "rain_prob": [[0, 1], [25, .8], [55, .25], [70, .0]],
    "aqi": [[0, 1], [60, 1], [120, .7], [200, .35], [350, .05]],
    "uv": [[0, 1], [7, .9], [9, .6], [12, .3]],
    "weights": {"temp": .28, "rain_prob": .27, "aqi": .14, "humidity": .10, "wind": .16, "uv": .05},
    "needs_daylight": True,
    "hard_rain_prob": 70,
}
ACTIVITY_PROFILES["gentle"] = {
    "label": "Gentle outdoor activity",
    "temp": [[5, .25], [14, .8], [22, 1], [31, 1], [36, .5], [42, .1]],
    "wind": [[0, 1], [22, 1], [38, .55], [55, .2]],
    "humidity": [[20, 1], [72, 1], [92, .45]],
    "rain_prob": [[0, 1], [28, .82], [52, .3], [72, .0]],
    "aqi": [[0, 1], [60, 1], [120, .72], [200, .35], [350, .05]],
    "uv": [[0, 1], [6, .9], [8, .6], [11, .35]],
    "weights": {"rain_prob": .30, "temp": .28, "aqi": .14, "wind": .13, "humidity": .10, "uv": .05},
    "needs_daylight": True,
    "hard_rain_prob": 72,
}

# free-text activity -> a profile family
ACTIVITY_ALIASES = {
    "boating": "watersport", "boat": "watersport", "kayak": "watersport", "kayaking": "watersport",
    "canoe": "watersport", "canoeing": "watersport", "swim": "watersport", "swimming": "watersport",
    "surf": "watersport", "surfing": "watersport", "fish": "watersport", "fishing": "watersport",
    "sailing": "watersport", "rafting": "watersport", "paddle": "watersport",
    "paddleboarding": "watersport", "rowing": "watersport", "row": "watersport",
    "archery": "field", "skating": "field", "skate": "field", "skateboard": "field",
    "horse riding": "field", "riding": "field", "frisbee": "field", "badminton": "field",
    "volleyball": "field", "basketball": "field", "baseball": "field", "hockey": "field",
    "rugby": "field", "kabaddi": "field", "athletics": "field", "baseball": "field",
    "yoga": "gentle", "meditation": "gentle", "birdwatching": "gentle", "bird watching": "gentle",
    "stroll": "gentle", "sightseeing": "gentle", "gardening": "gentle",
    "walk": "hiking", "walking": "hiking", "camping": "hiking", "trek": "hiking",
    "trekking": "hiking", "mountaineering": "hiking",
    "jog": "running", "jogging": "running", "run": "running", "marathon": "running",
    "bike": "cycling", "bicycle": "cycling", "biking": "cycling",
    "soccer": "football", "futsal": "football",
    "photography": "photo", "shoot": "photo",
    "barbecue": "picnic", "bbq": "picnic",
    "wedding": "event", "fair": "event", "party": "event", "gathering": "event",
    "festival": "event", "function": "event",
    # objectives that are not classic sports
    "movie": "event", "movies": "event", "cinema": "event", "film": "event",
    "screening": "event", "theatre": "event", "theater": "event", "open air cinema": "event",
    "shopping": "gentle", "market": "gentle", "bazaar": "gentle",
    "food": "picnic", "dining": "picnic", "dinner": "picnic", "lunch": "picnic",
    "brunch": "picnic", "breakfast": "picnic", "coffee": "picnic", "cafe": "picnic",
    "stargazing": "gentle", "astronomy": "gentle", "star gazing": "gentle",
    "kite": "field", "kite flying": "field",
    "painting": "gentle", "sketching": "gentle", "drawing": "gentle", "reading": "gentle",
    "climbing": "hiking", "bouldering": "hiking",
    "pilates": "gentle", "zumba": "gentle", "aerobics": "gentle", "exercise": "field",
    "workout": "field", "games": "field", "sports": "field", "play": "field",
    "photoshoot": "photo", "photo walk": "photo",
    "multiplex": "movie", "talkies": "movie", "picture house": "movie",
    "monument": "gentle", "landmark": "gentle", "heritage": "gentle",
    "fort": "gentle", "palace": "gentle", "museum": "gentle",
    # place categories
    "school": "gentle", "college": "gentle", "university": "gentle",
    "tuition": "gentle", "coaching": "gentle", "campus": "gentle",
    "hospital": "gentle", "clinic": "gentle", "temple": "gentle",
    "mosque": "gentle", "church": "gentle", "gurudwara": "gentle",
    "atm": "gentle", "bank": "gentle", "petrol": "gentle", "fuel": "gentle",
    "hotel": "gentle", "gym": "field", "playground": "field",
    "restaurant": "picnic", "pizza": "picnic",
}


def resolve_profile_key(activity: str | None) -> str:
    """Map ANY activity name to a profile family that exists in ACTIVITY_PROFILES."""
    if not activity:
        return "picnic"
    a = activity.lower().strip()
    if a in ACTIVITY_PROFILES:
        return a
    if a in ACTIVITY_ALIASES:
        return ACTIVITY_ALIASES[a]
    for w in ("water", "boat", "kayak", "swim", "surf", "fish", "sail", "paddle", "raft", "row", "canoe"):
        if w in a:
            return "watersport"
    for w in ("yoga", "medit", "bird", "garden", "stroll", "nature", "sightsee"):
        if w in a:
            return "gentle"
    for w in ("skat", "arch", "ride", "frisbee", "badminton", "volley", "basket", "hockey", "kabaddi"):
        if w in a:
            return "field"
    return "gentle"   # unknown outdoor activity -> safe general profile


# the activities shown as pickable "tasks" in the UI (10 core + extras)
FEATURED_ACTIVITIES = [
    ("running", "Running"),
    ("cycling", "Cycling"),
    ("cricket", "Cricket"),
    ("football", "Football"),
    ("tennis", "Tennis"),
    ("golf", "Golf"),
    ("hiking", "Hiking"),
    ("picnic", "A picnic"),
    ("photo", "Photography"),
    ("event", "An outdoor event"),
    ("watersport", "Boating"),
    ("field", "Archery"),
    ("gentle", "Yoga"),
]

# human-readable factor names
FACTOR_LABELS = {
    "temp": "temperature", "wind": "wind", "humidity": "humidity",
    "rain_prob": "chance of rain", "aqi": "air quality", "uv": "UV", "golden": "light",
}

# persona -> hard rules. Each rule: factor, op, value, cap (max score) or None to zero.
# 'cap' means the score cannot exceed this value; None means reject outright.
PERSONA_RULES: dict[str, list[dict]] = {
    "asthma": [
        {"factor": "aqi", "op": ">", "value": 150, "cap": None, "why": "air quality is unsafe for asthma"},
        {"factor": "aqi", "op": ">", "value": 100, "cap": 45, "why": "elevated air pollution for asthma"},
        {"factor": "pollen", "op": ">", "value": 50, "cap": 50, "why": "high pollen for asthma"},
        {"factor": "temp", "op": "<", "value": 5, "cap": 45, "why": "cold air can trigger asthma"},
    ],
    "senior": [
        {"factor": "heat_index", "op": ">", "value": 40, "cap": None, "why": "heat stress risk for seniors"},
        {"factor": "heat_index", "op": ">", "value": 34, "cap": 45, "why": "high heat for seniors"},
        {"factor": "temp", "op": "<", "value": 5, "cap": 45, "why": "cold risk for seniors"},
        {"factor": "slope", "op": ">", "value": 20, "cap": 40, "why": "steep terrain for seniors"},
    ],
    "child": [
        {"factor": "aqi", "op": ">", "value": 120, "cap": 40, "why": "poor air quality for a child"},
        {"factor": "uv", "op": ">", "value": 8, "cap": 50, "why": "high UV for a child"},
        {"factor": "heat_index", "op": ">", "value": 36, "cap": 40, "why": "heat risk for a child"},
    ],
    "pregnancy": [
        {"factor": "heat_index", "op": ">", "value": 36, "cap": 40, "why": "heat risk in pregnancy"},
        {"factor": "aqi", "op": ">", "value": 120, "cap": 40, "why": "air quality risk in pregnancy"},
        {"factor": "temp", "op": "<", "value": 5, "cap": 45, "why": "cold risk in pregnancy"},
    ],
}


def get_profile(activity: str | None) -> dict:
    """Return the profile for ANY activity, resolved through the family map."""
    return ACTIVITY_PROFILES[resolve_profile_key(activity)]


def normalise_condition(text: str | None) -> str | None:
    """Map free text to a known persona key."""
    if not text:
        return None
    t = text.lower()
    if "asthma" in t or "wheez" in t:
        return "asthma"
    if "senior" in t or "elder" in t or "parent" in t or "old" in t:
        return "senior"
    if "child" in t or "kid" in t or "baby" in t or "toddler" in t:
        return "child"
    if "pregnan" in t or "expecting" in t:
        return "pregnancy"
    return None
