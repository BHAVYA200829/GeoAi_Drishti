"""
Query understanding for Drishti.

Two paths:
  * SLM path  -- the <=3B model fills the slot schema (job 1 of the model).
  * Rule path -- a deterministic fallback so the assistant ALWAYS works,
                 even if the model is not loaded or the GPU/host is down.

The deterministic post-processing always runs, so a partial SLM answer is
still completed and never trusted blindly.
"""
from __future__ import annotations

import re

from .config import DEFAULT_RADIUS_KM, ENABLE_SLM
from .profiles import normalise_condition

ACTIVITY_KEYWORDS = [
    ("monument", ["monument", "landmark", "heritage", "fort", "palace", "museum",
                  "memorial", "archaeolog", "historic", "tourist attraction",
                  "monuments", "landmarks", "forts", "palaces", "museums"]),
    ("movie", ["movie", "movies", "cinema", "film", "screening", "picture house",
               "multiplex", "talkies", "picture hall", "picture"]),
    ("photo", ["photo", "photograph", "shoot"]),
    ("running", ["running", "run", "jog", "marathon"]),
    ("cycling", ["cycl", "bike", "bicycle"]),
    ("golf", ["golf"]),
    ("cricket", ["cricket"]),
    ("football", ["football", "soccer", "futsal", "five-a-side"]),
    ("tennis", ["tennis"]),
    ("hiking", ["hike", "hiking", "trek", "trail", "nature walk", "walk", "camp", "mountaineer"]),
    ("picnic", ["picnic", "park outing", "day out", "barbecue", "barbeque", "bar-b-que", "bbq"]),
    ("event", ["event", "wedding", "fair", "gathering", "birthday", "party", "function", "festival"]),
    # --- extra activities, so anything the user names works ---
    ("watersport", ["boating", "boat", "kayak", "canoe", "swim", "surf", "fish", "sail",
                    "raft", "paddle", "rowing", "row", "jet ski", "water sport", "watersport",
                    "snorkel", "diving", "dive"]),
    ("field", ["archery", "skat", "skateboard", "horse riding", "riding", "frisbee",
               "badminton", "volleyball", "basketball", "hockey", "rugby", "kabaddi",
               "athletics", "sports day"]),
    ("gentle", ["yoga", "meditation", "birdwatch", "bird watch", "birding", "stroll",
                "sightseeing", "gardening", "nature", "dhyan", "pranayama", "walking"]),
    ("reading", ["read", "reading", "book", "library visit", "study"]),
    ("sportwatch", ["match", "stadium", "sports day", "tournament", "ipl", "league"]),
    ("photo", ["sunset", "sunrise", "golden hour", "photo walk", "photo walk"]),
]

# ---------------------------------------------------------------- Hinglish
# so "mujhe kal subah cricket khelna hai powai me" is understood as it is
HINGLISH_ACTIVITY = [
    ("cricket", ["cricket khel", "cricket kheln"]),
    ("running", ["daud", "doud", "running", "jogging"]),
    ("cycling", ["cycle chala", "cycle", "cycling", "bike chala"]),
    ("watersport", ["tair", "tairn", "swim", "naha"]),
    ("hiking", ["trek", "trekking", "pahad", "parvat"]),
    ("picnic", ["picnic", "piknik", "barbeque", "barbecue", "khaana khane"]),
    ("gentle", ["ghumn", "ghoom", "tahal", "sair", "dhyan", "yoga", "walk", "paidal"]),
    ("reading", ["padh", "padhn", "kitab"]),
    ("food", ["khana", "khaana", "nashta", "chai", "coffee"]),
    ("field", ["khel", "kheln", "khelne", "match khel"]),
    ("sportwatch", ["match dekh", "match dekhn"]),
    ("photo", ["photo", "tasveer", "shoot"]),
    ("event", ["shaadi", "function", "program", "mela"]),
]

HINGLISH_TIME = [
    ("parso", "day after tomorrow"), ("kal", "tomorrow"), ("aaj", "today"),
    ("abhi", "now"), ("subah", "morning"), ("dopahar", "afternoon"),
    ("shaam", "evening"), ("sham", "evening"), ("raat", "night"),
    ("raat ko", "night"), ("sona", "night"), ("hafte", "weekend"),
    ("weekend", "weekend"), ("agle hafte", "next week"),
]


def normalise_time(expr: str | None) -> str | None:
    """Translate Hinglish time words into the English the parser understands."""
    if not expr:
        return expr
    t = expr.lower()
    for hi, en in HINGLISH_TIME:
        if re.search(r"\b" + re.escape(hi) + r"\b", t):
            t = re.sub(r"\b" + re.escape(hi) + r"\b", en, t)
    return t.strip()


FRIENDLY_FAMILY = {
    "running": "a run", "cycling": "a cycle ride", "hiking": "a hike",
    "cricket": "cricket", "football": "football", "tennis": "tennis", "golf": "golf",
    "picnic": "a picnic", "photo": "photography", "watersport": "the water",
    "field": "a game", "gentle": "a walk", "event": "an outing",
    "reading": "reading", "food": "food", "sportwatch": "a match",
    "movie": "a movie", "shopping": "shopping",
}

ENGLISH_BLOCK = {
    "suggest", "recommend", "give", "tell", "show", "find", "want", "need", "help",
    "please", "best", "good", "some", "any", "which", "what", "where", "can", "could",
    "would", "should", "movie", "cinema", "film", "running", "cycling", "walking",
    "cricket", "football", "tennis", "golf", "hiking", "picnic", "swimming", "yoga",
    "playing", "going", "watch", "see", "visit", "planning", "looking", "me",
    "you", "we", "us", "them", "him", "her", "it",
}

HINGLISH_STOP = {
    "kya", "ka", "ki", "ke", "ko", "se", "me", "mein", "mai", "hum", "main", "mujhe",
    "aaj", "kal", "parso", "abhi", "subah", "shaam", "sham", "raat", "dopahar",
    "jana", "jaana", "jau", "jaun", "ja", "chal", "chalo", "kaha", "kahan", "kahaan",
    "bahar", "andar", "sakta", "sakti", "sakte", "hu", "hun", "hoon", "hai", "hain",
    "chahiye", "karna", "karne", "khelna", "khelne", "ghumne", "ghumna", "dekhne",
    "padhna", "padhne", "khana", "khaana", "lena", "dena", "ek", "aur", "bhi",
    "the", "a", "an", "for", "to", "of", "and", "is", "it", "good", "best",
    "please", "want", "would", "like", "going", "go", "me", "my", "near", "around",
    "today", "tomorrow", "now", "this", "next", "at", "in", "on", "with",
}

PRONOUN_PLACES = {"me", "my place", "my home", "home", "here", "my location",
                  "my area", "my city", "nearby", "around", "my town", "my side",
                  "mere", "yaha", "yahaan", "idhar"}

CONDITION_WORDS = ["aqi", "air quality", "air pollution", "pollution", "weather",
                   "temperature", "how hot", "how cold", "raining", "humidity",
                   "forecast", "climate", "smog"]


GO_OUT_PHRASES = ["should i go", "shall i go", "can i go", "should i head", "shall i head",
                  "is it ok to go", "is it okay to go", "is it good to go", "is it safe to go",
                  "should i step out", "can i step out", "do i go", "is it fine to go",
                  "should i visit", "can i visit", "should i travel", "go out or not",
                  "should we go", "can we go", "is it a good time to go",
                  # Hinglish / Marathi
                  "jana chahiye", "jaana chahiye", "jana chaiye", "ja sakta", "ja sakti",
                  "bahar jana", "bahar jaana", "bahar nikal", "chalna chahiye",
                  "jau kya", "jaun kya", "nikalna chahiye"]

WEATHER_WORDS = ["weather", "aqi", "air quality", "air pollution", "pollution",
                 "temperature", "how hot", "how cold", "how warm", "raining", "rain today",
                 "humidity", "forecast", "smog", "climate",
                 # Hinglish / Marathi
                 "mausam", "mausam kaisa", "havaman", "garmi", "thand", "sardi",
                 "barish", "baarish", "barsaat", "pradushan", "dhuan"]

MORE_PHRASES = ["more places", "show more", "more options", "other places", "other options",
                "any more", "more spots", "aur jagah", "more nearby",
                "suggest more", "recommend more", "give me more", "more suggestions",
                "more locations", "nearby more", "aur dikhao", "aur batao", "aur option",
                "aur jagah batao", "more", "and more", "others"]


BEST_DAY_PHRASES = ["best day", "which day", "what day", "best time this week",
                   "when this week", "which day this week", "best day of the week",
                   "best day this week"]


TASK_PHRASES = ["want to", "wanna", "planning", "plan to", "going to", "where can i",
                "where should i", "suggest", "recommend", "looking for", "help me",
                "things to do", "what can i do", "should i", "is it okay", "is it good",
                "is it safe", "good time", "best time", "best place", "which place",
                "any good spot", "spots for", "place for", "place to"]


def _detect_intent(text: str, activity: str | None) -> str:
    """Decide between a place's conditions and an activity recommendation.

    Default is 'conditions' -- asking about a place (or its weather/AQI) should
    just answer that. A task is only assumed when the user names an activity or
    clearly says they want to do something.
    """
    t = text.lower()
    if any(p in t for p in BEST_DAY_PHRASES):
        return "best_day"
    has_weather = any(w in t for w in WEATHER_WORDS)
    # "should I go ... what's the weather in X" -> a go / no-go question, even
    # when the sentence also contains a noun like "college"
    if has_weather:
        return "go_out" if any(p in t for p in GO_OUT_PHRASES) else "conditions"
    # a genuine activity wins over "can I go <activity>"
    if activity:
        return "recommend"
    if any(p in t for p in GO_OUT_PHRASES):
        return "go_out"
    if any(p in t for p in TASK_PHRASES):
        return "recommend"
    return "conditions"


def detect_more(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in MORE_PHRASES)


# Whole phrases only, and on word boundaries. A plain substring test made
# "Where can I go running?" match on the "here" inside "Where".
MY_LOCATION_RE = re.compile(
    r"\b(?:use my (?:current )?location|using my (?:current )?location|"
    r"my (?:current )?location|current location|near me|around me|nearby me|"
    r"where i am|my place|here)\b")


def wants_my_location(text: str) -> bool:
    return bool(MY_LOCATION_RE.search((text or "").lower()))


def detect_keywords(text: str, slots: dict) -> list:
    """The things we picked out of the user's sentence, for the UI to show."""
    out = []
    t = (text or "").lower()
    if slots.get("activity_label"):
        out.append({"type": "activity", "value": slots["activity_label"]})
    if slots.get("location_text"):
        out.append({"type": "place", "value": slots["location_text"]})
    if slots.get("time_expression"):
        out.append({"type": "time", "value": slots["time_expression"]})
    cond = (slots.get("persona") or {}).get("condition")
    if cond:
        out.append({"type": "persona", "value": cond})
    if slots.get("radius_km"):
        out.append({"type": "radius", "value": "%s km" % slots["radius_km"]})
    if any(w in t for w in WEATHER_WORDS):
        out.append({"type": "weather", "value": "weather / AQI"})
    if (slots.get("intent") == "go_out") or (
            not slots.get("activity") and any(p in t for p in GO_OUT_PHRASES)):
        out.append({"type": "question", "value": "go out or not"})
    if wants_my_location(t):
        out.append({"type": "location", "value": "use my location"})
    return out


CONDITION_HINTS = {
    "asthma": ["asthma", "wheez", "breathing"],
    "senior": ["senior", "elder", "my parents", "grandparent", "grandfather",
               "grandmother", "old age", "65 year", "70 year"],
    "child": ["child", "kid", "toddler", "baby", "my son", "my daughter"],
    "pregnancy": ["pregnan", "expecting"],
}


def _detect_activity(text: str) -> str | None:
    """Match activity words at a word start, so 'row' never fires inside 'tomorrow'.

    Falls back to the broad lexicon so objectives like "movie" or "zorbing" are
    recognised too -- unless the word is part of a place phrase ("near Market Road").
    """
    t = text.lower()
    for act, kws in ACTIVITY_KEYWORDS:
        for kw in kws:
            if re.search(r"\b" + re.escape(kw), t):
                return act
    # Hinglish -- "mujhe cricket khelna hai", "aaj shaam ghumne", "tairne"
    for act, stems in HINGLISH_ACTIVITY:
        for st in stems:
            if st in t:
                return act
    for w in dict.fromkeys(re.findall(r"[a-z]+", t)):
        if not _in_lexicon(w):
            continue
        for mm in re.finditer(r"\b" + re.escape(w) + r"\b", t):
            before = t[max(0, mm.start() - 9):mm.start()]
            # only a real locative preposition disqualifies the word
            # ("near Market Road") -- an article must not ("for a movie")
            if not re.search(r"\b(near|around|beside|by|of|from|in|at|to)\s+"
                             r"(?:the|a|an|my|our|this|that)?\s*$", before):
                return w
    return None


_ACT_STRIP = re.compile(
    r"^(?:for\s+(?:a|an|the)\s+|for\s+|a\s+|an\s+|the\s+|"
    r"i\s+(?:want|wanna|would\s+like)\s+to\s+(?:go\s+)?|"
    r"to\s+go\s+|go\s+|play\s+|do\s+|planning\s+(?:a|an|the)?\s*|plan\s+(?:a|an|the)?\s*)"
)


def activity_label_from_text(text: str) -> str | None:
    """The user's own word for the activity, e.g. 'boating' from a full question."""
    t = text.lower()
    for _act, kws in ACTIVITY_KEYWORDS:
        for kw in kws:
            if re.search(r"\b" + re.escape(kw), t):
                return kw
    for act, stems in HINGLISH_ACTIVITY:
        for st in stems:
            if st in t:
                return FRIENDLY_FAMILY.get(act, act)
    return None


# A broad lexicon of things people go outdoors to do. Used to tell an
# objective ("movie", "zorbing") apart from a place name ("Bandra").
ACTIVITY_LEXICON = {
    # sports & active
    "running", "jogging", "jog", "run", "cycling", "biking", "bike", "bicycle",
    "cricket", "football", "soccer", "futsal", "tennis", "golf", "hockey", "badminton",
    "volleyball", "basketball", "kabaddi", "rugby", "archery", "skating", "skateboarding",
    "swimming", "surfing", "kayaking", "canoeing", "boating", "sailing", "rafting",
    "rowing", "fishing", "diving", "snorkeling", "riding", "hiking", "trekking",
    "camping", "climbing", "mountaineering", "marathon", "athletics", "frisbee",
    "bowling", "boxing", "karate", "judo", "gymnastics", "pilates", "zumba", "aerobics",
    # leisure & objectives
    "movie", "movies", "cinema", "film", "screening", "theatre", "theater",
    "picnic", "barbecue", "bbq", "party", "wedding", "festival", "fair", "event",
    "gathering", "function", "photography", "photoshoot", "painting", "sketching",
    "drawing", "reading", "stargazing", "astronomy", "kite", "birdwatching", "birding",
    "gardening", "shopping", "market", "food", "dining", "dinner", "lunch", "brunch",
    "breakfast", "coffee", "cafe", "sightseeing", "tour", "walk", "walking", "stroll",
    "yoga", "meditation", "exercise", "workout", "games", "sports", "play", "swim",
    "trek", "hike", "camp", "run", "jog", "cycle", "zorbing", "paragliding", "zipline",
    # place categories people ask for directly
    "school", "schools", "college", "colleges", "university", "universities",
    "tuition", "coaching", "classes", "campus", "hospital", "clinic", "temple",
    "mosque", "church", "gurudwara", "atm", "bank", "petrol", "fuel", "hotel",
    "gym", "playground", "restaurant", "pizza", "library",
    "temples", "hospitals", "banks", "atms", "hotels", "gyms", "playgrounds",
    "restaurants", "clinics", "cafes", "markets", "malls", "cinemas", "theatres",
    "schools", "colleges", "universities", "parks", "lakes", "grounds", "courts",
    "monuments", "landmarks", "heritage", "forts", "palaces", "museums",
    # Hinglish
    "ghumna", "ghumne", "ghoomna", "tahalna", "sair", "daudna", "daud",
    "khelna", "khelne", "tairna", "padhna", "padhne", "khana", "khaana",
    "tasveer", "shaadi", "mela", "dhyan",
}


def _in_lexicon(word: str) -> bool:
    if word in ACTIVITY_LEXICON:
        return True
    if word.endswith("s") and word[:-1] in ACTIVITY_LEXICON:
        return True
    if word.endswith("es") and word[:-2] in ACTIVITY_LEXICON:
        return True
    return False


def is_lexicon_activity(text: str) -> bool:
    t = re.sub(r"[^a-z ]", " ", text.lower())
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return False
    if _in_lexicon(t):
        return True
    words = t.split()
    return len(words) <= 3 and any(_in_lexicon(w) for w in words)


def ends_in_ing(text: str) -> bool:
    words = re.sub(r"[^a-z ]", " ", text.lower()).split()
    return bool(words) and len(words[-1]) > 4 and words[-1].endswith("ing")


def clean_activity(text: str) -> str:
    t = text.strip().rstrip(".!?").lower()
    t = _ACT_STRIP.sub("", t).strip()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or text.strip()


def looks_like_activity(text: str) -> bool:
    """True when a short chat reply is probably an activity name."""
    t = text.strip().lower().rstrip(".!?")
    if not t or len(t) > 40:
        return False
    if any(w in t for w in _QUESTION_WORDS):
        return False
    if re.search(r"\d", t):
        return False
    words = [w for w in re.split(r"[ ,]+", t) if w]
    if not (1 <= len(words) <= 4):
        return False
    if any(w in _STOP_PLACE_WORDS for w in words):
        return False
    return True


def _detect_condition(text: str) -> str | None:
    t = text.lower()
    for cond, kws in CONDITION_HINTS.items():
        if any(k in t for k in kws):
            return cond
    return normalise_condition(t)


def _detect_age(text: str) -> int | None:
    m = re.search(r"\b(\d{1,3})\s*(?:year|yr|y/o)", text.lower())
    if m:
        v = int(m.group(1))
        if 0 < v < 120:
            return v
    return None


def _detect_radius(text: str) -> float | None:
    m = re.search(r"within\s+(\d+(?:\.\d+)?)\s*km", text.lower())
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*km\s+(?:of|from|around|near)", text.lower())
    if m:
        return float(m.group(1))
    return None


BARE_PLACE_HINT = re.compile(
    r"^[A-Za-z][A-Za-z .'\-]{2,40}$")


LOCATION_WORDS = ("location", "gps", "geo", "here", "my place", "current place",
                  "where i am", "where i'm", "around me", "near me")


def looks_like_bare_place(text: str) -> bool:
    """A short message with no verb that is probably just a place name."""
    t = (text or "").strip().rstrip("?.!")
    low0 = t.lower()
    # "use my current location" is a request, not a place
    if wants_my_location(t) or any(w in low0 for w in LOCATION_WORDS):
        return False
    if not t or len(t.split()) > 5:
        return False
    if not BARE_PLACE_HINT.match(t):
        return False
    low = t.lower()
    if any(w in low for w in WEATHER_WORDS):
        return False
    words = re.findall(r"[a-z]+", low)
    if any(_in_lexicon(w) for w in words):
        return False
    return True


# a way of travelling, which is what makes a question a journey
ROUTE_MODES = [
    ("cycling", ["cycle", "cycling", "bike", "biking", "bycycle", "pedal",
                 "scooter", "two-wheeler", "motorcycle", "motorbike"]),
    ("walking", ["walk", "walking", "on foot", "by foot"]),
    ("driving", ["drive", "driving", "car", "cab", "taxi", "commute by car"]),
]
# "to work" and friends name a routine, not a place we can find
ROUTE_VAGUE = {"work", "office", "college", "school", "home", "the office",
               "the gym", "workplace", "tuition", "class"}

FROM_TO = re.compile(r"\bfrom\s+(?P<a>[A-Za-z][A-Za-z .,'\-]{2,40}?)\s+to\s+"
                     r"(?P<b>[A-Za-z][A-Za-z .,'\-]{2,40}?)(?=\s+(?:at|by|on|in|around|"
                     r"tomorrow|today|tonight|this|next)\b|[?.!,]|$)", re.I)
X_TO_Y = re.compile(r"\b(?P<a>[A-Za-z][A-Za-z .,'\-]{2,40}?)\s+to\s+"
                    r"(?P<b>[A-Za-z][A-Za-z .,'\-]{2,40}?)(?=\s+(?:at|by|on|in|around|"
                    r"tomorrow|today|tonight|this|next)\b|[?.!,]|$)", re.I)


def _route_mode(text: str) -> str | None:
    low = (text or "").lower()
    for name, words in ROUTE_MODES:
        if any(re.search(r"\b" + re.escape(w), low) for w in words):
            return name
    return None


_LEAD_JUNK = re.compile(
    r"^(?:is it|it is|would it be|would it|can i|could i|should i|do i|will i|"
    r"i want to|i want|i|we|you|okay to|ok to|fine to|safe to|good to|okay|fine|"
    r"want|to|go)\s+", re.I)
_LEAD_VERB = re.compile(
    r"^(?:go\s+)?(?:cycle|cycling|walk|walking|drive|driving|ride|riding|bike|"
    r"biking|pedal|pedalling|commute|commuting)\s+", re.I)


def _clean_route_side(s: str | None) -> str | None:
    """A start or end point, with the question's scaffolding stripped off."""
    t = (s or "").strip(" .,")
    for _ in range(3):
        before = t
        t = _LEAD_JUNK.sub("", t).strip(" .,")
        t = _LEAD_VERB.sub("", t).strip(" .,")
        if t == before:
            break
    if len(t) < 3:
        return None
    low = t.lower()
    # scaffolding words left on their own are not a place
    if low in ("okay", "ok", "fine", "safe", "good", "it", "is", "want", "to", "go",
               "a", "the", "maybe", "sure", "possible", "advisable", "worth"):
        return None
    # a way of travelling, or a vague "somewhere", is not a place
    if any(re.search(r"\b" + re.escape(w), low)
           for _n, ws in ROUTE_MODES for w in ws):
        return None
    if any(w in low for w in ("near ", "around ", "somewhere", "anywhere",
                              " a place", " a spot", "good spot")):
        return None
    return t


def detect_route(text: str) -> dict | None:
    """
    A journey question: how are we travelling, from where, to where.

    Returns None when the message is not a journey at all, so ordinary activity
    questions are untouched. When someone says "cycle to work" there is no place to
    find, so both ends come back None and the assistant asks for them.
    """
    mode = _route_mode(text)
    if mode is None:
        return None
    t = (text or "").strip()
    m = FROM_TO.search(t) or X_TO_Y.search(t)
    if not m:
        return None

    a_raw = m.group("a").strip(" .,")
    b_raw = m.group("b").strip(" .,")
    a, b = _clean_route_side(a_raw), _clean_route_side(b_raw)

    words = re.findall(r"[a-z]+", (a_raw + " " + b_raw).lower())
    routine = any(w in ROUTE_VAGUE for w in words)

    if not a and not b:
        return {"origin": None, "destination": None, "mode": mode} if routine else None

    if b and b.lower() in ROUTE_VAGUE:
        b = None
    if a and a.lower() in ROUTE_VAGUE:
        a = None
    if not a and not b:
        return {"origin": None, "destination": None, "mode": mode}
    return {"origin": a, "destination": b, "mode": mode}


def _detect_location(text: str) -> str | None:
    t = text.strip().rstrip("?.!")
    patterns = [
        r"\d+(?:\.\d+)?\s*km\s+of\s+([A-Za-z][A-Za-z .'\-]+)",
        r"\b(?:near|around|beside|by|about)\s+([A-Za-z][A-Za-z .'\-]+)",
        r"\bof\s+([A-Za-z][A-Za-z .'\-]+)",
        r"\bin\s+([A-Za-z][A-Za-z .'\-]+)",
        r"\bat\s+([A-Za-z][A-Za-z .'\-]+)",
    ]
    for pat in patterns:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            loc = m.group(1).strip()
            loc = re.sub(r"\s+\b(?:running|cycling|walking|jogging|hiking|playing|going|"
                         r"swimming|cricket|football|tennis|golf|picnic|movie|cinema)\b.*$",
                         "", loc, flags=re.I).strip()
            loc = re.sub(r"\b(on|in|at|to|the|of|within|around|near|by|this|next|for|"
                         r"today|tomorrow|tonight|now|currently|right|there|here|outside|use|my|location|"
                         r"aqi|air|quality|weather|forecast|temperature|humidity|"
                         r"day|days|two|three|saturday|sunday|"
                         r"monday|tuesday|wednesday|thursday|friday|weekend|"
                         r"morning|evening|afternoon|night)\b.*$", "", loc, flags=re.I).strip()
            # a place name ends where the next instruction begins
            loc = re.split(r"\s+\b(?:with|and|plus|including|along|for|"
                           r"whose|that|which|to\s+see|to\s+visit)\b",
                           loc, 1, flags=re.I)[0]
            loc = loc.strip(" ,.")
            if loc.lower() in PRONOUN_PLACES:
                return None            # "near me" is not a place name
            if len(loc) >= 2:
                return loc

    # Hinglish places come BEFORE the marker: "powai me", "thane ke paas"
    m = re.search(r"([A-Za-z][A-Za-z .'\-]{1,60}?)\s+"
                  r"(?:ke\s+paas|ke\s+aas\s*-?\s*paas|ke\s+aas|mein|me|mai|par|pe)\b",
                  t, re.IGNORECASE)
    if m:
        words = [w for w in re.split(r"\s+", m.group(1)) if w]
        words = [w for w in words
                 if w.lower() not in HINGLISH_STOP and w.lower() not in ENGLISH_BLOCK]
        if words:
            prefix = {"navi", "greater", "new", "old", "east", "west", "north",
                      "south", "upper", "lower", "uttar", "dakshin", "purba", "paschim"}
            take = 2 if len(words) >= 2 and words[-2].lower() in prefix else 1
            cand = " ".join(words[-take:]).strip(" ,.")
            if cand.lower() not in PRONOUN_PLACES and len(cand) >= 2:
                return cand
    return None


_QUESTION_WORDS = ("is ", "can ", "should ", "where", "what", "when", "how",
                   "okay", "ok ", "good", "best", "?", "location", "current", "use my")
_STOP_PLACE_WORDS = {"tomorrow", "today", "morning", "evening", "night", "weekend",
                     "running", "cycling", "walk", "cricket", "tennis", "golf", "picnic"}


def clean_place(text: str) -> str:
    return text.strip().strip(" .,?!")


def looks_like_place(text: str) -> bool:
    """True when a short message is probably just a place name (a chat reply)."""
    t = text.strip().lower()
    if not t or len(t) > 48:
        return False
    if any(w in t for w in _QUESTION_WORDS):
        return False
    if re.search(r"\d", t):                      # avoids "8 am", "15 km"
        return False
    if not re.match(r"^[a-z][a-z .,'\-]+$", t):
        return False
    words = [w for w in re.split(r"[ ,]+", t) if w]
    if not (1 <= len(words) <= 5):
        return False
    if any(w in _STOP_PLACE_WORDS for w in words):
        return False
    return True


def _detect_time(text: str) -> str | None:
    """Combine a day reference with a part-of-day (and any clock time)."""
    t = text.lower()
    days = ["day after tomorrow", "tomorrow", "today", "tonight", "this weekend",
            "weekend", "next two days", "next 2 days", "next three days",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            # Hinglish
            "parso", "kal", "aaj", "abhi", "agle hafte"]
    parts_of_day = ["morning", "afternoon", "evening", "night",
                    "subah", "dopahar", "shaam", "sham", "raat"]
    day = next((d for d in days if d in t), None)
    part = next((p for p in parts_of_day if p in t), None)
    clock = re.search(r"\b\d{1,2}\s*(?::\d{2})?\s*(?:am|pm)\b", t)
    if not clock:
        clock = re.search(r"\b(?:at|around|by|near)\s+(\d{1,2})(?:\s*(?:am|pm))?\b", t)
    bits = [x for x in (day, part) if x]
    if clock:
        bits.append(clock.group(0))
    return " ".join(bits) if bits else None


def rule_parse(question: str) -> dict:
    """Deterministic fallback parser."""
    activity = _detect_activity(question)
    return {
        "intent": _detect_intent(question, activity),
        "activity": activity,
        "location_text": _detect_location(question),
        "radius_km": _detect_radius(question),
        "time_expression": _detect_time(question),
        "persona": {"age": _detect_age(question), "condition": _detect_condition(question)},
        "needs_clarification": False,
        "clarification_question": None,
        "parser": "rules",
    }


def parse_query(question: str, lat: float | None = None, lon: float | None = None,
                location_hint: str | None = None, activity_hint: str | None = None) -> dict:
    """
    Best-effort slot extraction. Uses the SLM when enabled, always falls back
    to rules, and merges so a partial SLM answer is completed deterministically.
    """
    slots = None
    if ENABLE_SLM:
        try:
            from .slm import parse_slots
            slots = parse_slots(question)
        except Exception:
            slots = None

    rule = rule_parse(question)
    if not slots:
        slots = dict(rule)
        slots["parser"] = "rules"
    else:
        # complete / correct the SLM answer with deterministic signals
        slots.setdefault("persona", {})
        if not slots.get("activity"):
            slots["activity"] = rule["activity"]
        if not slots.get("location_text"):
            slots["location_text"] = rule["location_text"]
        if slots.get("radius_km") is None:
            slots["radius_km"] = rule["radius_km"]
        if not slots.get("time_expression"):
            slots["time_expression"] = rule["time_expression"]
        if not slots["persona"].get("condition"):
            slots["persona"]["condition"] = rule["persona"]["condition"]
        if slots["persona"].get("age") is None:
            slots["persona"]["age"] = rule["persona"]["age"]
        slots["parser"] = "slm+rules"

    # intent is always decided deterministically
    slots["intent"] = rule["intent"]

    # a follow-up chat reply may have supplied the missing place
    if not slots.get("location_text") and location_hint:
        slots["location_text"] = location_hint

    # ...or the missing activity (which may be one we do not have a keyword for)
    if not slots.get("activity") and activity_hint:
        slots["activity"] = activity_hint

    # if we have a live GPS fix, we do not need a location string
    if not slots.get("location_text") and lat is None:
        slots["needs_clarification"] = True
        slots.setdefault("clarification_question",
                         "Which area should I look at? A landmark, neighbourhood or city works.")

    # default radius
    if slots.get("radius_km") is None:
        slots["radius_km"] = DEFAULT_RADIUS_KM

    # out-of-scope guard -- but a plain weather/AQI question needs no activity
    if not slots.get("activity") and slots.get("intent") not in ("conditions", "go_out", "best_day"):
        slots["needs_clarification"] = True
        slots.setdefault("clarification_question",
                         "Which outdoor activity are you planning? "
                         "For example running, cycling, cricket, a picnic or a photo walk.")

    return slots
