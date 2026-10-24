"""
Drishti data layer -- live, free, open sources, fetched at query time.

Key robustness + speed features:
  * Overpass is tried across several public mirrors, with a Nominatim-based
    fallback so nearby places are found even when one endpoint is down.
  * Fetches run in parallel (ThreadPoolExecutor), so a query is not a long
    chain of sequential HTTP calls.
  * Every failure is recorded (see last_error) and surfaced as a data gap
    rather than silently returning nothing.
  * Responses are cached so repeat queries are instant and rate limits are
    respected.
"""
from __future__ import annotations

import json
import os
import re
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import CACHE_TTL_SECONDS, HTTP_TIMEOUT, USER_AGENT

# ---------------------------------------------------------------- transport
# Two things break outbound calls on a container that a laptop never hits.
#
# 1. IPv6. Several of the hosts below publish an AAAA record
#    (air-quality-api.open-meteo.com, overpass-api.de, nominatim all do).
#    A container with no IPv6 route tries the v6 address first, the connect
#    fails, and urllib does NOT fall back to the v4 address the way a browser
#    does. So we resolve v4 first. Set DRISHTI_FORCE_IPV4=0 to turn it off.
#
# 2. Proxy environment variables. urllib honours HTTP_PROXY / HTTPS_PROXY /
#    ALL_PROXY from the environment. A stray one on the host silently reroutes
#    or blocks the request. We build our own opener that ignores them.
#
# Both are cheap and neither changes the result on a machine that was working.
_FORCE_IPV4 = os.environ.get("DRISHTI_FORCE_IPV4", "1").strip() not in ("0", "false", "no")
_orig_getaddrinfo = socket.getaddrinfo

if _FORCE_IPV4:
    def _getaddrinfo_v4(host, port, family=0, type=0, proto=0, flags=0):
        infos = _orig_getaddrinfo(host, port, family, type, proto, flags)
        v4 = [i for i in infos if i[0] == socket.AF_INET]
        return v4 or infos          # fall back if a host really is v6 only
    socket.getaddrinfo = _getaddrinfo_v4

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

RETRY_STATUS = {429, 500, 502, 503, 504}
# One retry by default. Retrying a hard failure three times turns a slow source
# into a hung request, which is worse than the failure itself.
RETRY_ATTEMPTS = int(os.environ.get("DRISHTI_HTTP_RETRIES", "2"))
def _hopeless(e: BaseException) -> bool:
    """
    Will another attempt plausibly help? A name that does not resolve, a port
    that refuses, or a bad certificate will fail exactly the same way, so those
    are raised at once. urllib wraps these in URLError, hence the unwrapping.
    """
    seen = e
    for _ in range(3):
        if isinstance(seen, (socket.gaierror, ConnectionRefusedError, ssl.SSLError)):
            return True
        nxt = getattr(seen, "reason", None)
        if nxt is None or nxt is seen:
            break
        seen = nxt
    return False

_CACHE: dict[str, tuple[float, object]] = {}
_LAST_ERROR: dict[str, str] = {}

SOURCES = [
    "Open-Meteo Forecast API (weather)",
    "Open-Meteo Air-Quality API (PM2.5/PM10/AQI/pollen)",
    "OpenStreetMap via Overpass API (places, GIS features)",
    "Nominatim (geocoding)",
    "OSRM (routing / travel time)",
    "OpenTopoData (elevation)",
    "WAQI (official CPCB-backed AQI, optional)",
    "IMD API (official nowcast, optional)",
    "Map data (c) OpenStreetMap contributors",
]

# several public Overpass mirrors -> failover instead of a single point of failure
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.osm.jp/api/interpreter",
]

# major Indian cities for the landing dashboard
MAJOR_CITIES = [
    ("Mumbai", 19.0760, 72.8777),
    ("Delhi", 28.6139, 77.2090),
    ("Bengaluru", 12.9716, 77.5946),
    ("Chennai", 13.0827, 80.2707),
    ("Kolkata", 22.5726, 88.3639),
    ("Hyderabad", 17.3850, 78.4867),
    ("Pune", 18.5204, 73.8567),
    ("Ahmedabad", 23.0225, 72.5714),
]


# ---------------------------------------------------------------- plumbing
def _cached(key: str, ttl: int = CACHE_TTL_SECONDS):
    hit = _CACHE.get(key)
    if hit and (time.time() - hit[0]) < ttl:
        return hit[1]
    return None


def _store(key: str, value):
    _CACHE[key] = (time.time(), value)
    return value


def _fetch(url: str, headers: dict | None = None, timeout: int | None = None,
           data: bytes | None = None):
    """One attempt. Raises on anything that is not a 2xx."""
    req = urllib.request.Request(url, data=data,
                                 headers={"User-Agent": USER_AGENT, **(headers or {})})
    with _OPENER.open(req, timeout=timeout or HTTP_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def _get_json(url: str, headers: dict | None = None, timeout: int | None = None):
    """
    A GET that survives a flaky moment. A 429 or a 5xx is retried with a short
    backoff, because rate limits on the shared egress of a free host are the
    usual reason a call that works locally fails in production. The error that
    finally escapes carries the status code, so it is readable in the logs.
    """
    last: Exception | None = None
    for i in range(max(1, RETRY_ATTEMPTS)):
        try:
            return _fetch(url, headers, timeout)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", "replace")[:200]
            except Exception:  # noqa: BLE001
                pass
            last = RuntimeError(f"HTTP {e.code} from {urllib.parse.urlsplit(url).netloc}"
                                + (f": {body}" if body else ""))
            if e.code not in RETRY_STATUS or i == RETRY_ATTEMPTS - 1:
                raise last from e
        except Exception as e:  # noqa: BLE001  (URLError, timeout, TLS)
            if _hopeless(e) or i == RETRY_ATTEMPTS - 1:
                raise
            last = e
        time.sleep(0.4 * (2 ** i))
    raise last if last else RuntimeError("request failed")


def _post_json(url: str, data: bytes, headers: dict | None = None, timeout: int | None = None):
    return _fetch(url, headers, timeout, data=data)


def parallel(fn, items: list, max_workers: int = 8) -> list:
    """Run fn over items concurrently, preserving order. Failures become None."""
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as ex:
        return list(ex.map(lambda x: _safe(fn, x), items))


def _safe(fn, arg):
    try:
        return fn(arg)
    except Exception:
        return None


def last_error(name: str) -> str | None:
    return _LAST_ERROR.get(name)


# ---------------------------------------------------------------- geocoding
# old / common names -> the name the geocoders know best
CITY_ALIASES = {
    "bombay": "Mumbai", "calcutta": "Kolkata", "madras": "Chennai",
    "bangalore": "Bengaluru", "mysore": "Mysuru", "poona": "Pune",
    "trivandrum": "Thiruvananthapuram", "cochin": "Kochi", "ernakulam": "Kochi",
    "benares": "Varanasi", "banaras": "Varanasi", "simla": "Shimla",
    "gurgaon": "Gurugram", "allahabad": "Prayagraj", "pondicherry": "Puducherry",
    "baroda": "Vadodara", "cawnpore": "Kanpur", "calicut": "Kozhikode",
    "travancore": "Thiruvananthapuram", "goa": "Goa", "bombai": "Mumbai",
}


def _normalise_place(place: str) -> str:
    """Replace old Indian city names with the current ones (Bombay -> Mumbai)."""
    return re.sub(r"[A-Za-z]+",
                  lambda m: CITY_ALIASES.get(m.group(0).lower(), m.group(0)),
                  place)


def _geocode_openmeteo(place: str, country_code: str | None = "IN"):
    """Open-Meteo geocoding -- usually the fastest of the two."""
    params = {"name": place, "count": 1, "language": "en", "format": "json"}
    if country_code:
        params["countryCode"] = country_code
    d = _get_json("https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(params),
                  timeout=8)
    res = d.get("results") or []
    if not res:
        return None
    r = res[0]
    label = ", ".join([x for x in [r.get("name"), r.get("admin1"), r.get("country")] if x])
    return {"lat": float(r["latitude"]), "lon": float(r["longitude"]), "display_name": label}


def _geocode_nominatim(place: str, country: str | None = "India", country_code: str | None = "in"):
    params = {"q": f"{place}, {country}" if country else place, "format": "json", "limit": 1}
    if country_code:
        params["countrycodes"] = country_code
    data = _get_json("https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(params),
                     timeout=10)
    if not data:
        return None
    h = data[0]
    return {"lat": float(h["lat"]), "lon": float(h["lon"]), "display_name": h["display_name"]}


def _race_geocode(place: str, country: str | None, country_code: str | None):
    """Run both geocoders in parallel and take the first answer."""
    ex = ThreadPoolExecutor(max_workers=2)
    futs = [ex.submit(_geocode_openmeteo, place, country_code),
            ex.submit(_geocode_nominatim, place, country, country_code)]
    result, err = None, None
    for f in as_completed(futs):
        try:
            r = f.result()
        except Exception as e:  # noqa: BLE001
            err = e
            continue
        if r:
            result = r
            break
    ex.shutdown(wait=False)          # do not block on the slower geocoder
    return result, err


def geocode(place: str, country: str = "India"):
    """
    Geocode a place, India-first. Old Indian names are normalised
    (Bombay -> Mumbai), both geocoders are restricted to India, and only if
    that finds nothing do we retry worldwide.
    """
    key = f"geo:{place}:{country}"
    if (c := _cached(key)) is not None:
        return c

    normalised = _normalise_place(place)
    result, err = _race_geocode(normalised, country, "in")
    if not result:
        result, err = _race_geocode(normalised, None, None)   # global fallback

    if result is None:
        _LAST_ERROR["geocode"] = str(err) if err else f"no match for '{place}'"
    else:
        _LAST_ERROR.pop("geocode", None)
    return _store(key, result)


def reverse_geocode(lat: float, lon: float, timeout: int = 4):
    """A human name for a GPS fix. Deliberately short-timeout: it is only a
    label, so it must never hold up the answer."""
    key = f"rgeo:{round(lat, 3)}:{round(lon, 3)}"
    if (c := _cached(key, ttl=24 * 3600)) is not None:
        return c
    params = {"lat": lat, "lon": lon, "format": "json", "zoom": 12}
    try:
        d = _get_json("https://nominatim.openstreetmap.org/reverse?" + urllib.parse.urlencode(params),
                      timeout=timeout)
        return _store(key, d.get("display_name"))
    except Exception:
        return _store(key, None)


# ---------------------------------------------------------------- weather / AQ
HOURLY_VARS = ["temperature_2m", "apparent_temperature", "relative_humidity_2m",
               "precipitation_probability", "precipitation", "wind_speed_10m",
               "wind_direction_10m", "wind_gusts_10m", "uv_index", "visibility",
               "cloud_cover", "is_day"]
DAILY_VARS = ["sunrise", "sunset", "uv_index_max", "precipitation_probability_max"]


def _weather_fallback_wttr(lat: float, lon: float, days: int = 3):
    """Fallback provider used when Open-Meteo is rate-limited/unavailable.

    wttr.in returns a different schema, so normalize the useful values to the
    hourly/daily shape Drishti already consumes. Missing fields are left empty
    rather than fabricated.
    """
    url = f"https://wttr.in/{lat:.4f},{lon:.4f}?format=j1"
    raw = _get_json(url, headers={"Accept": "application/json"}, timeout=18)
    current = (raw.get("current_condition") or [{}])[0]
    forecasts = (raw.get("weather") or [])[:max(1, min(int(days), 7))]
    hourly = {name: [] for name in HOURLY_VARS}
    times = []
    for day in forecasts:
        date = day.get("date", "")
        for hour in day.get("hourly", []):
            h = str(hour.get("time", "0")).zfill(4)
            times.append(f"{date}T{h[:2]}:{h[2:4]}")
            mappings = {
                "temperature_2m": hour.get("tempC"),
                "apparent_temperature": hour.get("FeelsLikeC"),
                "relative_humidity_2m": hour.get("humidity"),
                "precipitation_probability": hour.get("chanceofrain"),
                "precipitation": hour.get("precipMM"),
                "wind_speed_10m": hour.get("windspeedKmph"),
                "wind_direction_10m": hour.get("winddirDegree"),
                "wind_gusts_10m": hour.get("WindGustKmph"),
                "uv_index": hour.get("uvIndex"),
                "visibility": hour.get("visibility"),
                "cloud_cover": hour.get("cloudcover"),
                "is_day": 0 if h.startswith(("00", "01", "02", "03", "04", "05", "18", "19", "20", "21", "22", "23")) else 1,
            }
            for name, value in mappings.items():
                try: hourly[name].append(float(value) if value not in (None, "") else None)
                except (TypeError, ValueError): hourly[name].append(None)
    daily = {name: [] for name in DAILY_VARS}
    daily_times = []
    for day in forecasts:
        daily_times.append(day.get("date"))
        astro = (day.get("astronomy") or [{}])[0]
        h0 = (day.get("hourly") or [{}])[0]
        for name, value in {
            "sunrise": astro.get("sunrise"), "sunset": astro.get("sunset"),
            "uv_index_max": day.get("uvIndex"),
            "precipitation_probability_max": max([int(x.get("chanceofrain", 0) or 0) for x in day.get("hourly", [])] or [0]),
        }.items():
            daily[name].append(value)
    try: temp = float(current.get("temp_C"))
    except (TypeError, ValueError): temp = None
    return {
        "latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata",
        "current": {"temperature_2m": temp, "relative_humidity_2m": current.get("humidity"),
                    "apparent_temperature": current.get("FeelsLikeC"), "wind_speed_10m": current.get("windspeedKmph"),
                    "is_day": 1 if current.get("isDayTime", "yes") == "yes" else 0},
        "hourly": {"time": times, **hourly}, "daily": {"time": daily_times, **daily},
        "drishti_provider": "wttr.in fallback",
    }


def weather(lat: float, lon: float, days: int = 3):
    key = f"wx:{round(lat, 3)}:{round(lon, 3)}:{days}"
    if (c := _cached(key)) is not None:
        return c
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(HOURLY_VARS),
              "daily": ",".join(DAILY_VARS), "forecast_days": days, "timezone": "Asia/Kolkata"}
    try:
        result = _get_json("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params))
        return _store(key, result)
    except Exception as primary_error:
        _LAST_ERROR["weather"] = f"Open-Meteo failed: {type(primary_error).__name__}: {primary_error}"
        print(f"[drishti] Open-Meteo weather failed; trying fallback: {primary_error}", flush=True)
        try:
            result = _weather_fallback_wttr(lat, lon, days)
            _LAST_ERROR.pop("weather", None)
            return _store(key, result)
        except Exception as fallback_error:
            _LAST_ERROR["weather"] += f"; wttr.in fallback failed: {type(fallback_error).__name__}: {fallback_error}"
            print(f"[drishti] weather fallback failed: {fallback_error}", flush=True)
            return None


def air_quality(lat: float, lon: float, days: int = 3):
    key = f"aq:{round(lat, 3)}:{round(lon, 3)}:{days}"
    if (c := _cached(key)) is not None:
        return c
    params = {"latitude": lat, "longitude": lon,
              "hourly": "pm10,pm2_5,us_aqi,european_aqi,alder_pollen,birch_pollen,grass_pollen",
              "forecast_days": days, "timezone": "Asia/Kolkata"}
    try:
        return _store(key, _get_json("https://air-quality-api.open-meteo.com/v1/air-quality?" + urllib.parse.urlencode(params)))
    except Exception as e:
        _LAST_ERROR["air_quality"] = f"{type(e).__name__}: {e}"
        print(f"[drishti] air quality fetch failed: {type(e).__name__}: {e}", flush=True)
        return None


# ---------------------------------------------------------------- places / GIS
ACTIVITY_FILTERS = {
    "running": 'nwr["leisure"~"park|track|pitch"]',
    "cycling": 'nwr["highway"~"cycleway|path"]',
    "golf": 'nwr["leisure"="golf_course"]',
    "cricket": 'nwr["sport"~"cricket"]',
    "football": 'nwr["sport"~"soccer|football"]',
    "tennis": 'nwr["sport"="tennis"]',
    "hiking": 'nwr["highway"~"path|track"]',
    "picnic": 'nwr["leisure"~"park|garden"]',
    "event": 'nwr["leisure"~"park|common|garden"]',
    "photo": 'nwr["tourism"~"viewpoint"]',
    # extra families (any other activity resolves to one of these)
    "watersport": 'nwr["natural"="water"]',
    "field": 'nwr["leisure"~"pitch|sports_centre|track"]',
    "gentle": 'nwr["leisure"~"park|garden"]',
}
# plain-language search terms for the Nominatim fallback
ACTIVITY_SEARCH = {
    "running": "park", "cycling": "park", "golf": "golf course",
    "cricket": "cricket ground", "football": "football ground",
    "tennis": "tennis court", "hiking": "park", "picnic": "park",
    "event": "park", "photo": "viewpoint",
    "watersport": "lake", "field": "ground", "gentle": "park",
}

# ordered fallbacks -- a generic term like "lake" often returns nothing from
# Nominatim's bounded search, so we try a few before giving up
ACTIVITY_SEARCH_ALT = {
    "watersport": ["lake", "water body", "river", "park"],
    "field": ["ground", "park"],
    "gentle": ["park", "garden"],
    "running": ["park"],
    "cycling": ["park"],
    "golf": ["golf course", "park"],
    "cricket": ["cricket ground", "ground", "park"],
    "football": ["football ground", "ground", "park"],
    "tennis": ["tennis court", "park"],
    "hiking": ["park", "hill"],
    "picnic": ["park"],
    "event": ["park"],
    "photo": ["viewpoint", "park"],
}


# Some objectives need their own kind of venue, not a park. These take priority
# over the comfort-family filter when the user names such an objective.
OBJECT_FILTERS = {
    "movie": ('nwr["amenity"~"cinema|theatre"]|OR|nwr["shop"~"mall|department_store"]'
              '|OR|nwr["amenity"="cinema"]'),
    "monument": ('nwr["tourism"~"attraction|museum|artwork"]|OR|nwr["historic"]'
                 '|OR|nwr["heritage"]'),
    "movies": 'nwr["amenity"="cinema"]',
    "cinema": 'nwr["amenity"="cinema"]',
    "film": 'nwr["amenity"="cinema"]',
    "screening": 'nwr["amenity"="cinema"]',
    "theatre": 'nwr["amenity"="cinema"]',
    "theater": 'nwr["amenity"="cinema"]',
    "shopping": 'nwr["shop"~"mall|department_store"]',
    "mall": 'nwr["shop"="mall"]',
    "bazaar": 'nwr["amenity"="marketplace"]',
    "market": 'nwr["amenity"="marketplace"]',
    "food": 'nwr["amenity"~"restaurant|cafe"]',
    "dining": 'nwr["amenity"~"restaurant|cafe"]',
    "dinner": 'nwr["amenity"~"restaurant|cafe"]',
    "lunch": 'nwr["amenity"~"restaurant|cafe"]',
    "brunch": 'nwr["amenity"~"restaurant|cafe"]',
    "breakfast": 'nwr["amenity"~"restaurant|cafe"]',
    "coffee": 'nwr["amenity"="cafe"]',
    "cafe": 'nwr["amenity"="cafe"]',
    "reading": 'nwr["amenity"="library"]',
    # place categories people ask for directly
    "school": 'nwr["amenity"~"school|college|university"]',
    "college": 'nwr["amenity"~"college|university"]',
    "university": 'nwr["amenity"="university"]',
    "tuition": 'nwr["amenity"="school"]',
    "coaching": 'nwr["amenity"="school"]',
    "hospital": 'nwr["amenity"="hospital"]',
    "clinic": 'nwr["amenity"~"clinic|doctors"]',
    "temple": 'nwr["amenity"="place_of_worship"]',
    "mosque": 'nwr["amenity"="place_of_worship"]',
    "church": 'nwr["amenity"="place_of_worship"]',
    "gurudwara": 'nwr["amenity"="place_of_worship"]',
    "atm": 'nwr["amenity"="atm"]',
    "bank": 'nwr["amenity"="bank"]',
    "petrol": 'nwr["amenity"="fuel"]',
    "fuel": 'nwr["amenity"="fuel"]',
    "hotel": 'nwr["tourism"="hotel"]',
    "gym": 'nwr["leisure"="fitness_centre"]',
    "playground": 'nwr["leisure"="playground"]',
    "restaurant": 'nwr["amenity"="restaurant"]',
    "pizza": 'nwr["amenity"="restaurant"]',
}
OBJECT_MATCHERS = {
    "movie": lambda t: (t.get("amenity") in ("cinema", "theatre")
                        or t.get("shop") in ("mall", "department_store")),
    "monument": lambda t: (t.get("tourism") in ("attraction", "museum", "artwork")
                           or bool(t.get("historic")) or bool(t.get("heritage"))),
    "movies": lambda t: t.get("amenity") == "cinema",
    "cinema": lambda t: t.get("amenity") == "cinema",
    "film": lambda t: t.get("amenity") == "cinema",
    "screening": lambda t: t.get("amenity") == "cinema",
    "theatre": lambda t: t.get("amenity") == "cinema",
    "theater": lambda t: t.get("amenity") == "cinema",
    "shopping": lambda t: t.get("shop") in ("mall", "department_store"),
    "mall": lambda t: t.get("shop") == "mall",
    "bazaar": lambda t: t.get("amenity") == "marketplace",
    "market": lambda t: t.get("amenity") == "marketplace",
    "food": lambda t: t.get("amenity") in ("restaurant", "cafe"),
    "dining": lambda t: t.get("amenity") in ("restaurant", "cafe"),
    "dinner": lambda t: t.get("amenity") in ("restaurant", "cafe"),
    "lunch": lambda t: t.get("amenity") in ("restaurant", "cafe"),
    "brunch": lambda t: t.get("amenity") in ("restaurant", "cafe"),
    "breakfast": lambda t: t.get("amenity") in ("restaurant", "cafe"),
    "coffee": lambda t: t.get("amenity") == "cafe",
    "cafe": lambda t: t.get("amenity") == "cafe",
    "reading": lambda t: t.get("amenity") == "library",
    "school": lambda t: t.get("amenity") in ("school", "college", "university"),
    "college": lambda t: t.get("amenity") in ("college", "university"),
    "university": lambda t: t.get("amenity") == "university",
    "tuition": lambda t: t.get("amenity") == "school",
    "coaching": lambda t: t.get("amenity") == "school",
    "hospital": lambda t: t.get("amenity") == "hospital",
    "clinic": lambda t: t.get("amenity") in ("clinic", "doctors"),
    "temple": lambda t: t.get("amenity") == "place_of_worship",
    "mosque": lambda t: t.get("amenity") == "place_of_worship",
    "church": lambda t: t.get("amenity") == "place_of_worship",
    "gurudwara": lambda t: t.get("amenity") == "place_of_worship",
    "atm": lambda t: t.get("amenity") == "atm",
    "bank": lambda t: t.get("amenity") == "bank",
    "petrol": lambda t: t.get("amenity") == "fuel",
    "fuel": lambda t: t.get("amenity") == "fuel",
    "hotel": lambda t: t.get("tourism") == "hotel",
    "gym": lambda t: t.get("leisure") == "fitness_centre",
    "playground": lambda t: t.get("leisure") == "playground",
    "restaurant": lambda t: t.get("amenity") == "restaurant",
    "pizza": lambda t: t.get("amenity") == "restaurant",
}
OBJECT_SEARCH = {
    "movie": ["cinema", "movie theater", "multiplex", "mall"],
    "monument": ["monument", "museum", "fort", "palace", "landmark"],
    "movies": ["cinema", "movie theater", "multiplex", "mall"],
    "cinema": ["cinema", "movie theater"],
    "film": ["cinema", "movie theater"],
    "screening": ["cinema"],
    "theatre": ["cinema", "theatre"],
    "theater": ["cinema", "theater"],
    "shopping": ["mall", "shopping centre", "market"],
    "mall": ["mall", "shopping centre"],
    "bazaar": ["market", "bazaar"],
    "market": ["market", "bazaar"],
    "food": ["restaurant", "cafe"],
    "dining": ["restaurant"],
    "dinner": ["restaurant"],
    "lunch": ["restaurant"],
    "brunch": ["cafe", "restaurant"],
    "breakfast": ["cafe", "restaurant"],
    "coffee": ["cafe"],
    "cafe": ["cafe"],
    "reading": ["library"],
    "school": ["school", "college", "university"],
    "college": ["college", "university"],
    "university": ["university"],
    "tuition": ["school", "tuition"],
    "coaching": ["school", "coaching"],
    "hospital": ["hospital"],
    "clinic": ["clinic", "hospital"],
    "temple": ["temple", "place of worship"],
    "mosque": ["mosque"],
    "church": ["church"],
    "gurudwara": ["gurudwara"],
    "atm": ["atm"],
    "bank": ["bank"],
    "petrol": ["petrol", "fuel"],
    "fuel": ["fuel", "petrol"],
    "hotel": ["hotel"],
    "gym": ["gym", "fitness centre"],
    "playground": ["playground"],
    "restaurant": ["restaurant"],
    "pizza": ["restaurant", "pizza"],
}


def _frag_sql(frag: str, radius_m: int, lat: float, lon: float) -> str:
    """Turn a filter (or a '|OR|'-joined union of filters) into Overpass SQL,
    giving every part its own around-clause."""
    return "".join(f"{part}(around:{radius_m},{lat},{lon});"
                   for part in frag.split("|OR|") if part.strip())


def _objective_key(activity: str | None) -> str | None:
    """The objective word if it has its own venue type, else None.

    Handles punctuation and plurals, so "School/ colleges" -> "school".
    """
    if not activity:
        return None
    a = re.sub(r"[^a-z ]", " ", activity.lower())
    words = [w for w in a.split() if w]
    for w in words:
        if w in OBJECT_FILTERS:
            return w
        if w.endswith("s") and w[:-1] in OBJECT_FILTERS:
            return w[:-1]
        if w.endswith("es") and w[:-2] in OBJECT_FILTERS:
            return w[:-2]
    return None


GENERIC_PLACE_WORDS = {"ground", "park", "lake", "garden", "viewpoint", "playground",
                       "pitch", "field", "river", "water", "sports centre", "court",
                       "recreation ground", "play ground", "place of worship",
                       "school", "college", "hospital", "temple", "atm", "bank",
                       "restaurant", "cafe", "market", "mall", "cinema", "hotel"}


def _overpass(query: str, timeout: int = 10):
    """Race the mirrors in parallel and take the first success -- no waiting on
    a slow or dead mirror. (Results are cached for 6 hours, so each place is
    queried at most once.)"""
    body = urllib.parse.urlencode({"data": query}).encode()
    ex = ThreadPoolExecutor(max_workers=len(OVERPASS_MIRRORS))
    futs = [ex.submit(_post_json, url, body, None, timeout) for url in OVERPASS_MIRRORS]
    data, last = None, None
    for f in as_completed(futs):
        try:
            data = f.result()
            break
        except Exception as e:  # noqa: BLE001
            last = e
    ex.shutdown(wait=False)
    if data is None:
        raise last if last else RuntimeError("all Overpass mirrors failed")
    return data


def _viewbox(lat: float, lon: float, radius_m: int) -> str:
    dlat = radius_m / 111_000.0
    dlon = radius_m / (111_000.0 * max(0.2, abs(__import__("math").cos(__import__("math").radians(lat)))))
    return f"{lon - dlon},{lat + dlat},{lon + dlon},{lat - dlat}"


# words that prove a result belongs to a different kind of place
OBJ_MISMATCH = {
    "monument": ["restaurant", "cafe", "burger", "pizza", "dhaba", "bakery", "bar &",
                 "cinema", "mall", "hotel "],
    "cinema": ["restaurant", "cafe", "caf\u00e9", "burger", "pizza", "dhaba", "bakery",
               "kitchen", "bistro", "food", "sweets", "bar &", "pub", "hotel ", "diner",
               "coffee", "juice", "ice cream", "chaat", "thali", "dairy", "barbeque",
               "barbecue", "bbq", "park", "ground", "sagar", "bhavan", "veg "],
    "reading": ["restaurant", "burger", "bar &", "pub", "pizza", "dhaba"],
    "gym": ["restaurant", "cafe", "burger", "pizza", "cinema"],
    "school": ["restaurant", "cafe", "burger", "pizza", "cinema", "bar &"],
    "hospital": ["restaurant", "cafe", "burger", "cinema"],
    "temple": ["restaurant", "cafe", "burger", "cinema", "pizza"],
}


# "movie", "film" and "theatre" are all the cinema category
OBJ_MISMATCH_ALIAS = {"movie": "cinema", "movies": "cinema", "film": "cinema",
                      "screening": "cinema", "theatre": "cinema", "theater": "cinema",
                      "bazaar": "market", "dining": "food", "dinner": "food",
                      "lunch": "food", "brunch": "food", "breakfast": "food",
                      "coffee": "cafe", "pizza": "restaurant"}


def _plausible(name: str, obj: str | None) -> bool:
    """A cinema search must not hand back a food court."""
    if not obj:
        return True
    key = OBJ_MISMATCH_ALIAS.get(obj, obj)
    n = (name or "").lower()
    return not any(bad in n for bad in OBJ_MISMATCH.get(key, []))


def _places_nominatim(lat: float, lon: float, radius_m: int, activity: str, limit: int):
    """Fallback place source when Overpass is unreachable. Tries several terms."""
    from .profiles import resolve_profile_key
    key_act = resolve_profile_key(activity)
    obj = _objective_key(activity)
    terms = (OBJECT_SEARCH.get(obj) if obj
             else ACTIVITY_SEARCH_ALT.get(key_act, [ACTIVITY_SEARCH.get(key_act, "park"), "park"]))
    out: list[dict] = []
    for term in terms:
        params = {"q": term, "format": "json", "limit": limit, "bounded": 1,
                  "viewbox": _viewbox(lat, lon, radius_m)}
        try:
            data = _get_json("https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(params))
        except Exception as e:  # noqa: BLE001
            _LAST_ERROR["places"] = str(e)
            return out
        for h in data:
            parts = [x.strip() for x in h.get("display_name", "Unnamed").split(",")]
            name = parts[0] if parts else "Unnamed"
            if not _plausible(name, obj):
                continue
            # avoid bare generic names like "ground" or "park"
            if len(name) < 8 or name.lower() in GENERIC_PLACE_WORDS:
                name = ", ".join(parts[:2]) if len(parts) > 1 else name
            # Nominatim sometimes shouts ("PLACE OF WORSHIP") -- make it readable
            if name.isupper():
                name = name.title()
            out.append({"name": name,
                        "lat": float(h["lat"]), "lon": float(h["lon"]), "tags": {}})
        if out:
            break
    return out


def places(lat: float, lon: float, radius_m: int, activity: str, limit: int = 12):
    from .profiles import resolve_profile_key
    key_act = resolve_profile_key(activity)
    frag = ACTIVITY_FILTERS.get(key_act, ACTIVITY_FILTERS.get(activity, 'nwr["leisure"="park"]'))
    key = f"poi:{round(lat, 2)}:{round(lon, 2)}:{radius_m}:{activity}"
    if (c := _cached(key, ttl=6 * 3600)) is not None:
        return c
    q = f"[out:json][timeout:6];({_frag_sql(frag, radius_m, lat, lon)});out center tags {limit};"
    out = []
    try:
        data = _overpass(q, timeout=6)
        _LAST_ERROR.pop("places", None)
        for el in data.get("elements", []):
            c_ = el.get("center") or {"lat": el.get("lat"), "lon": el.get("lon")}
            if c_.get("lat") is None:
                continue
            out.append({"name": el.get("tags", {}).get("name", "Unnamed place"),
                        "lat": c_["lat"], "lon": c_["lon"], "tags": el.get("tags", {})})
    except Exception as e:  # noqa: BLE001
        _LAST_ERROR["places"] = f"Overpass unreachable ({e})"
        out = []

    if not out:
        # fall back to Nominatim so the user still gets nearby landmarks
        out = _places_nominatim(lat, lon, radius_m, activity, limit)
        if out:
            _LAST_ERROR["places"] = "used Nominatim fallback (Overpass unavailable)"

    # de-duplicate by name+rounded coords
    seen, uniq = set(), []
    for p in out:
        k = (p["name"], round(p["lat"], 4), round(p["lon"], 4))
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return _store(key, uniq[:limit])


# when a narrow filter comes back empty, widen like a local would
FALLBACK_CHAIN = {
    "cricket": ["field", "picnic"], "football": ["field", "picnic"],
    "tennis": ["field", "picnic"], "golf": ["field", "picnic"],
    "field": ["picnic"], "running": ["picnic"], "cycling": ["picnic"],
    "hiking": ["picnic"], "picnic": ["field"], "photo": ["picnic"],
    "watersport": ["picnic"], "gentle": ["picnic"], "movie": ["mall", "picnic"],
    "shopping": ["market", "picnic"], "food": ["picnic"], "reading": ["picnic"],
}


def places_with_fallback(lat: float, lon: float, radius_m: int, activity: str):
    """Try the exact venue type, then broaden, then widen -- and never give up
    without at least a green space to offer.

    Returns (places, radius_used, filter_used, note).
    """
    from .profiles import resolve_profile_key
    key = resolve_profile_key(activity)
    obj = _objective_key(activity)

    attempts = [(activity, radius_m, None)]
    if obj is None:
        # a family activity can borrow a sibling venue type ...
        for alt in FALLBACK_CHAIN.get(key, ["picnic"]):
            attempts.append((alt, radius_m,
                             f"no {activity} venue was tagged nearby, so I looked for {alt} instead"))
    # ... but a cinema search only ever looks for cinemas, further out
    attempts.append((activity, radius_m * 3,
                     f"nothing was tagged within {radius_m / 1000:.0f} km, so I searched wider"))
    if obj is None:
        attempts.append(("picnic", radius_m * 5,
                         "nothing closer was tagged, so I searched a wider area for parks and gardens"))

    for act, rad, note in attempts:
        try:
            got = places(lat, lon, int(rad), act)
        except Exception:
            got = None
        if got:
            if rad != radius_m and note is None:
                note = f"I widened the search to {int(rad) / 1000:.0f} km to find somewhere"
            return got, int(rad), act, note
    return [], radius_m, None, None


def geospatial_context(lat: float, lon: float, radius_m: int = 3000):
    key = f"gis:{round(lat, 2)}:{round(lon, 2)}:{radius_m}"
    if (c := _cached(key, ttl=6 * 3600)) is not None:
        return c
    q = f"""[out:json][timeout:10];(
      nwr["landuse"="industrial"](around:{radius_m},{lat},{lon});
      nwr["natural"~"water|bay"](around:{radius_m},{lat},{lon});
      nwr["waterway"~"riverbank|river|canal"](around:{radius_m},{lat},{lon});
      nwr["leisure"~"park|garden|common|nature_reserve"](around:{radius_m},{lat},{lon});
      nwr["landuse"~"forest|grass|meadow|recreation_ground|village_green"](around:{radius_m},{lat},{lon});
      nwr["natural"~"wood|scrub"](around:{radius_m},{lat},{lon});
      nwr["power"="line"](around:{radius_m},{lat},{lon});
      way["highway"~"motorway|trunk|primary"](around:{radius_m},{lat},{lon});
    );out center tags 90;"""
    try:
        data = _overpass(q, timeout=7)
    except Exception:
        data = {"elements": []}
    b = {"industrial": [], "water": [], "green": [], "power": [], "roads": []}
    for el in data.get("elements", []):
        t = el.get("tags", {})
        c_ = el.get("center") or {"lat": el.get("lat"), "lon": el.get("lon")}
        item = {"name": t.get("name", "unnamed"), "lat": c_.get("lat"), "lon": c_.get("lon")}
        if t.get("landuse") == "industrial":
            b["industrial"].append(item)
        elif t.get("natural") in ("water", "bay") or t.get("waterway") in ("riverbank", "river", "canal"):
            b["water"].append(item)
        elif (t.get("leisure") in ("park", "garden", "common", "nature_reserve")
              or t.get("landuse") in ("forest", "grass", "meadow", "recreation_ground", "village_green")
              or t.get("natural") in ("wood", "scrub")):
            b["green"].append(item)
        elif t.get("power") == "line":
            b["power"].append(item)
        elif "highway" in t:
            b["roads"].append(item)
    summary = dict(b)
    summary["counts"] = {k: len(v) for k, v in b.items()}
    return _store(key, summary)


# ---------------------------------------------------------------- combined places + context
_ACT_MATCHERS = {
    "running":  lambda t: t.get("leisure") in ("park", "track", "pitch") or
                          any(x in (t.get("sport") or "") for x in ("running", "athletics", "multi")),
    "cycling":  lambda t: t.get("highway") in ("cycleway", "path"),
    "golf":     lambda t: t.get("leisure") == "golf_course",
    "cricket":  lambda t: "cricket" in (t.get("sport") or ""),
    "football": lambda t: any(x in (t.get("sport") or "") for x in ("soccer", "football")),
    "tennis":   lambda t: t.get("sport") == "tennis",
    "hiking":   lambda t: t.get("highway") in ("path", "track") or t.get("leisure") in ("park", "garden"),
    "picnic":   lambda t: t.get("leisure") in ("park", "garden", "common", "picnic_table"),
    "photo":    lambda t: t.get("tourism") == "viewpoint",
    "watersport": lambda t: t.get("natural") == "water" or t.get("leisure") in ("marina", "water_park"),
    "field":    lambda t: t.get("leisure") in ("pitch", "sports_centre", "track") or "sport" in t,
    "event":    lambda t: t.get("leisure") in ("park", "common", "garden"),
    "gentle":   lambda t: t.get("leisure") in ("park", "garden"),
}


def _matches_activity(tags: dict, key_act: str) -> bool:
    fn = _ACT_MATCHERS.get(key_act)
    try:
        return bool(fn(tags)) if fn else tags.get("leisure") in ("park", "garden")
    except Exception:  # noqa: BLE001
        return False


def places_and_context(lat: float, lon: float, radius_m: int, activity: str, limit: int = 12):
    """
    Fetch candidate places AND the geospatial context in a SINGLE Overpass
    round-trip (each statement keeps its own radius). Halves the network time.
    Falls back to a Nominatim place search if Overpass is unreachable.
    """
    from .profiles import resolve_profile_key
    key_act = resolve_profile_key(activity)
    obj = _objective_key(activity)
    # an objective like "movie" searches its own venue type, not a park
    frag = OBJECT_FILTERS.get(obj) if obj else ACTIVITY_FILTERS.get(key_act, 'nwr["leisure"="park"]')
    ctx_r = radius_m + 3000
    q = f"""[out:json][timeout:10];(
      {_frag_sql(frag, radius_m, lat, lon)}
      nwr["landuse"="industrial"](around:{ctx_r},{lat},{lon});
      nwr["natural"~"water|bay"](around:{ctx_r},{lat},{lon});
      nwr["waterway"~"riverbank|river|canal"](around:{ctx_r},{lat},{lon});
      nwr["leisure"~"park|garden|common|nature_reserve"](around:{ctx_r},{lat},{lon});
      nwr["landuse"~"forest|grass|meadow|recreation_ground|village_green"](around:{ctx_r},{lat},{lon});
      nwr["natural"~"wood|scrub"](around:{ctx_r},{lat},{lon});
      nwr["power"="line"](around:{ctx_r},{lat},{lon});
      way["highway"~"motorway|trunk|primary"](around:{ctx_r},{lat},{lon});
    );out center tags 260;"""

    ctx = {"industrial": [], "water": [], "green": [], "power": [], "roads": []}
    places_list: list[dict] = []
    try:
        data = _overpass(q, timeout=6)
        _LAST_ERROR.pop("places", None)
    except Exception as e:  # noqa: BLE001
        _LAST_ERROR["places"] = f"Overpass unreachable ({e})"
        data = None

    if data:
        for el in data.get("elements", []):
            t = el.get("tags", {})
            c_ = el.get("center") or {"lat": el.get("lat"), "lon": el.get("lon")}
            if c_.get("lat") is None:
                continue
            item = {"name": t.get("name", "unnamed"), "lat": c_["lat"], "lon": c_["lon"], "tags": t}
            if t.get("landuse") == "industrial":
                ctx["industrial"].append(item)
            if t.get("natural") in ("water", "bay") or t.get("waterway") in ("riverbank", "river", "canal"):
                ctx["water"].append(item)
            if (t.get("leisure") in ("park", "garden", "common", "nature_reserve")
                    or t.get("landuse") in ("forest", "grass", "meadow", "recreation_ground", "village_green")
                    or t.get("natural") in ("wood", "scrub")):
                ctx["green"].append(item)
            if t.get("power") == "line":
                ctx["power"].append(item)
            if "highway" in t and t.get("highway") in ("motorway", "trunk", "primary"):
                ctx["roads"].append(item)
            # when the objective has its own venue type, ONLY that type counts
            if obj:
                matched = bool(OBJECT_MATCHERS.get(obj, lambda _t: False)(t))
            else:
                matched = _matches_activity(t, key_act)
            if matched:
                places_list.append({"name": t.get("name", "Unnamed place"),
                                    "lat": c_["lat"], "lon": c_["lon"], "tags": t})

    if not places_list:
        places_list = _places_nominatim(lat, lon, radius_m, activity, limit)
        if places_list:
            _LAST_ERROR["places"] = "used Nominatim fallback (Overpass unavailable)"

    # de-duplicate places
    seen, uniq = set(), []
    for pl in places_list:
        k = (pl["name"], round(pl["lat"], 4), round(pl["lon"], 4))
        if k not in seen:
            seen.add(k)
            uniq.append(pl)

    ctx["counts"] = {k: len(v) for k, v in ctx.items()}
    return {"places": uniq[:limit], "context": ctx}


# ---------------------------------------------------------------- routing / terrain
def route_time(olat: float, olon: float, dlat: float, dlon: float, profile: str = "driving"):
    key = f"route:{profile}:{round(olat, 3)},{round(olon, 3)}:{round(dlat, 3)},{round(dlon, 3)}"
    if (c := _cached(key, ttl=3600)) is not None:
        return c
    url = f"https://router.project-osrm.org/route/v1/{profile}/{olon},{olat};{dlon},{dlat}?overview=false"
    try:
        d = _get_json(url, timeout=5)
        r = d["routes"][0]
        return _store(key, {"seconds": r["duration"], "metres": r["distance"]})
    except Exception:
        return _store(key, None)


def route_geometry(olat: float, olon: float, dlat: float, dlon: float,
                   profile: str = "driving"):
    """
    The route as a shape, with the roads it runs along and one alternative.

    OSRM's public server gives us overview=full (the polyline) and steps=true (the
    manoeuvres, each carrying the road name), which is what we need to say *where*
    on the journey the trouble is.
    """
    key = (f"routegeo:{profile}:{round(olat, 3)},{round(olon, 3)}:"
           f"{round(dlat, 3)},{round(dlon, 3)}")
    if (c := _cached(key, ttl=3600)) is not None:
        return c
    # the public OSRM server only serves the car profile; ask for it and scale the
    # duration when the user is walking or cycling
    asked, used = profile, profile
    url = (f"https://router.project-osrm.org/route/v1/{asked}/"
           f"{olon},{olat};{dlon},{dlat}"
           f"?overview=full&geometries=geojson&steps=true&alternatives=true")
    try:
        d = _get_json(url, timeout=8)
        routes = d.get("routes") or []
        if not routes and profile != "driving":
            used = "driving"
            url = (f"https://router.project-osrm.org/route/v1/driving/"
                   f"{olon},{olat};{dlon},{dlat}"
                   f"?overview=full&geometries=geojson&steps=true&alternatives=true")
            d = _get_json(url, timeout=8)
            routes = d.get("routes") or []
        if not routes:
            return _store(key, None)
        # walking ~5 km/h, cycling ~15 km/h against the car's own average
        scale = {"walking": 0.0, "cycling": 0.0}.get(used, 1.0) if used != profile else 1.0
        if used != profile:
            km = (routes[0].get("distance") or 0) / 1000.0
            speed = 5.0 if profile == "walking" else 15.0
            est = (km / speed) * 3600.0 if km else None
        else:
            est = None

        def pack(r):
            coords = [[c[1], c[0]] for c in (r.get("geometry", {}).get("coordinates") or [])]
            steps = []
            for leg in r.get("legs") or []:
                for st in leg.get("steps") or []:
                    name = (st.get("name") or "").strip()
                    loc = (st.get("maneuver") or {}).get("location") or [None, None]
                    steps.append({"name": name, "distance": st.get("distance"),
                                  "duration": st.get("duration"),
                                  "lat": loc[1], "lon": loc[0]})
            return {"seconds": r.get("duration"), "metres": r.get("distance"),
                    "coords": coords, "steps": steps}

        main = pack(routes[0])
        if est:
            main["seconds"] = est
            main["estimated"] = True
        alt = pack(routes[1]) if len(routes) > 1 else None
        if alt and est:
            alt["seconds"] = (alt.get("metres") or 0) / 1000.0 / (
                5.0 if profile == "walking" else 15.0) * 3600.0
            alt["estimated"] = True
        if alt and main.get("metres") and alt.get("metres", 0) > main["metres"] * 1.8:
            alt = None                     # not a real alternative, just a detour
        return _store(key, {"main": main, "alternative": alt})
    except Exception:  # noqa: BLE001
        return _store(key, None)


def elevation(lat: float, lon: float):
    key = f"elev:{round(lat, 4)}:{round(lon, 4)}"
    if (c := _cached(key, ttl=24 * 3600)) is not None:
        return c
    url = f"https://api.opentopodata.org/v1/srtm90m?locations={lat},{lon}"
    try:
        d = _get_json(url)
        return _store(key, d["results"][0]["elevation"])
    except Exception:
        return _store(key, None)


# ---------------------------------------------------------------- optional official
def waqi_aqi(lat: float, lon: float, token: str | None):
    if not token:
        return None
    key = f"waqi:{round(lat, 3)}:{round(lon, 3)}"
    if (c := _cached(key)) is not None:
        return c
    try:
        d = _get_json(f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={token}")
        return _store(key, d.get("data") if d.get("status") == "ok" else None)
    except Exception:
        return None


def imd_nowcast(station_id: str, api_key: str | None, jwt: str | None):
    if not (api_key and jwt):
        return None
    try:
        return _get_json(f"https://api.imd.gov.in/api/v1/nowcast?id={station_id}",
                         headers={"X-API-KEY": api_key, "Authorization": f"Bearer {jwt}"})
    except Exception:
        return None


# ---------------------------------------------------------------- conditions snapshot
def _aqi_band(aqi: float | None) -> tuple[str, str]:
    """Return (label, colour) for a US-AQI value."""
    if aqi is None:
        return "unknown", "#8b949e"
    if aqi <= 50:
        return "Good", "#2ea043"
    if aqi <= 100:
        return "Moderate", "#d29922"
    if aqi <= 150:
        return "Unhealthy (sensitive)", "#db6d28"
    if aqi <= 200:
        return "Unhealthy", "#f85149"
    if aqi <= 300:
        return "Very unhealthy", "#a371f7"
    return "Hazardous", "#8b2f97"


def _now_index(hourly: dict) -> int:
    """Index of the hour closest to now."""
    times = hourly.get("time", [])
    if not times:
        return 0
    from datetime import datetime
    from .tz import IST
    now = datetime.now(IST).strftime("%Y-%m-%dT%H:00")
    try:
        return max(0, times.index(now))
    except ValueError:
        return 0


def conditions(lat: float, lon: float, label: str | None = None) -> dict:
    """A compact current-conditions snapshot for one location."""
    key = f"cond:{round(lat, 3)}:{round(lon, 3)}"
    if (c := _cached(key)) is not None:
        if label and not c.get("label"):
            c["label"] = label
        return c
    wx, aq = weather(lat, lon, days=2), air_quality(lat, lon, days=2)
    snap = {"label": label, "lat": lat, "lon": lon, "available": bool(wx)}
    if wx:
        i = _now_index(wx["hourly"])
        h = wx["hourly"]
        def at(v):
            arr = h.get(v) or []
            return arr[i] if i < len(arr) else None
        snap.update({
            "temperature": at("temperature_2m"),
            "feels_like": at("apparent_temperature"),
            "humidity": at("relative_humidity_2m"),
            "wind": at("wind_speed_10m"),
            "wind_dir": at("wind_direction_10m"),
            "rain_prob": at("precipitation_probability"),
            "uv": at("uv_index"),
            "is_day": at("is_day"),
            "time": h["time"][i] if i < len(h.get("time", [])) else None,
            "sunrise": (wx.get("daily", {}).get("sunrise") or [None])[0],
            "sunset": (wx.get("daily", {}).get("sunset") or [None])[0],
        })
    if aq:
        i = _now_index(aq["hourly"])
        h = aq["hourly"]
        def at(v):
            arr = h.get(v) or []
            return arr[i] if i < len(arr) else None
        aqi = at("us_aqi")
        band, colour = _aqi_band(aqi)
        snap.update({"aqi": aqi, "pm25": at("pm2_5"), "aqi_band": band, "aqi_colour": colour,
                     "pollen": max([x for x in [at("grass_pollen"), at("birch_pollen"), at("alder_pollen")]
                                    if x is not None], default=None)})
    else:
        snap.update({"aqi": None, "aqi_band": "unknown", "aqi_colour": "#8b949e"})
    return _store(key, snap)


def city_conditions() -> list[dict]:
    """Current conditions for the major cities, fetched in parallel."""
    key = "cities"
    if (c := _cached(key, ttl=900)) is not None:
        return c
    results = parallel(lambda c: conditions(c[1], c[2], label=c[0]), MAJOR_CITIES, max_workers=8)
    out = []
    for snap in results:
        if snap:
            snap = dict(snap)
            out.append(snap)
    return _store(key, out)


# ---------------------------------------------------------------- diagnostics
def diagnose(lat: float = 28.6139, lon: float = 77.2090) -> dict:
    """
    Ask every upstream from THIS process and report what each one said.

    Built because a call that works on a laptop can fail on a host for reasons
    the app cannot show you: a blocked egress, an IPv6 route that is not there,
    a rate limit on a shared IP. One request to /diag tells you which source is
    actually broken instead of guessing from a user-facing error.

    The defaults are Delhi.
    """
    import urllib.parse as _up

    wx = ("https://api.open-meteo.com/v1/forecast?" + _up.urlencode(
        {"latitude": lat, "longitude": lon, "hourly": "temperature_2m",
         "forecast_days": 1, "timezone": "Asia/Kolkata"}))
    aq = ("https://air-quality-api.open-meteo.com/v1/air-quality?" + _up.urlencode(
        {"latitude": lat, "longitude": lon, "hourly": "pm2_5", "forecast_days": 1}))
    geo = ("https://nominatim.openstreetmap.org/search?" + _up.urlencode(
        {"q": "Delhi", "format": "json", "limit": 1, "countrycodes": "in"}))
    ov = "https://overpass-api.de/api/status"

    checks = [("weather (Open-Meteo)", wx), ("air quality (Open-Meteo)", aq),
              ("geocoding (Nominatim)", geo), ("places (Overpass)", ov)]
    out: list[dict] = []
    for name, url in checks:
        host = _up.urlsplit(url).netloc
        t0 = time.time()
        row = {"source": name, "host": host}
        try:
            # a short, bounded probe so /diag answers quickly even when a
            # source is black-holing the connection
            data = _get_json(url, timeout=min(HTTP_TIMEOUT, 6))
            row.update(ok=True, ms=int((time.time() - t0) * 1000),
                       detail=f"reached, {len(data) if hasattr(data, '__len__') else '?'} fields")
        except Exception as e:  # noqa: BLE001
            row.update(ok=False, ms=int((time.time() - t0) * 1000),
                       detail=f"{type(e).__name__}: {e}")
        out.append(row)

    return {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "location": {"lat": lat, "lon": lon},
        "transport": {
            "ipv4_forced": _FORCE_IPV4,
            "http_timeout_s": HTTP_TIMEOUT,
            "retries": RETRY_ATTEMPTS,
            "proxy_env_seen": {k: True for k in
                               ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
                               if os.environ.get(k)},
        },
        "upstreams": out,
        "ok_count": sum(1 for r in out if r["ok"]),
        "note": ("Every upstream reached. If the chat still reports missing "
                 "conditions, the problem is not the network."
                 if all(r["ok"] for r in out) else
                 "At least one upstream is unreachable from this host. The "
                 "detail column carries the exact error."),
    }
