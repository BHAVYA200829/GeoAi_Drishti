"""
Map links and "how to reach" guidance.

Uses the keyless Google Maps URL scheme, which opens the Maps app on a phone
and Maps in a browser on desktop:
    search   https://www.google.com/maps/search/?api=1&query=<lat>,<lon>
    dirs     https://www.google.com/maps/dir/?api=1&origin=..&destination=..&travelmode=..

Also offers an OpenStreetMap link, since the map data comes from OSM.
"""
from __future__ import annotations

import urllib.parse

# activity -> how the user most likely gets there
TRAVEL_MODE = {
    "cycling": "bicycling",
    "hiking": "walking",
    "running": "walking",
}
MODE_LABEL = {"driving": "driving", "walking": "walking", "bicycling": "cycling", "transit": "transit"}


def place_url(lat: float, lon: float, name: str | None = None) -> str:
    q = f"{lat},{lon}"
    if name:
        q = f"{name} {lat},{lon}"
    return "https://www.google.com/maps/search/?" + urllib.parse.urlencode({"api": "1", "query": q})


def directions_url(olat: float, olon: float, dlat: float, dlon: float, mode: str = "driving") -> str:
    params = {"api": "1", "origin": f"{olat},{olon}", "destination": f"{dlat},{dlon}",
              "travelmode": mode}
    return "https://www.google.com/maps/dir/?" + urllib.parse.urlencode(params)


def osm_url(lat: float, lon: float) -> str:
    return f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=16/{lat}/{lon}"


def travel_mode_for(activity: str | None) -> str:
    return TRAVEL_MODE.get(activity or "", "driving")


def how_to_reach(activity: str | None, metres: float | None, minutes: float | None,
                 olat: float, olon: float, dlat: float, dlon: float) -> dict:
    """Everything the UI needs to describe the trip and link out to Maps."""
    mode = travel_mode_for(activity)
    dist = f"{metres/1000:.1f} km" if metres else None
    dur = f"{minutes:.0f} min" if minutes is not None else None
    parts = []
    if dist and dur:
        parts.append(f"about {dist} away, roughly {dur} by {MODE_LABEL[mode]}")
    elif dist:
        parts.append(f"about {dist} away")
    text = ("Getting there: " + parts[0] + ".") if parts else "Open the map for directions."
    return {
        "travel_mode": mode,
        "travel_mode_label": MODE_LABEL[mode],
        "distance": dist,
        "duration": dur,
        "text": text,
        "google_maps_place": place_url(dlat, dlon),
        "google_maps_directions": directions_url(olat, olon, dlat, dlon, mode),
        "openstreetmap": osm_url(dlat, dlon),
    }

