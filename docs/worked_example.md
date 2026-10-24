# Worked example - the flip that makes the problem interesting

**Question:** *"Is tomorrow good for a run near Koregaon Park?"*

This example shows the headline idea: the right answer depends on the hour and
the place together. Values below are from a deterministic run of the pipeline
with live-shaped inputs.

## What the pipeline did

1. **Parsed** the question → `activity = running`, `location = Koregaon Park`,
   `time = tomorrow`, no persona. Resolved "tomorrow" into the forecast hours.
2. **Geocoded** Koregaon Park (Nominatim) → 18.5366, 73.8933.
3. **Fetched** the hourly forecast and air quality (Open-Meteo) for the centre.
4. **Found candidates** within the radius (OpenStreetMap via Overpass):
   *Riverside Track* and *Expressway Ground*.
5. **Fetched geospatial context** - one industrial zone, three green areas, one
   water body, and the NH-48 highway near the centre.
6. **Fetched travel time** to each place (OSRM).
7. **Built the place × time comfort surface** and analysed it.

## The comfort surface (score per hour, 0-100)

The wind blows **from the north until noon**, then **shifts to the south**.

| Place | 06:00 | 08:00 | 10:00 | 12:00 | 14:00 | 16:00 | 18:00 | 20:00 |
|---|---|---|---|---|---|---|---|---|
| Riverside Track | 88 | 88 | 88 | 88 | 88 | 88 | 88 | 88 |
| Expressway Ground | **58** | **58** | **58** | 88 | 88 | 88 | 88 | 88 |

*Expressway Ground* sits just south of the NH-48 highway. In the morning the wind
blows from the north, so the ground is **downwind of the highway** and its score
is cut to 58. At noon the wind shifts to the south, the ground is no longer
downwind, and its score jumps to 88.

## What Drishti returned

- **Verdict:** GO
- **Best:** Riverside Track, 12:00 PM, score 88/100 (~10 min away)
- **Golden window:** 12:00 PM → 8:00 PM at Riverside Track
- **Flip:** *"Expressway Ground is poor early but improves around 12:00
  (downwind of NH-48 while the wind blows from there)."*

### Advice

> Best bet for running: **Riverside Track** at Fri 23 Oct, 12:00 PM
> (suitability 88.0/100). That is about 10 min away.
> The good window runs 12:00 PM to 8:00 PM at Riverside Track.
> Live conditions (Open-Meteo): 27°C, 5% chance of rain, wind 12 km/h, AQI 70.
> Carry: hydration and a charged phone.
> Watch for: Riverside Track improves around 12:00 PM (downwind of NH-48 while
> the wind blows from there).

## Why this is the whole point

A plain weather app would say "27°C, light wind, AQI 70 - good for a run" for the
entire day, for every place. Drishti says something the app cannot: *this
particular ground is worse in the morning because of where it sits relative to
the highway and the wind, and it becomes just as good by noon.* That is the
place-and-time reasoning the challenge asks for.

## Provenance

Every claim above carries its source and fetch time: temperature and AQI from
Open-Meteo, the place list and the NH-48 feature from OpenStreetMap via Overpass,
the coordinates from Nominatim, and the travel time from OSRM.
