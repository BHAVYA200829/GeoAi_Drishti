# Drishti - presentation outline (5 slides)

A tight 5-slide structure. Each slide: a headline, 3-4 bullets, and what to show live.

---

## Slide 1 - Problem
**Headline: "Weather apps tell you the conditions. Drishti tells you where and when to go."**

- Free-form question in, a specific recommendation out: best nearby place **and**
  time, what to carry, what to watch for, when to avoid.
- The answer depends on place *and* hour together - a ground downwind of an
  expressway at 8 AM can be pleasant by evening.
- Personal profiles (asthma, senior, child, pregnancy) tighten the limits.
- Everything must be grounded in live data - no invented weather or places.

*Show:* the ten supported activities and two example questions.

---

## Slide 2 - Approach & architecture
**Headline: "A deterministic core with a small model on top."**

- Pipeline: parse → geocode → fetch weather/AQ/places/GIS → build a place × time
  comfort surface → narrate with provenance.
- The ≤3B SLM does only two jobs: parse the question, and write the advice from
  the evidence it is handed. It never recalls facts.
- Live, free, open sources only: Open-Meteo, OpenStreetMap/Overpass, Nominatim,
  OSRM, OpenTopoData (+ optional WAQI / IMD).
- If the model is unavailable, rule-based fallbacks keep the assistant working.

*Show:* the architecture diagram.

---

## Slide 3 - The USP: Place × Time Comfort Surface
**Headline: "We score every place at every hour, then read the surface."**

- A 0-100 suitability score per (place, hour), from activity comfort curves.
- Read off the best slot, the **golden window**, the avoid window, and
  **flip points**.
- Downwind reasoning: industrial zones and highways from OpenStreetMap compared
  against live wind direction.
- Worth-the-trip scoring: travel time subtracted, so a far venue at rush hour
  loses to a good one nearby.

*Show:* the comfort-surface heat strip in the UI, with the flip visible.

---

## Slide 4 - Worked example
**Headline: "Same day, same conditions - different verdict by hour."**

- Question: *"Is tomorrow good for a run near Koregaon Park?"*
- Riverside Track scores 88 all day. Expressway Ground is cut to **58 in the
  morning** (downwind of NH-48) and recovers to **88 by noon** when the wind shifts.
- Drishti recommends Riverside Track at midday and flags the flip.
- Every number is traceable to a named source.

*Show:* the live `/ask` response and the provenance table.

---

## Slide 5 - Model & limitations (honest)
**Headline: "Small model, grounded system, known limits."**

- Base model: a ≤3B open-weights model (e.g. Qwen2.5-1.5B-Instruct), hosted on
  the Hugging Face Hub with a model card and an open licence. Fine-tuning optional.
- Limitations stated plainly: free-flow travel time approximates rush hour; free
  Overpass/Nominatim are rate-limited and cached; a ≤3B model can mis-parse messy
  input, so rules complete it and missing details trigger a clarifying question;
  unavailable data is reported as a gap, never invented.
- What we would do next: live traffic, finer shade/terrain modelling, a small
  fine-tune on synthetic (question → slots) pairs.

*Show:* the HF model repo and the README's data-source list.

---

## Demo script (2 minutes)

1. Open the UI, click the example *"Where can I go running tomorrow evening near
   Koregaon Park?"* → show the map, the heat strip, and the provenance table.
2. Ask the asthma variant → show the persona cap kicking in (score drops / avoid).
3. Ask a vague question with no location → show the clarifying question instead
   of a guess.
4. Open `/health` and the HF model repo to close on deliverables.
