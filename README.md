# Drishti - where and when to head outdoors

Drishti is a grounded AI assistant for outdoor activities. Give it a free-form
question like *"Where can I go running tomorrow evening near Koregaon Park?"* and
it returns the best nearby place **and time**, what to carry, what to watch for,
and when to avoid going - with every claim traced to the live data it used.

Built for the **GeoAI Hackathon @ CSRE, IIT Bombay**.

## The idea in one line

Most weather apps tell you *what the conditions will be*. Drishti tells you
*which nearby place, at what time, is best for you* - because the right answer
depends on the hour and the place together. A ground lying downwind of an
expressway at 8 AM can be pleasant by evening once the wind shifts.

## Headline USP - the Place × Time Comfort Surface

Instead of checking one place at one time, Drishti builds an explicit surface:
for every candidate place and every upcoming hour it computes a 0-100
suitability score, then reads that surface to produce:

- the **best (place, hour)**,
- the **golden window** (contiguous good hours),
- the **avoid window**,
- **flip points** - hours where the verdict changes, e.g. a place that is poor
  early because it is downwind of a road but improves once the wind turns.

Supporting USPs:

- **Downwind reasoning** - industrial zones and major roads from OpenStreetMap
  are compared against live wind direction; a place upwind of a pollution source
  is penalised.
- **Worth-the-trip scoring** - travel time (OSRM) is subtracted from the
  suitability score, so a venue 40 minutes away at rush hour is not recommended
  over a good one nearby.
- **Persona hard overrides** - asthma, senior, child and pregnancy profiles cap
  or reject slots (AQI, pollen, heat index, cold, slope). These are constraints,
  not soft advice.
- **Conversational** - Drishti chats. If a question is missing a detail it asks
  back in the same conversation (activity first, then place), remembers the
  partial question and your own words, and completes the answer once you reply.
  Pass `session_id` to continue a conversation. A greeting or an unrelated
  message never reuses a previous location or task - it starts clean, and the
  UI has a **New chat** button to reset the conversation at any time, and the
  resolved activity is remembered across turns so a later place reply completes
  the original task.
- **Directions, not just a verdict** - every recommendation comes with a keyless
  Google Maps place link, a Google Maps directions link (travel mode chosen per
  activity) and an OpenStreetMap link, plus distance and travel time.
- **Provenance ledger** - every factual claim carries its source and timestamp.
- **Deterministic core** - the small model never computes weather or picks
  places; it only parses the question and narrates the result. If the model is
  unavailable, rule-based fallbacks keep the assistant fully working.

## Architecture

```
question
  -> query_parser   (SLM slot-filling + deterministic rules)   NLU
  -> data_fetcher   (Nominatim, Open-Meteo, Overpass, OSRM)    live data
  -> scorer         (place x time surface, persona, downwind)  reasoning
  -> narration      (SLM or deterministic advice + provenance)  NLG
  -> FastAPI /ask
```

The SLM is a small (<= 3B) open-weights model used for exactly two jobs: turning
a messy question into structured slots, and writing the final advice from the
evidence it is handed. It is never asked to recall facts.

## Files

```
drishti/
├── app/
│   ├── main.py           FastAPI app  (/, /ask, /health, /sources, /cities, /conditions, /static)
│   ├── i18n.py           UI + advice strings (en, hi, mr, ta, bn)
│   ├── session.py        multi-turn conversation memory
│   ├── maps.py           Google Maps links + how-to-reach guidance
│   ├── config.py         env-driven settings
│   ├── schemas.py        request/response models
│   ├── data_fetcher.py   all live open-data sources
│   ├── query_parser.py   NLU (SLM + rules)
│   ├── timeparse.py      time phrases -> forecast hours (deterministic)
│   ├── profiles.py       activity comfort curves + persona rules
│   ├── scorer.py         the comfort surface engine
│   ├── reasoning.py      orchestrator
│   ├── narration.py      advice + provenance
│   ├── slm.py            the <=3B model (optional)
│   └── static/
│       ├── index.html    entry page (the uploaded lamp/sign-in template)
│       ├── app.html      chat (sidebar, voice, themes, sequential answers)
│       ├── logo.png      the Drishti mark (transparent)
│       ├── logo-192.png
│       └── favicon.png
├── tests/test_pipeline.py
├── scripts/push_to_hub.py   push the model/adapter to the HF Hub
├── docs/
│   ├── worked_example.md    a full worked example (the flip)
│   └── presentation_outline.md
├── requirements.txt
├── .env.example
├── Dockerfile
├── render.yaml              one-click Render deploy
├── Procfile
├── DEPLOY.md                deployment guide (Render / Railway / HF / Docker)
├── run.sh
└── MODEL_CARD.md
```

## Any outdoor activity works

The ten listed activities are the minimum; Drishti is not limited to them.
Any activity you name is resolved to a sensible profile family and handled:

- Named activities (running, cycling, golf, cricket, football, tennis, hiking,
  picnic, event, photo) use their own tuned comfort curves.
- Additional activities - **boating, kayaking, swimming, surfing, fishing,
  sailing, rafting, rowing, archery, skating, horse riding, frisbee, badminton,
  volleyball, basketball, hockey, kabaddi, yoga, meditation, birdwatching,
  gardening, camping, trekking** and more - map to one of three extra families
  (water-based, field/court, gentle outdoor) with appropriate sensitivities.
- Anything else still works. A broad objective lexicon (movie, cinema, shopping,
  stargazing, kite flying, painting, dining, paragliding …) plus an "‑ing"
  heuristic means an unrecognised objective is still treated as a task and gets a
  sensible profile and place search - it is never ignored or looped back.
- Some objectives get their own kind of venue rather than a park: **movie / cinema
  -> cinemas**, **shopping / mall -> malls**, **market -> marketplaces**,
  **food / dining / coffee -> restaurants and cafes**, **reading -> libraries**.
- When you send a short standalone message, Drishti tells an **objective** apart
  from a **place name**: objective words are recognised from the lexicon, and
  anything else is geocoded - if it resolves to a place, it is treated as a
  location; otherwise it is treated as an objective.

The assistant keeps *your* word for the activity in the reply (e.g. "Best bet
for boating: …"), not its internal family name.

## Weather/AQI questions vs activity questions

Drishti answers a **weather or air-quality question with just the conditions**,
and only offers activity suggestions when you actually ask to do something.

- "What is the weather in Delhi?", "AQI in Mumbai right now", or simply
  "Bengaluru" -> current conditions for that place (temperature, feels-like,
  humidity, wind, rain chance, AQI, sunrise/sunset) and nothing else.
- "Where can I go running near Bandra?", "I want to do something outdoors near
  Bandra", "good place for golf on Saturday" -> an activity recommendation.
- Place categories work directly too: **school / college / university, hospital,
  temple, atm, bank, petrol, hotel, gym, playground, restaurant, library, mall,
  cinema, market** - so "School/ colleges" or "Hospitals in Pune" returns those
  places, not a park.

## Location handling (important)

Drishti answers for **any place in India** you name - a city, a neighbourhood,
a landmark or a district. Two rules govern location:

1. **A named place always wins.** If your message contains a place, Drishti
   resolves that place. Coordinates from your device are ignored for that query.
   So "good hiking spots near Manali" is answered for Manali even if you are in Pune.
2. **Your own location is opt-in, never sticky.** Coordinates are used only when
   you explicitly ask (the "📍 Use my location" control, or the "Use my current
   location" quick reply), and you can turn it off at any time. It is never
   attached to later questions about other places.

Drishti also answers plain weather/air-quality questions for any place - e.g.
*"What is the AQI in Delhi right now?"* - without needing an activity.

## Submitting (the five deliverables)

| # | Deliverable | Where it is |
|---|---|---|
| 1 | **Live API endpoint** | **<PASTE YOUR DEPLOYED URL HERE>** - e.g. `https://drishti.onrender.com/ask` |
| 2 | **Model on Hugging Face** | **<PASTE YOUR HF REPO URL HERE>** - e.g. `https://huggingface.co/<user>/drishti-slm` |
| 3 | **Code repository** | **<PASTE YOUR GITHUB URL HERE>** |
| 4 | **This README** | how to run, the base model, the data sources, and the endpoint above |
| 5 | **Presentation (4-6 slides)** | `docs/presentation_outline.md`, deck in `docs/` |

### Deploy it (do this first - the endpoint must stay up through judging)

```bash
# any host that runs Python: Render, Railway, Fly, a VPS
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Set `DRISHTI_ENABLE_SLM=1` and `DRISHTI_MODEL_ID=<your-hf-user>/drishti-slm` in the
host's environment to serve the fine-tuned model. `Procfile`, `Dockerfile` and
`render.yaml` are already in the repo.

Then check `GET /health` - it lists the routes the process actually loaded, so you
can see at a glance that the deploy is running the current code.

### Publish the model

```bash
huggingface-cli login
python training/generate_dataset.py --out data --n 1500
python training/train_lora.py --data data/all.jsonl --out out/drishti-lora
python scripts/push_to_hub.py --repo-id <your-hf-user>/drishti-slm --adapter out/drishti-lora
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | the web UI (landing dashboard + assistant) |
| GET | `/health` | liveness for the judging window |
| GET | `/sources` | the list of data sources used |
| GET | `/activities` | the pickable activities (tasks) shown in the UI |
| GET | `/cities` | current weather + AQI for major Indian cities (dashboard) |
| GET | `/conditions?lat=&lon=&label=` | current conditions for any location |
| POST | `/ask` | the assistant: `{"question": "...", "lat": ..., "lon": ..., "session_id": ...}` |

## The web app

- **`/` - the entry screen.** Just the uploaded logo, the Drishti mascot doing a
  **namaste gesture** (folded hands and a gentle bow) and the word **"Namaste,"**
  - then a "Continue to Drishti" button. No account needed.
- **`/signin` - the lamp template**, kept exactly as uploaded (dark room, lamp,
  pull-cord, sign-in / create-account card, and the "Continue without signing in"
  button). The welcome screen links to it.
- **`/app` - the chat.** Light and dark themes, a left sidebar with chat history
  and favourite places, voice in/out, and the answer format below.

### Understanding the sentence

Drishti reads the whole sentence, not a single keyword, and it understands
**Hinglish and Marathi** as well as English:

| You say | Understood as |
|---|---|
| "I feel like going for a jog this evening near Bandra" | running · Bandra · this evening |
| "mujhe kal subah cricket khelna hai powai me" | cricket · powai · tomorrow morning |
| "kya aaj bahar jana chahiye thane me" | a go / no-go question for thane |
| "aaj shaam ko ghumne kaha ja sakta hu" | a walk · today evening |
| "bandra ke paas jogging" | running · bandra |
| "navi mumbai me running ke liye jagah" | running · navi mumbai |
| "shall we do a barbeque this saturday" | a picnic · saturday |
| "suggest a cafe for a quiet read" | reading |
| "mausam kaisa hai delhi me" | the weather in Delhi |
| "which park is good near me" | *your* location, not a place called "me" |

It handles synonyms (jog→running, bike→cycling, trek→hiking, swim→watersport,
match→a sports venue, barbeque→picnic, read→reading, **movie / film / cinema /
multiplex / picture → the cinema**), Hinglish time words
(kal, parso, aaj, abhi, subah, shaam, dopahar, raat), Hinglish place markers
("powai **me**", "bandra **ke paas**"), and clock times ("at 6").

The reply is written the way a friend would say it - *"Conditions look pleasant
- about 28°C, no rain expected."* - never as a raw dump of numbers.


Drishti reads the whole sentence, not a single keyword:

- *"Should I go for the college today, what's the weather in Thane, use my location"*
  is understood as a **go / no-go question for Thane** - not as a request for a
  "college" activity.
- **Go / no-go** answers are friendly: *"Yes - it's a good time to head out -
  Thane. 32°C, feels like 35°C … Because: quite hot."*
- **Weather / AQI questions** answer with conditions only. Activity questions get
  a recommendation. **Place categories** (school, college, hospital, temple, atm,
  gym, cinema, mall …) return those places.
- The **keywords** Drishti picked out (activity, place, time, persona, radius,
  weather, question) are shown as chips under the answer, and the answer is built
  from them - including **"use my location"**, which is now acted on: if the fix
  is available it is used, and if it is not, Drishti says so and offers the
  location button instead of silently asking for a place.

### Several places, not just one

A single answer gives you a real choice:

1. A plain sentence - **"Go to Rajiv Gandhi Park - around 07:00."**
2. **The top pick**, as a full brief: the card (place, time, score, directions)
   followed by numbered steps - **Why**, **How to reach**, **What to carry**,
   **What to watch for**
3. **Also worth a look** - the next **two** places, each with the same full brief
4. **Best hours** for the top pick - the comfortable stretches in words
   (*07:00-11:00*), not a colour grid
5. **When to avoid** and a **timing note** if the verdict changes during the day
6. **More locations to explore** - every other nearby place as a one-line row
   (name · time · score). **Tap any row and its brief opens right there** - no
   extra request, it is already in the answer.
7. **More places** - widens the radius and finds new ones further out
8. **Sources** - every claim and where it came from

### Features

- **Theme** - light / dark toggle in the top bar.
- **Language** - English, हिन्दी, मराठी, தமிழ், বাংলা (top bar). **The whole
  interface and the whole reply are in the chosen language** - the narration is
  rewritten through a phrase table (`app/l10n.py`) plus numeric patterns, so
  sentences, headings, buttons, keywords and the answer itself all change. Only
  proper nouns and numbers stay as they are.
- **Voice** - replies are spoken in the chosen language, with a **female voice**:
  the engine picks a voice matching the language and prefers known female names
  (Swara, Heera, Kalpana, Lekha, Zira, Samantha and so on), with the pitch lifted
  slightly as a fallback.
- **Interrupt** - while Drishti is answering the Send button becomes **Stop** and
  cancels the request.
- **Location is opt-in** - a named place always wins; your own location is used
  only when you turn it on, and you can turn it off at any time.
- **Favourite places** - the star on any answer saves the place to the sidebar.
- **Voice** - speak your question (mic), and toggle spoken replies.
- **Chat history** - every conversation is a thread in the sidebar; signing in
  saves them (and your favourites) to this device.
- **Listen / Share / Helpful / Not quite** on every answer.

### Look

The **Indian floral** theme - a warm white ground with a subtle paisley/lotus
motif, forest green and antique gold, and a neutral slate gradient used for the
brand, the primary buttons and the hero text. Space Grotesk for headings,
Instrument Serif italics for accent words, Inter for body text, Lucide icons
throughout. A matching low-contrast dark mode (deep forest green) is one click
away in the top bar.

### Where the namaste shows

The **namaste greeting lives only on the entry screen** (`/`) - the logo, the
mascot doing the folded-hands namaste, and the word "Namaste,". It appears when
someone first opens the site or refreshes it. The chat page (`/app`) carries no
mascot, so the gesture never repeats mid-conversation.

### Speed

A GPS fix used to be the slow path. What changed:

- **Reverse geocoding no longer blocks the answer.** It only supplies a display
  name, so it now runs *in parallel* with the weather, air-quality and places
  calls, with a **4-second timeout** and a 24-hour cache. The label falls back to
  "your location" if it does not arrive in time.
- **Overpass is capped at 7 seconds** (query and client), so a slow mirror cannot
  stall a reply.
- **The global HTTP timeout dropped from 25s to 12s.**
- **The browser asks for a coarse, cached fix** (`enableHighAccuracy: false`,
  `maximumAge: 5 min`, `timeout: 7s`) instead of waiting for a high-accuracy one.
- **Choosing "Use my current location" sends the question immediately** once the
  fix lands - no second tap needed. (It used to do nothing at all: the send
  helper bailed out on an empty text box.)
- **The cache is warmed the moment a fix arrives.** The page posts the coordinate
  to `POST /warm`, which fetches weather and air quality into the cache, so the
  question that follows answers in a fraction of the time.
- **Travel-time routing** has a 5-second timeout and a 4-way fan-out.
- Everything is cached (weather 30 min, places/context 6 h, geocode 30 min), so a
  repeat question is near-instant.

Measured in the sandbox: a fresh GPS + activity query 2.9 s, the same query cached
0.15 s, a go/no-go question 0.01 s.


