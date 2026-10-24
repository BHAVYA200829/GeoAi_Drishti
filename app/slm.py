"""
Drishti SLM layer.

The <=3B model does exactly two things and nothing else:
  1. parse_slots(question)      -> JSON slots (NLU)
  2. polish_advice(...)         -> natural-language narration (NLG)

It never recalls or invents weather, places or times. If the model is not
available the assistant falls back to deterministic parsing/narration.

Two backends are supported, chosen with DRISHTI_SLM_BACKEND:
  * "transformers" (default) -- Hugging Face weights, works on CPU or GPU.
  * "llamacpp"               -- a quantised GGUF file, 3-4x faster on CPU.
Set DRISHTI_GGUF_PATH to the .gguf file when using the llamacpp backend.
"""
from __future__ import annotations

import json
import os

from .config import MODEL_ID, SLM_MAX_NEW_TOKENS

_pipe = None
_llm = None

SLOT_SYSTEM = """You are a query parser for an outdoor-activity advisor in India.
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

SLOT_FEWSHOT = """Q: "Where can I go running tomorrow evening near Baner?"
A: {"activity":"running","location_text":"Baner","radius_km":null,"time_expression":"tomorrow evening","persona":{"age":null,"condition":null},"needs_clarification":false,"clarification_question":null}
Q: "I have mild asthma, where's good for a morning walk near Salt Lake in the next two days?"
A: {"activity":"hiking","location_text":"Salt Lake","radius_km":null,"time_expression":"next two days morning","persona":{"age":null,"condition":"asthma"},"needs_clarification":false,"clarification_question":null}"""

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
 "carry": [up to 3 short items from: water, a hat, sun cream, sunglasses, an umbrella, a rain jacket, a mask, a warm layer, a jacket, insect repellent, a torch, sturdy shoes, a charged phone, snacks],
 "watch": [up to 3 from: heat, cold, rain, air quality, wind, uv, pollen, darkness, crowds, closed, the trip, uneven ground, insects],
 "confidence": "high" | "medium" | "low"}

Weigh the person, the task and the hour: a shaded place beside water wins on a hot
afternoon, a place downwind of a road loses, a place that is shut at that hour is
not a choice, and a long trip only pays off if the conditions really are better.
Do not quote any number in "why" or "why_not_runner_up". Output JSON only."""

DECIDE_FEWSHOT = """FACTS: {"task":"running","person":"none","when":"tomorrow morning","places":[{"name":"Kamala Nehru Park","best_hour":"07:00","best_score":91.2,"travel_min":6,"shaded_by":"a garden","near_water":null,"hours":[{"hour":"07:00","score":91.2,"temp_c":26,"feels_like_c":27,"rain_pct":0,"aqi":72,"wind_kmh":5,"uv":2,"travel_min":6,"flags":["conditions are comfortable for this activity"]}]},{"name":"Bare Maidan","best_hour":"07:00","best_score":78.4,"travel_min":14,"shaded_by":null,"near_water":null,"hours":[{"hour":"07:00","score":78.4,"temp_c":27,"feels_like_c":28,"rain_pct":0,"aqi":118,"wind_kmh":9,"uv":2,"travel_min":14,"flags":["downwind of a road while the wind blows from there"]}]}]}
A: {"best_place":"Kamala Nehru Park","best_hour":"07:00","why":"It is the cooler, cleaner option at that hour and it is close by.","runner_up":"Bare Maidan","why_not_runner_up":"It sits downwind of a road and the air there is poorer.","carry":["water"],"watch":["air quality"],"confidence":"high"}"""

NARRATE_SYSTEM = """You are Drishti, an outdoor-activity advisor in India.
Write a short, warm recommendation using ONLY the facts in EVIDENCE.
Never invent a weather value, place or time that is not in EVIDENCE.
Name the data source for factual claims. If a value is missing, say it is unavailable.
Cover: best place and time, what to carry, what to watch for, and when to avoid going.
Keep it under 160 words."""


def _backend() -> str:
    return os.environ.get("DRISHTI_SLM_BACKEND", "transformers").strip().lower()


def _load_transformers():
    global _pipe
    if _pipe is not None:
        return _pipe
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map="cpu"
    )
    _pipe = pipeline("text-generation", model=model, tokenizer=tok)
    return _pipe


def _load_llamacpp():
    global _llm
    if _llm is not None:
        return _llm
    from llama_cpp import Llama  # pip install llama-cpp-python
    path = os.environ.get("DRISHTI_GGUF_PATH")
    if not path or not os.path.exists(path):
        raise RuntimeError("Set DRISHTI_GGUF_PATH to a .gguf file for the llamacpp backend")
    _llm = Llama(
        model_path=path,
        n_ctx=int(os.environ.get("DRISHTI_N_CTX", "4096")),
        n_threads=int(os.environ.get("DRISHTI_N_THREADS", str(os.cpu_count() or 4))),
        verbose=False,
    )
    return _llm


def _chat(system: str, user: str, max_new_tokens: int = SLM_MAX_NEW_TOKENS,
          temperature: float = 0.0) -> str:
    if _backend() == "llamacpp":
        llm = _load_llamacpp()
        out = llm.create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return out["choices"][0]["message"]["content"].strip()

    pipe = _load_transformers()
    out = pipe(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        return_full_text=False,
    )
    return out[0]["generated_text"].strip()


def _extract_json(text: str) -> dict | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def parse_slots(question: str, retries: int = 2) -> dict | None:
    user = f"{SLOT_FEWSHOT}\nQ: {json.dumps(question)}\nA:"
    for _ in range(retries + 1):
        slots = _extract_json(_chat(SLOT_SYSTEM, user, max_new_tokens=256))
        if slots is not None:
            return slots
    return None


def polish_advice(recommendation: dict, evidence: dict) -> str | None:
    user = ("RECOMMENDATION:\n" + json.dumps(recommendation, indent=2) +
            "\n\nEVIDENCE (cite these sources):\n" + json.dumps(evidence, indent=2))
    try:
        return _chat(NARRATE_SYSTEM, user, max_new_tokens=400, temperature=0.3)
    except Exception:
        return None


def decide(facts: dict, question: str = "", retries: int = 1) -> dict | None:
    """
    Ask the model to choose between the scored candidates.

    The model only ever sees places and hours the engine already fetched, and its
    answer is validated against them afterwards (see app/planner.py), so it cannot
    introduce a place or a number of its own.
    """
    user = (DECIDE_FEWSHOT + "\nFACTS: " + json.dumps(facts, ensure_ascii=False) +
            ("\nQUESTION: " + question if question else "") + "\nA:")
    for _ in range(retries + 1):
        raw = _chat(DECIDE_SYSTEM, user, max_new_tokens=320)
        got = _extract_json(raw)
        if isinstance(got, dict) and got.get("best_place"):
            return got
    return None


def model_name() -> str:
    if _backend() == "llamacpp":
        return f"{MODEL_ID} (GGUF)"
    return MODEL_ID
