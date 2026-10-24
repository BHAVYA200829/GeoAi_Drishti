---
library_name: peft
base_model: Qwen/Qwen2.5-1.5B-Instruct
license: apache-2.0
language:
  - en
tags:
  - geospatial
  - weather
  - tool-use
  - lora
  - qlora
  - india
datasets:
  - drishti-synthetic-outdoor-advisories
pipeline_tag: text-generation
---

# Drishti SLM

A small open model, fine-tuned to **reason over fetched geospatial and weather
data** for an outdoor-activity advisor in India. Built for the
**GeoAI Hackathon @ CSRE, IIT Bombay**.

The model never recalls weather or geography. It is handed a set of places, hours
and the real numbers for each, and it chooses between them.

## What it does

Three jobs, one adapter:

| job | input | output |
|---|---|---|
| **parse** | `"I have mild asthma, where's good for a morning walk near Powai in the next two days?"` | strict JSON slots: activity, place, radius, time, persona |
| **decide** | a `FACTS` block: the scored (place, hour) surface with real weather, AQI, travel time and geospatial flags | which place and hour, why, why the runner-up lost, what to carry, what to watch for |
| **narrate** | the chosen recommendation plus its evidence | a short, grounded answer that names its sources |

The **decide** job is the reason this model exists. A deterministic engine does the
fetching and the scoring; the model makes the judgement call that a scorer cannot:
that a shaded place beside water is worth more on a hot afternoon, that a ground
downwind of an expressway is worth less at 8 AM than at 6 PM, that a venue shut at
the suggested hour is not a choice at all, and that a venue forty minutes away only
pays off if the conditions really are better.

## Grounding, by construction

Every decision is validated before it reaches the user (`app/planner.py`):

- a `best_place` that is not in the supplied `FACTS` is **rejected**;
- a `best_hour` that the place does not have falls back to that place's best hour;
- any sentence quoting a number the facts do not contain is **dropped**;
- `carry` and `watch` items outside a fixed vocabulary are discarded;
- if the decision does not survive, the engine's own pick stands.

The model can therefore only ever choose between real, already-fetched options. It
cannot introduce a place, an hour or a number of its own.

## Base model

`Qwen/Qwen2.5-1.5B-Instruct` - 1.54B parameters, Apache-2.0, open weights.
Other options, all under the 3B limit: `Qwen/Qwen2.5-0.5B-Instruct`,
`microsoft/Phi-3-mini-4k-instruct`, `meta-llama/Llama-3.2-1B-Instruct`.
Swap with `--base` in `training/train_lora.py`.

## Training

QLoRA (4-bit NF4), LoRA rank 16 / alpha 32 on all attention and MLP projections.
Two epochs, cosine schedule, lr 2e-4, effective batch 16, max length 1536.

```bash
pip install -r training/requirements-train.txt
python training/generate_dataset.py --out data --n 1500
python training/train_lora.py --data data/all.jsonl --out out/drishti-lora
```

Data is **synthetic**: ~1500 chat-format examples covering the three jobs, drawn
from realistic Indian weather, AQI, travel-time and geospatial ranges. It teaches
the *shape* of the task - choose only from what you were given, quote no number you
were not given, keep it short - not the climate of any city. Regenerate with a
different `--seed` for a fresh split.

## Using it

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(base)
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(base), "<this-repo>")

msgs = [{"role": "system", "content": DECIDE_SYSTEM},
        {"role": "user", "content": "FACTS: " + facts_json}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
print(tok.decode(model.generate(ids, max_new_tokens=320)[0][ids.shape[-1]:],
                 skip_special_tokens=True))
```

In the assistant itself, set `DRISHTI_ENABLE_SLM=1` and point
`DRISHTI_MODEL_ID` at this repo (or at a local path). A quantised GGUF file can be
used instead with `DRISHTI_SLM_BACKEND=llamacpp` and `DRISHTI_GGUF_PATH`, which is
3-4x faster on CPU.

## Limitations

- **Synthetic training data.** The model has not seen real advisories; it has seen
  the structure of them. Real behaviour depends on the live data the engine feeds
  it.
- **It decides, it does not compute.** Scores, weather values, travel times and
  distances all come from the engine. The model's contribution is the choice and
  the framing.
- **English.** Prompts are English. The assistant's own reply localisation
  (English, Hindi, Marathi, Tamil, Bengali) is a separate phrase-table layer, not
  the model.
- **Small.** At 1.5B it can be inconsistent on unusual phrasings; the deterministic
  fallback exists precisely for that.
- **Not a safety system.** Health advice is bounded by fixed persona rules in the
  engine, not by the model.

## Licence

Apache-2.0, matching the base model. Built with open data:
Open-Meteo, OpenStreetMap via Overpass, Nominatim, OSRM.
Map data © OpenStreetMap contributors.
