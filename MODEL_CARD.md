# Model Card - Drishti (SLM component)

## Model details
- **Base model:** Qwen/Qwen2.5-1.5B-Instruct  *(replace with the exact model used)*
- **Parameters:** 1.5B (must be <= 3B)
- **Fine-tuning:** none / LoRA adapter  *(delete whichever does not apply)*
- **Adapter repo (if fine-tuned):** link to the LoRA adapter here
- **Licence:** Apache-2.0  *(match the base model licence)*
- **Developed for:** GeoAI Hackathon @ CSRE, IIT Bombay

> Note: Phi-3-mini is 3.8B and exceeds the <=3B rule, so it is not used.

## What this model does
Drishti's language component performs exactly two jobs:
1. **Query parsing** - free-form question -> structured slots
   (activity, location, radius, time, persona).
2. **Narration** - a computed recommendation plus fetched evidence -> short,
   grounded advice.

It is **never** used to recall or predict weather, geography, or places. All
factual content is fetched live and supplied as evidence. If the model is
unavailable, deterministic rule-based fallbacks take over, so the assistant
still works.

## Intended use
- Input: a user question (parsing), or a recommendation + evidence object (narration).
- Output: JSON slots, or a short piece of grounded advice.
- Language: English (India context).

## Out of scope
- No weather or geographic recall.
- Not for medical, safety-critical, or emergency decisions.
- Not for non-outdoor-activity tasks.

## Training data (if fine-tuned)
Describe the synthetic `(question -> slots)` and `(evidence -> advice)` pairs
used, their size, and how they were generated.

## How to load
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("<your-repo-id>")
model = AutoModelForCausalLM.from_pretrained("<your-repo-id>")
```

## Limitations
- A <= 3B model can make slot-extraction mistakes on messy input; deterministic
  rules complete and correct its output, and missing details trigger a
  clarifying question rather than a guess.
- Narration quality depends on the evidence supplied; missing values are
  reported as unavailable rather than invented.

## The model's role

The small model is Drishti's **decision layer**. The deterministic engine fetches
the data and scores every (place, hour) pair; the model is handed that surface as
`FACTS` and decides which place and hour to go for, why, why the runner-up lost,
and what to carry and watch for. `app/planner.py` validates the decision against
the facts before it is shown: a place or hour that is not in `FACTS` is rejected, a
sentence quoting a number the facts do not contain is dropped, and if the decision
does not survive, the engine's own pick stands.

Three jobs share one adapter: **parse** (question to slots), **decide** (the
reasoning step), **narrate** (grounded prose).

Full model card, training recipe and limitations: `hf/README.md`.
Dataset generator and training script: `training/`.
