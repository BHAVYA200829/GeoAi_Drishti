# Fine-tuning Drishti's small language model

The running assistant uses a small open model for three jobs, and this folder
trains it for all three:

| job | input | output |
|---|---|---|
| **parse** | a free-form question | strict JSON slots (activity, place, radius, time, persona) |
| **decide** | the scored comfort surface (`FACTS`) | which place and hour, why, and why the runner-up lost |
| **narrate** | the recommendation plus its evidence | a short grounded answer |

The **decide** job is the point of the model: the deterministic engine fetches and
scores, the model chooses between real options, and `app/planner.py` validates the
choice before anything is shown. A place or an hour that is not in the facts is
rejected, and a sentence quoting a number the facts do not contain is dropped.

## Run it

```bash
pip install -r training/requirements-train.txt

python training/generate_dataset.py --out data --n 1500
python training/train_lora.py --data data/all.jsonl --out out/drishti-lora

# and push the adapter to the Hub (the submission needs a public repo)
python training/train_lora.py --data data/all.jsonl --out out/drishti-lora \
       --push <your-hf-user>/drishti-slm
```

A free Colab or Kaggle GPU is enough. `--base Qwen/Qwen2.5-0.5B-Instruct` trains
in a few minutes if you want to iterate quickly.

## Wiring the adapter into the assistant

```bash
export DRISHTI_ENABLE_SLM=1
export DRISHTI_MODEL_ID=out/drishti-lora        # or <your-hf-user>/drishti-slm
python -m uvicorn app.main:app --port 8000
```

The assistant reports which layer decided each answer in `decided_by`
(`slm` or `engine`) and shows it in `/health`.

## Honest note

The dataset is generated from realistic Indian ranges, not from real observed
weather. It teaches the model the *shape* of the job -- choose only from the
given options, quote no number you were not given, keep the advice short -- not
the climate of any particular city. The live data still comes from Open-Meteo,
OpenStreetMap, Nominatim and OSRM at query time.
