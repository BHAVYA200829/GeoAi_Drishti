"""
Push the Drishti model (or LoRA adapter) to the Hugging Face Hub.

The challenge requires a public HF repository with the weights (or your adapter
plus a pointer to the exact base model), a model card, and an open licence.

Auth: log in first with `huggingface-cli login`, or set HUGGINGFACE_TOKEN.
Never hardcode a token in this file or commit one.

Usage:
    python scripts/push_to_hub.py --repo-id your-name/drishti-slm
    python scripts/push_to_hub.py --repo-id your-name/drishti-slm --adapter ./out/lora
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile

from huggingface_hub import HfApi, create_repo, upload_folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="e.g. your-name/drishti-slm")
    ap.add_argument("--adapter", default=None, help="folder with a LoRA adapter (optional)")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--licence", default="apache-2.0")
    args = ap.parse_args()

    token = os.environ.get("HUGGINGFACE_TOKEN")  # optional; else uses cached login
    api = HfApi(token=token)
    create_repo(args.repo_id, repo_type="model", exist_ok=True, token=token)

    with tempfile.TemporaryDirectory() as tmp:
        # model card
        card = f"""---
license: {args.licence}
base_model: {args.base_model}
tags:
  - outdoor-activity
  - geospatial
  - weather
  - drishti
---

# Drishti SLM

Language component of Drishti, an outdoor-activity advisor.
Base model: `{args.base_model}` (<= 3B parameters).

## What it does
1. Parses a free-form question into structured slots.
2. Narrates a computed recommendation into grounded advice.

It never recalls or invents weather, geography or places; all facts are fetched
live and supplied as evidence. See the project README for the full system.

## How to load
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("{args.repo_id}")
model = AutoModelForCausalLM.from_pretrained("{args.repo_id}")
```

## Licence
{args.licence}. Map data (c) OpenStreetMap contributors.
"""
        with open(os.path.join(tmp, "README.md"), "w", encoding="utf-8") as f:
            f.write(card)

        if args.adapter:
            for name in os.listdir(args.adapter):
                shutil.copy(os.path.join(args.adapter, name), os.path.join(tmp, name))
        else:
            # no fine-tuning: a config + card is enough to document the base model
            with open(os.path.join(tmp, "drishti_config.json"), "w", encoding="utf-8") as f:
                f.write('{"base_model": "%s", "usage": "slot parsing + narration"}\n' % args.base_model)

        upload_folder(repo_id=args.repo_id, folder_path=tmp, repo_type="model", token=token)

    print(f"Pushed to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
