#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QLoRA fine-tune for Drishti's small language model.

Runs on a free Colab or Kaggle GPU. It loads an open base model of at most 3B
parameters, attaches a LoRA adapter, and trains on the three jobs the assistant
actually needs -- parsing, deciding, and narrating -- from the set that
`generate_dataset.py` writes.

    pip install -r training/requirements-train.txt
    python training/generate_dataset.py --out data --n 1500
    python training/train_lora.py --data data/all.jsonl --out out/drishti-lora

Then push it to the Hub (the hackathon requires a public repo):

    python training/train_lora.py --data data/all.jsonl --out out/drishti-lora --push <hf-user>/drishti-slm

Base model: change with --base. Anything open and <= 3B works, e.g.
    Qwen/Qwen2.5-1.5B-Instruct      (default)
    Qwen/Qwen2.5-0.5B-Instruct      (smallest, fastest)
    microsoft/Phi-3-mini-4k-instruct
    meta-llama/Llama-3.2-1B-Instruct
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/all.jsonl")
    ap.add_argument("--out", default="out/drishti-lora")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--eval-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--push", default=None, help="hub repo id, e.g. user/drishti-slm")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, DataCollatorForLanguageModeling,
                              Trainer, TrainingArguments)

    rows = load_rows(args.data)
    if not rows:
        raise SystemExit(f"no rows in {args.data}; run generate_dataset.py first")
    print(f"{len(rows)} examples from {args.data}")

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def to_text(rec):
        msgs = rec["messages"]
        try:
            return {"text": tok.apply_chat_template(msgs, tokenize=False,
                                                    add_generation_prompt=False)}
        except Exception:  # noqa: BLE001  -- models without a template
            body = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in msgs)
            return {"text": body + "<|end|>\n"}

    ds = Dataset.from_list(rows).map(to_text, remove_columns=["messages"])
    split = ds.train_test_split(test_size=args.eval_frac, seed=args.seed)

    def tok_fn(batch):
        out = tok(batch["text"], truncation=True, max_length=args.max_len)
        out["labels"] = [x[:] for x in out["input_ids"]]
        return out

    train = split["train"].map(tok_fn, batched=True, remove_columns=["text"])
    valid = split["test"].map(tok_fn, batched=True, remove_columns=["text"])

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=quant, device_map="auto",
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=10, save_strategy="epoch", eval_strategy="epoch",
        bf16=True, optim="paged_adamw_8bit", report_to="none", seed=args.seed,
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=train, eval_dataset=valid,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"adapter saved to {args.out}")

    if args.push:
        model.push_to_hub(args.push, use_auth_token=True)
        tok.push_to_hub(args.push, use_auth_token=True)
        print(f"pushed to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
