This folder is the Hugging Face model repository.

To publish it:

  1. python training/generate_dataset.py --out data --n 1500
  2. python training/train_lora.py --data data/all.jsonl --out out/drishti-lora \
         --push <your-hf-user>/drishti-slm

Step 2 writes the adapter weights (adapter_model.safetensors), the tokenizer and
adapter_config.json into out/drishti-lora, and pushes them together with the model
card above. If you prefer to push by hand:

  huggingface-cli upload <your-hf-user>/drishti-slm out/drishti-lora .
  huggingface-cli upload <your-hf-user>/drishti-slm hf/README.md README.md

The model card must name the base model and how it is used, which hf/README.md
already does.
