#!/usr/bin/env python3
"""Merge GRPO LoRA into base model and save final model.

Usage on Kaggle:
    python3 scripts/merge_final.py \
      --base-model /kaggle/input/datasets/bibocat/qwen3-ner-sft-merged \
      --lora-path models/qwen-ner-grpo-v2/lora \
      --output models/qwen-ner-grpo-final

Or locally (if base model available):
    python3 scripts/merge_final.py \
      --base-model /tmp/qwen3-ner-merged \
      --lora-path models/qwen-ner-grpo-v2/lora \
      --output models/qwen-ner-grpo-final
"""

import argparse, json, os, sys, torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(base_path: str, lora_path: str, output_path: str, push_to_hf: str = None):
    print(f"Base model: {base_path}")
    print(f"LoRA adapter: {lora_path}")
    print(f"Output: {output_path}")

    # Load base model
    print("Loading base model...")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)

    # Load LoRA
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, lora_path)

    # Merge
    print("Merging LoRA into base model...")
    model = model.merge_and_unload()

    # Save
    print(f"Saving merged model to {output_path}...")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    total_gb = sum(
        os.path.getsize(os.path.join(output_path, f)) / 1e9
        for f in os.listdir(output_path)
        if os.path.isfile(os.path.join(output_path, f))
    )
    print(f"Done! Model size: {total_gb:.1f}GB at {output_path}")

    if push_to_hf:
        print(f"Pushing to HuggingFace: {push_to_hf}...")
        model.push_to_hub(push_to_hf, safe_serialization=True, token=os.getenv("HF_TOKEN"))
        tokenizer.push_to_hub(push_to_hf, token=os.getenv("HF_TOKEN"))
        print("Pushed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="/tmp/qwen3-ner-merged")
    parser.add_argument("--lora-path", default="models/qwen-ner-grpo-v2/lora")
    parser.add_argument("--output", default="models/qwen-ner-grpo-final")
    parser.add_argument("--push-to-hf", default=None, help="HuggingFace repo to push to")
    args = parser.parse_args()

    if not os.path.exists(args.base_model):
        print(f"ERROR: Base model not found: {args.base_model}")
        print("On Kaggle, use: --base-model /kaggle/input/datasets/bibocat/qwen3-ner-sft-merged")
        sys.exit(1)
    if not os.path.exists(args.lora_path):
        print(f"ERROR: LoRA adapter not found: {args.lora_path}")
        sys.exit(1)

    merge(args.base_model, args.lora_path, args.output, args.push_to_hf)
