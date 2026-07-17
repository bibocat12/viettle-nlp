#!/usr/bin/env python3
"""Merge LoRA adapter into base model and upload to Fireworks as full model.

Usage:
    export FIREWORKS_API_KEY=fw_xxx
    python3 scripts/merge_and_upload.py \
        --base-model models/Qwen2.5-7B-Vietnamese-Medical-NER-GRPO \
        --lora-path models/qwen-ner-sft-4epochs-100k \
        --output /tmp/merged-model \
        --upload
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def merge(base_model_path: str, lora_path: str, output_path: str):
    """Merge LoRA into base model and save full model."""
    import torch
    from unsloth import FastLanguageModel

    print(f"Loading base model: {base_model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=4096,
        load_in_4bit=True,
        dtype=torch.float16,
    )

    print(f"Loading LoRA adapter: {lora_path}")
    model.load_adapter(lora_path)

    print("Merging LoRA into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Merge complete!")

    return output_path


def upload_to_fireworks(model_path: str, model_id: str, api_key: str):
    """Upload merged model to Fireworks as HF_BASE_MODEL."""
    import requests

    account_id = "giaphuc190206-eww3fr"
    base_url = "https://api.fireworks.ai/v1"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # List required model files
    model_files = [
        "config.json", "model.safetensors", "model.safetensors.index.json",
        "tokenizer.json", "tokenizer_config.json",
    ]
    # Check what actually exists
    files = [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]
    file_sizes = {f: os.path.getsize(os.path.join(model_path, f)) for f in files}

    total_gb = sum(file_sizes.values()) / 1e9
    print(f"\nUploading {len(files)} files ({total_gb:.1f}GB) to Fireworks as '{model_id}'...")

    # Step 1: Create model entry
    print("Creating model entry...")
    huggingface_url = "https://huggingface.co/Qwen/Qwen3-8B"

    create_payload = {
        "modelId": model_id,
        "model": {
            "kind": "HF_BASE_MODEL",
            "huggingFaceUrl": huggingface_url,
            "baseModelDetails": {
                "checkpointFormat": "HUGGINGFACE",
                "worldSize": 1,
                "huggingfaceFiles": files,
            },
        },
    }

    resp = requests.post(
        f"{base_url}/accounts/{account_id}/models",
        headers=headers,
        json=create_payload,
    )
    if resp.status_code == 409:
        print("Model already exists, continuing with upload...")
    else:
        resp.raise_for_status()
        print("Model entry created.")

    # Step 2: Get signed upload URLs
    print("Getting upload URLs...")
    resp = requests.post(
        f"{base_url}/accounts/{account_id}/models/{model_id}:getUploadEndpoint",
        headers=headers,
        json={"filenameToSize": file_sizes, "enableResumableUpload": True},
    )
    resp.raise_for_status()
    upload_urls = resp.json()["filenameToSignedUrls"]
    print(f"Got {len(upload_urls)} upload URLs.")

    # Step 3: Upload files
    for filename, url in upload_urls.items():
        size = file_sizes[filename]
        filepath = os.path.join(model_path, filename)
        print(f"  Uploading {filename} ({size/1e9:.1f}GB)...", flush=True)
        with open(filepath, "rb") as f:
            r = requests.put(
                url, data=f,
                headers={
                    "Content-Type": "application/octet-stream",
                    "x-goog-content-length-range": f"{size},{size}",
                },
            )
            r.raise_for_status()
        print(f"    Done.")

    # Step 4: Validate
    print("Validating upload...")
    validate_url = f"{base_url}/accounts/{account_id}/models/{model_id}:validateUpload"
    for attempt in range(60):
        resp = requests.get(validate_url, headers=headers)
        if resp.status_code == 200:
            print(f"\nModel '{model_id}' is READY on Fireworks!")
            print(f"  https://app.fireworks.ai/dashboard/models/{model_id}")
            return
        if "FAILED_PRECONDITION" in resp.text:
            if attempt % 5 == 0:
                print(f"  Waiting for validation... ({attempt}/60)")
            time.sleep(10)
            continue
        print(f"  Error: {resp.text[:300]}")
        return

    print("Validation timed out.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="/kaggle/input/models/qwen-lm/qwen-3/transformers/8b/1")
    parser.add_argument("--lora-path", default="models/qwen-ner-sft-4epochs-100k")
    parser.add_argument("--output", default="/tmp/merged-model")
    parser.add_argument("--model-id", default="ner-sft-merged-v1")
    parser.add_argument("--upload", action="store_true", help="Upload to Fireworks after merging")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge, just upload existing")
    parser.add_argument("--api-key", default=os.getenv("FIREWORKS_API_KEY"))
    args = parser.parse_args()

    if not args.skip_merge:
        merge(args.base_model, args.lora_path, args.output)
    else:
        print(f"Skipping merge, using existing: {args.output}")

    if args.upload:
        if not args.api_key:
            print("ERROR: FIREWORKS_API_KEY not set")
            sys.exit(1)
        upload_to_fireworks(args.output, args.model_id, args.api_key)


if __name__ == "__main__":
    main()
