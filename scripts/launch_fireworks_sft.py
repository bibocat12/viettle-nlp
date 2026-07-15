#!/usr/bin/env python3
"""Convert NER dataset to Fireworks SFT format (with tools) and launch job.

Usage:
    export FIREWORKS_API_KEY=fw_xxx
    python3 scripts/launch_fireworks_sft.py --dataset data_train/grpo_combined-100.jsonl
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "search_icd10",
            "description": "Tra cứu mã ICD-10 theo tên bệnh tiếng Việt hoặc tiếng Anh.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Tên bệnh / chẩn đoán cần tra (tiếng Việt). VD: tăng huyết áp, lao phổi",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_rxnorm",
            "description": "Tra cứu mã RxNorm theo tên thuốc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Tên thuốc cần tra. VD: metformin, atorvastatin"}
                },
                "required": ["query"],
            },
        },
    },
]


def convert_to_fireworks_format(input_path: str, output_path: str):
    """Convert grpo_combined JSONL to Fireworks SFT format with tools."""
    converted = 0
    skipped = 0
    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            d = json.loads(line)
            messages = d.get("messages", [])
            entities = d.get("entities", [])

            if len(messages) < 3:
                skipped += 1
                continue

            system_msg = messages[0]
            user_msg = messages[1]

            # Build assistant response with entities
            # Include tool_calls for entities that have candidates
            tool_calls = []
            final_entity_list = []

            for e in entities:
                ent_text = e.get("entity") or e.get("text", "")
                etype = e.get("type", "")
                pos = e.get("position", [0, 0])
                candidates = e.get("candidates", [])

                final_entity_list.append({
                    "entity": ent_text,
                    "type": etype,
                    "position": pos,
                })

                # Add tool calls for diagnosis/drug entities with candidates
                if candidates and etype == "CHẨN_ĐOÁN":
                    tool_calls.append({
                        "type": "function",
                        "function": {
                            "name": "search_icd10",
                            "arguments": json.dumps({"query": ent_text}, ensure_ascii=False),
                        },
                    })
                elif candidates and etype == "THUỐC":
                    tool_calls.append({
                        "type": "function",
                        "function": {
                            "name": "search_rxnorm",
                            "arguments": json.dumps({"query": ent_text}, ensure_ascii=False),
                        },
                    })

            fw_messages = [system_msg, user_msg]

            # Add tool_calls if any
            if tool_calls:
                # Add candidate results as tool responses
                fw_messages.append({
                    "role": "assistant",
                    "tool_calls": tool_calls,
                })
                for tc in tool_calls:
                    cname = tc["function"]["name"]
                    cquery = json.loads(tc["function"]["arguments"])["query"]
                    fw_messages.append({
                        "role": "tool",
                        "name": cname,
                        "content": json.dumps([{"query": cquery}], ensure_ascii=False),
                    })

            # Final assistant response
            fw_messages.append({
                "role": "assistant",
                "content": json.dumps(final_entity_list, ensure_ascii=False),
            })

            out = {"tools": TOOLS_DEFINITION, "messages": fw_messages}
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            converted += 1

    print(f"Converted {converted} samples to {output_path} ({skipped} skipped)")
    return output_path


def upload_and_launch(dataset_path: str, api_key: str):
    """Upload dataset and launch SFT job on Fireworks."""
    import time
    from fireworks import Fireworks

    client = Fireworks(account_id="accounts/giaphuc190206-eww3fr", api_key=api_key)

    # 1. Create dataset
    dataset_id = "ner-grpo-100-tool"
    print(f"Creating dataset '{dataset_id}'...")
    ds = client.datasets.create(
        dataset_id=dataset_id,
        dataset={"user_uploaded": {}},
        example_count=100,
    )
    print(f"  Dataset: {ds.name}")

    # 2. Upload file
    print(f"Uploading {dataset_path}...")
    with open(dataset_path, "rb") as f:
        upload = client.datasets.upload(
            dataset_id=dataset_id,
            file=f,
        )
    print(f"  Uploaded: {upload}")

    # 3. Launch SFT job
    job_id = "ner-sft-tool-100"
    print(f"Launching SFT job '{job_id}'...")
    try:
        job = client.supervised_fine_tuning_jobs.create(
            job_id=job_id,
            base_model="accounts/fireworks/models/qwen3-8b",
            dataset=dataset_id,
            output_model="ner-grpo-tool-v1",
            epochs=2,
            lora_rank=16,
        )
        print(f"  SFT Job: {job.name}")
        print(f"  Status: {job.status}")
    except Exception as e:
        print(f"  Error: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data_train/grpo_combined-100.jsonl")
    parser.add_argument("--output", default="data_train/fireworks_sft_100.jsonl")
    parser.add_argument("--launch", action="store_true", default=True)
    parser.add_argument("--api-key", default=os.getenv("FIREWORKS_API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: FIREWORKS_API_KEY not set")
        sys.exit(1)

    # Convert
    out = convert_to_fireworks_format(args.dataset, args.output)

    # Launch
    if args.launch:
        upload_and_launch(out, args.api_key)


if __name__ == "__main__":
    main()
