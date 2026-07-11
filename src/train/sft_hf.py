#!/usr/bin/env python3
"""SFT Fine-tuning with HuggingFace + PEFT + TRL — no Unsloth dependency.

Usage:
    python3 src/train/sft_hf.py \
        --model PeterPaker123/Qwen2.5-7B-Vietnamese-Medical-NER-GRPO \
        --dataset data_train/sft_dataset.jsonl \
        --output models/qwen-ner-sft-hf
"""

import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Bypass HF hub validation for local model paths
import huggingface_hub.utils._validators as _hub_val
_hub_val.validate_repo_id = lambda repo_id: None


def load_jsonl(path: str, max_samples: int = None):
    data = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                item = json.loads(line)
                msgs = item.get("messages", [])
                if len(msgs) >= 3:
                    data.append({
                        "system": msgs[0]["content"],
                        "user": msgs[1]["content"][:2000],
                        "assistant": msgs[2]["content"][:2000],
                    })
            except:
                pass
    print(f"Loaded {len(data)} samples from {path}")
    return data


def format_chat(example, tokenizer):
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def main():
    parser = argparse.ArgumentParser(description="HuggingFace SFT Training")
    parser.add_argument("--model", default="PeterPaker123/Qwen2.5-7B-Vietnamese-Medical-NER-GRPO")
    parser.add_argument("--local-model-path", default=None)
    parser.add_argument("--dataset", default="data_train/sft_dataset.jsonl")
    parser.add_argument("--output", default="models/qwen-ner-sft-hf")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--use_4bit", action="store_true", default=True)
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────
    data = load_jsonl(args.dataset, args.max_samples)
    if not data:
        print("ERROR: No data"); sys.exit(1)

    # ── Load model ───────────────────────────────────────────
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer
    from transformers import TrainingArguments

    model_name = args.local_model_path or args.model
    print(f"Loading: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # ── Format data ──────────────────────────────────────────
    dataset = Dataset.from_list(data)
    dataset = dataset.map(lambda x: format_chat(x, tokenizer), remove_columns=dataset.column_names)

    # ── Train ────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        args=TrainingArguments(
            output_dir=args.output,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            warmup_steps=100,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=2,
            fp16=False,
            bf16=True,
            optim="adamw_8bit",
            report_to="none",
            save_strategy="steps",
            resume_from_checkpoint=args.resume,
        ),
        max_seq_length=args.max_seq_length,
    )

    print(f"\nTraining {len(data)} samples, {args.epochs} epochs...")
    trainer.train()

    # ── Save ─────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(output_dir / "lora"))
    tokenizer.save_pretrained(str(output_dir / "lora"))

    print(f"Saved to {output_dir}/lora")


if __name__ == "__main__":
    main()
