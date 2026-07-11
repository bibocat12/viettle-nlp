#!/usr/bin/env python3
"""SFT Fine-tuning with Unsloth — Qwen2.5-7B Vietnamese Medical NER.

Usage:
    python3 src/train/sft_train.py \
        --model PeterPaker123/Qwen2.5-7B-Vietnamese-Medical-NER-GRPO \
        --dataset data_train/sft_dataset.jsonl \
        --output models/qwen-ner-sft-v1 \
        --epochs 3 --lora_r 16 --max_seq_length 4096
"""

import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Bypass HF hub validation for local model paths
import huggingface_hub.utils._validators as _hub_val
_hub_val.validate_repo_id = lambda repo_id: None

# transformers 5.x validates repo IDs strictly; allow local paths
import huggingface_hub.utils._validators as _hub_val
_hub_val.validate_repo_id = lambda repo_id: None
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_jsonl(path: str, max_samples: int = None):
    """Load JSONL dataset with messages format."""
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


def format_chat_template(example, tokenizer):
    """Format messages for Qwen chat template."""
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def main():
    parser = argparse.ArgumentParser(description="Unsloth SFT Training")
    parser.add_argument("--model", default="PeterPaker123/Qwen2.5-7B-Vietnamese-Medical-NER-GRPO",
                        help="HuggingFace model name (used if --local-model-path not set)")
    parser.add_argument("--local-model-path", default=None,
                        help="Local path to model (e.g. ./models/Qwen2.5-7B). Overrides --model.")
    parser.add_argument("--dataset", default="data_train/sft_dataset.jsonl")
    parser.add_argument("--output", default="models/qwen-ner-sft-v1")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora-path", default=None,
                        help="Path to existing LoRA adapter to continue training from")
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint directory (e.g. models/qwen-ner-sft-v1/checkpoint-1500)")
    args = parser.parse_args()

    # ── Load dataset ──────────────────────────────────────────
    data = load_jsonl(args.dataset, args.max_samples)
    if not data:
        print("ERROR: No training data loaded")
        sys.exit(1)

    # ── Load model (fresh training) ───────────────────────────
    model_name = args.local_model_path or args.model
    print(f"Loading model: {model_name}")
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")

    # Load existing LoRA adapter to continue training
    if args.lora_path:
        print(f"Loading LoRA adapter: {args.lora_path}")
        model.load_adapter(args.lora_path)

    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407,
    )
    # ── Format data ───────────────────────────────────────────
    print(f"Formatting {len(data)} samples...")
    from datasets import Dataset
    dataset = Dataset.from_list(data)
    dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=dataset.column_names,
    )
    
    # ── Train ─────────────────────────────────────────────────
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainingArguments
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=TrainingArguments(
            output_dir=args.output,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            warmup_steps=100,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=3,
            fp16=False,
            bf16=True,
            optim="adamw_8bit",
            seed=3407,
            report_to="none",
            resume_from_checkpoint=args.resume,
            save_strategy="no",        # Disable save to avoid SFTConfig pickle bug
        ),
    )
    
    print(f"\nTraining {len(data)} samples, {args.epochs} epochs...")
    try:
        trainer.train()
    except Exception as e:
        print(f"Training interrupted: {e}")
        print("Model weights are still in memory — saving anyway...")
    
    # ── Save ──────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving LoRA adapter to {output_dir}/lora...")
    model.save_pretrained(str(output_dir / "lora"))
    tokenizer.save_pretrained(str(output_dir / "lora"))
    
    print(f"Done! Load with:")
    print(f"  model = FastLanguageModel.from_pretrained('{model_name}')")
    print(f"  model.load_adapter('{output_dir}/lora')")


if __name__ == "__main__":
    main()
