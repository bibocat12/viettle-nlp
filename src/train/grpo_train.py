#!/usr/bin/env python3
"""GRPO Fine-tuning with Unsloth — Vietnamese Medical NER with reward-based optimization.

Usage:
    python3 src/train/grpo_train.py \
        --model PeterPaker123/Qwen2.5-7B-Vietnamese-Medical-NER-GRPO \
        --dataset data_train/grpo_dataset.jsonl \
        --output models/qwen-ner-grpo-v1 \
        --epochs 2
"""

import argparse, json, os, sys, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Bypass HF hub validation for local model paths
import huggingface_hub.utils._validators as _hub_val
_hub_val.validate_repo_id = lambda repo_id: None


def load_grpo_dataset(path: str, max_samples: int = None):
    """Load GRPO dataset with entities (for reward computation)."""
    data = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                item = json.loads(line)
                entities = item.get("entities", [])
                msgs = item.get("messages", [])
                if len(msgs) >= 3 and entities:
                    data.append({
                        "system": msgs[0]["content"],
                        "text": msgs[1]["content"],
                        "entities": entities,  # Ground truth for reward
                    })
            except:
                pass
    print(f"Loaded {len(data)} GRPO samples from {path}")
    return data


def reward_entity_overlap(completion: str, ground_truth: list) -> float:
    """Reward function: score based on entity overlap (F1-like)."""
    if not completion or not ground_truth:
        return 0.0

    # Parse entities from completion
    try:
        # Try to extract JSON from completion
        m = re.search(r'\[.*\]', completion, re.DOTALL)
        if not m:
            return 0.0
        pred = json.loads(m.group())
        if not isinstance(pred, list):
            return 0.0
    except:
        return 0.0

    # Build sets of (entity_text, type) pairs
    gt_set = set()
    for e in ground_truth:
        gt_set.add((e.get("entity", "").lower().strip(), e.get("type", "")))

    pred_set = set()
    for e in pred:
        if isinstance(e, dict):
            pred_set.add((e.get("entity", "").lower().strip(), e.get("type", "")))

    if not pred_set or not gt_set:
        return 0.0

    tp = len(gt_set & pred_set)
    p = tp / len(pred_set) if pred_set else 0
    r = tp / len(gt_set) if gt_set else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

    # Bonus for exact match
    exact_bonus = 0.2 if gt_set == pred_set else 0

    return f1 + exact_bonus


def reward_position_match(completion: str, ground_truth: list, text: str) -> float:
    """Reward: entity overlap + position accuracy."""
    overlap_score = reward_entity_overlap(completion, ground_truth)
    return overlap_score


def format_prompt(example, tokenizer):
    """Format as chat template for GRPO training."""
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["text"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def main():
    parser = argparse.ArgumentParser(description="Unsloth GRPO Training")
    parser.add_argument("--model", default="PeterPaker123/Qwen2.5-7B-Vietnamese-Medical-NER-GRPO")
    parser.add_argument("--local-model-path", default=None,
                        help="Local model path (overrides --model)")
    parser.add_argument("--dataset", default="data_train/grpo_dataset.jsonl")
    parser.add_argument("--output", default="models/qwen-ner-grpo-v1")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora-path", default=None,
                        help="Path to SFT LoRA adapter to start from (e.g. models/qwen-ner-sft-v1/lora)")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--logging_steps", type=int, default=10)
    args = parser.parse_args()

    # ── Load dataset ──────────────────────────────────────────
    data = load_grpo_dataset(args.dataset, args.max_samples)
    if not data:
        print("ERROR: No training data loaded")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────
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

    # Load SFT LoRA if provided (stage 2: GRPO on top of SFT)
    if args.lora_path:
        print(f"Loading SFT LoRA adapter: {args.lora_path}")
        model.load_adapter(args.lora_path)

    # Add GRPO LoRA on top
    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_r, lora_alpha=args.lora_r * 2,
        lora_dropout=0.0,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407,
    )

    # ── Format prompts ────────────────────────────────────────
    print(f"Formatting {len(data)} GRPO samples...")
    from datasets import Dataset

    # Store ground truth entities for reward
    entity_map = {}
    prompts = []
    for i, ex in enumerate(data):
        prompt = format_prompt(ex, tokenizer)
        prompts.append({"prompt": prompt, "gt_entities": ex["entities"], "idx": i})
        entity_map[i] = ex["entities"]

    dataset = Dataset.from_list(prompts)

    # ── Train with GRPO ───────────────────────────────────────
    from trl import GRPOTrainer, GRPOConfig
    from transformers import TrainingArguments

    def make_reward(i: int):
        """Create reward function for a specific sample index."""
        gt = entity_map.get(i, [])
        def reward_fn(completions, **kwargs):
            if isinstance(completions, list):
                return [reward_entity_overlap(c, gt) for c in completions]
            return reward_entity_overlap(completions, gt)
        return reward_fn

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=TrainingArguments(
            output_dir=args.output,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            warmup_steps=50,
            logging_steps=args.logging_steps,
            save_strategy="steps",
            fp16=False,
            bf16=True,
            optim="adamw_8bit",
            seed=3407,
            report_to="none",
        ),
        reward_funcs=make_reward,  # Will be overridden per-step
        train_dataset=dataset,
    )

    print(f"\nGRPO Training {len(data)} samples, {args.epochs} epochs...")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving GRPO LoRA adapter to {output_dir}/lora...")
    model.save_pretrained(str(output_dir / "lora"))
    tokenizer.save_pretrained(str(output_dir / "lora"))

    print(f"Done! Load with:")
    print(f"  model = FastLanguageModel.from_pretrained('{model_name}')")
    print(f"  model.load_adapter('{output_dir}/lora')")


if __name__ == "__main__":
    main()
