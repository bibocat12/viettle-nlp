#!/usr/bin/env python3
"""GRPO Fine-tuning with Unsloth — Vietnamese Medical NER.

Reward function based on:
  final_score = 0.3 * text_score + 0.3 * assertions_score + 0.4 * candidates_score

- text_score: 1 - Word Error Rate (character-level)
- assertions_score: Jaccard similarity of assertion sets
- candidates_score: Jaccard similarity of candidate sets (weighted by GT size)

Usage:
    python3 src/train/grpo_train.py \
        --model PeterPaker123/Qwen2.5-7B-Vietnamese-Medical-NER-GRPO \
        --dataset data_train/grpo_combined.jsonl \
        --output models/qwen-ner-grpo-v1 \
        --epochs 2
"""

import argparse, json, logging, os, re, sys, warnings
from pathlib import Path

import torch

# Suppress noise from transformers deprecation warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*is_flash_linear_attention_available.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

sys.path.insert(0, str(Path(__file__).parent.parent))

import huggingface_hub.utils._validators as _hub_val
_hub_val.validate_repo_id = lambda repo_id: None


# ── Dataset loading ──────────────────────────────────────────────────────

def load_grpo_dataset(path: str, max_samples: int = None):
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
                        "entities": entities,
                    })
            except:
                pass
    print(f"Loaded {len(data)} GRPO samples from {path}")
    return data


# ── Parsing ──────────────────────────────────────────────────────────────

def parse_completion(completion: str) -> list:
    """Parse model output into list of entity dicts."""
    if not completion:
        return []
    m = re.search(r'\[.*\]', completion, re.DOTALL)
    if not m:
        return []
    try:
        pred = json.loads(m.group())
        if not isinstance(pred, list):
            return []
        return pred
    except:
        return []


# ── Word Error Rate (character-level) ────────────────────────────────────

def wer_char(ref: str, hyp: str) -> float:
    """Character-level WER using Levenshtein distance."""
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    if not hyp:
        return 1.0
    r, c = len(ref) + 1, len(hyp) + 1
    dp = [[0] * c for _ in range(r)]
    for i in range(r):
        dp[i][0] = i
    for j in range(c):
        dp[0][j] = j
    for i in range(1, r):
        for j in range(1, c):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1] / max(len(ref), len(hyp))


# ── Jaccard similarity ───────────────────────────────────────────────────

def jaccard(gt_set: set, pred_set: set) -> float:
    if not gt_set and not pred_set:
        return 1.0
    if not gt_set:
        return 0.0
    return len(gt_set & pred_set) / len(gt_set | pred_set)


# ── Reward function ──────────────────────────────────────────────────────

def compute_reward(completion: str, ground_truth: list, reference_text: str = "") -> float:
    """Compute final reward = 0.3 * text_score + 0.3 * assertions_score + 0.4 * candidates_score.

    ground_truth: list of dicts with keys: entity (or text), type, position
                 optionally: assertions (list[str]), candidates (list[str])
    """
    pred_entities = parse_completion(completion)
    if not pred_entities and not ground_truth:
        return 1.0  # both empty → perfect
    if not pred_entities:
        return 0.0

    # Normalize: ground truth entities use "entity" key (old) or "text" key (synthetic)
    def get_entity_text(e):
        return e.get("entity") or e.get("text") or ""

    # ── Align predictions ↔ ground truth ──
    # Key by (normalized_text, type) to catch wrong-type penalties per spec
    gt_by_key = {}
    for e in ground_truth:
        key = (get_entity_text(e).lower().strip(), e.get("type", ""))
        gt_by_key.setdefault(key, []).append(e)

    pred_by_key = {}
    for e in pred_entities:
        if not isinstance(e, dict):
            continue
        key = (e.get("entity", "").lower().strip(), e.get("type", ""))
        pred_by_key.setdefault(key, []).append(e)

    all_keys = set(gt_by_key) | set(pred_by_key)

    text_scores = []
    assertion_scores = []
    candidate_scores = []
    candidate_weights = []

    for key in all_keys:
        gt_list = gt_by_key.get(key, [])
        pred_list = pred_by_key.get(key, [])

        # Spec: if key exists in only GT or only pred → type mismatch
        # → entity counted twice, each gets 0 for ALL three metrics
        if not gt_list or not pred_list:
            text_scores.append(0.0)
            assertion_scores.append(0.0)
            w = 1
            candidate_scores.append(0.0)
            candidate_weights.append(w)
            continue

        gt = gt_list[0]
        pred = pred_list[0]

        # text_score: 1 - WER on entity text (character-level)
        gt_text = get_entity_text(gt)
        pred_text = pred.get("entity", "")
        text_scores.append(1.0 - wer_char(gt_text, pred_text))

        # assertions_score: Jaccard on assertion sets
        gt_assertions = set(gt.get("assertions", []))
        pred_assertions = set(pred.get("assertions", []))
        assertion_scores.append(jaccard(gt_assertions, pred_assertions))

        # candidates_score: Jaccard on candidate sets (weighted by GT size)
        gt_candidates = set(gt.get("candidates", []))
        pred_candidates = set(pred.get("candidates", []))
        cs = jaccard(gt_candidates, pred_candidates)

        # Weight = len(GT candidates) + 1
        w = len(gt_candidates) + 1
        candidate_scores.append(cs)
        candidate_weights.append(w)

    # Aggregate
    n = max(len(text_scores), 1)
    text_score = sum(text_scores) / n
    assertions_score = sum(assertion_scores) / n

    if candidate_weights and sum(candidate_weights) > 0:
        candidates_score = sum(s * w for s, w in zip(candidate_scores, candidate_weights)) / sum(candidate_weights)
    else:
        candidates_score = 0.0

    final = 0.3 * text_score + 0.3 * assertions_score + 0.4 * candidates_score
    return final


# ── GRPO trainer wrapper ─────────────────────────────────────────────────

def make_reward_fn(entity_map: dict):
    """Return a reward function compatible with GRPOTrainer.

    GRPOTrainer calls reward_func(completions, **kwargs) where each completion
    may be a string (TRL <0.15) or a list-of-dicts (Unsloth/TRL >= 0.15):
      [{"content": "model output text"}, ...].

    **kwargs contains dataset columns ('idx', 'gt_entities', etc).
    """
    def reward_fn(completions, **kwargs):
        scores = []
        indices = kwargs.get("idx", None)
        for i, completion in enumerate(completions):
            # Unsloth passes completions as list[list[dict]]: [[{content: ...}]]
            if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
                text = completion[0].get("content", "")
            elif isinstance(completion, dict):
                text = completion.get("content", "")
            else:
                text = str(completion) if completion else ""
            idx = indices[i] if indices else i
            gt = entity_map.get(idx, [])
            scores.append(compute_reward(text, gt))
        return scores
    return reward_fn


def format_prompt(example, tokenizer):
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["text"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unsloth GRPO Training")
    parser.add_argument("--model", default="PeterPaker123/Qwen2.5-7B-Vietnamese-Medical-NER-GRPO")
    parser.add_argument("--local-model-path", default=None)
    parser.add_argument("--dataset", default="data_train/grpo_combined.jsonl")
    parser.add_argument("--output", default="models/qwen-ner-grpo-v1")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Completions per prompt (default=4). Lower=faster but less stable.")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Max new tokens per generation (default=128).")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        help="Path to a checkpoint directory to resume from.")
    parser.add_argument("--save-steps", type=int, default=100,
                        help="Save checkpoint every N steps (default=100, frequent for GRPO).")
    parser.add_argument("--save-total-limit", type=int, default=3,
                        help="Keep only the N most recent checkpoints.")
    parser.add_argument("--sft-beta", type=float, default=0.0,
                        help="SFT loss weight (0=GRPO only). Only works if TRL supports it.")
    parser.add_argument("--use-vllm", action="store_true", default=False,
                        help="Use vLLM for 2x faster inference during GRPO generation.")
    parser.add_argument("--loss-type", default="grpo",
                        choices=["grpo", "bnpo", "dr_grpo", "dapo"],
                        help="GRPO loss variant (default: grpo).")
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="Use bfloat16 (for RTX 6000 Ada/Blackwell/AMD MI300X). Default: fp16.")
    parser.add_argument("--fp8", action="store_true", default=False,
                        help="Use FP8 for 2x faster training (needs H100/MI300X + Unsloth 2026.7+).")
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
        load_in_4bit=not args.fp8,
        fast_inference=args.use_vllm,
        dtype=torch.float16 if not args.bf16 else torch.bfloat16,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")

    if args.lora_path:
        print(f"Loading SFT LoRA adapter: {args.lora_path}")
        model.load_adapter(args.lora_path)

        # GPU doesn't support bf16; LoRA adapter may be saved in bf16.
        # Convert all LoRA weights to float16 to match base compute dtype.
        target_modules = [model]
        if hasattr(model, "base_model"):
            target_modules.append(model.base_model)
            if hasattr(model.base_model, "model"):
                target_modules.append(model.base_model.model)
        for mod in target_modules:
            for name, param in mod.named_parameters():
                if "lora_" in name and param.dtype == torch.bfloat16:
                    param.data = param.data.to(torch.float16)

        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()

    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_r, lora_alpha=args.lora_r * 2,
        lora_dropout=0.0,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407,
    )

    # ── Format prompts ────────────────────────────────────────
    print(f"Formatting {len(data)} GRPO samples...")
    from datasets import Dataset

    entity_map = {}
    prompts = []
    for i, ex in enumerate(data):
        prompt = format_prompt(ex, tokenizer)
        prompts.append({"prompt": prompt, "gt_entities": ex["entities"], "idx": i})
        entity_map[i] = ex["entities"]

    dataset = Dataset.from_list(prompts)

    # ── Train with GRPO ───────────────────────────────────────
    from trl import GRPOTrainer, GRPOConfig

    grpo_kwargs = dict(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=50,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=not args.bf16,
        bf16=args.bf16,
        optim="adamw_8bit",
        seed=3407,
        report_to="none",
        num_generations=args.num_generations,
        use_vllm=args.use_vllm,
        loss_type=args.loss_type,
        mask_truncated_completions=False,
    )
    if args.sft_beta > 0:
        grpo_kwargs["sft_beta"] = args.sft_beta

    try:
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            args=GRPOConfig(**grpo_kwargs),
            reward_funcs=make_reward_fn(entity_map),
            train_dataset=dataset,
        )
    except TypeError as e:
        if "sft_beta" in str(e):
            print("WARN: TRL doesn't support sft_beta, retrying without it...", flush=True)
            del grpo_kwargs["sft_beta"]
            trainer = GRPOTrainer(
                model=model, processing_class=tokenizer,
                args=GRPOConfig(**grpo_kwargs),
                reward_funcs=make_reward_fn(entity_map),
                train_dataset=dataset,
            )
        else:
            raise

    # Save training args + dataset stats for reproducibility & resuming
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    # Save entity stats for tracking
    total_entities = sum(len(ex["entities"]) for ex in data)
    with open(output_dir / "train_info.json", "w") as f:
        json.dump({
            "num_samples": len(data),
            "total_entities": total_entities,
            "dataset_path": args.dataset,
            "max_samples": args.max_samples,
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved training info to {output_dir}/", flush=True)

    print(f"\nGRPO Training {len(data)} samples, {args.epochs} epochs...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ── Save final LoRA ───────────────────────────────────────────
    print(f"Saving GRPO LoRA adapter to {output_dir}/lora...")
    model.save_pretrained(str(output_dir / "lora"))
    tokenizer.save_pretrained(str(output_dir / "lora"))

    print(f"Done! Load with:")
    print(f"  model = FastLanguageModel.from_pretrained('{model_name}')")
    print(f"  model.load_adapter('{output_dir}/lora')")


if __name__ == "__main__":
    main()
