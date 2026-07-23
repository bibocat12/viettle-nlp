# GRPO v2 — Vietnamese Medical NER

This dataset uses the exact five entity types and final JSON schema required by
the competition. It was generated only from the existing training corpus;
leaderboard inputs were not used as labels.

## Dataset families

- `extraction_*`: all cleaned span/type examples. Use this for initial SFT.
- `full_*`: rows where every diagnosis/drug entity has trustworthy candidate
  supervision. Use this as the main GRPO dataset.
- `linking_*`: candidate-positive subset of `full_*`. Oversample during the
  linking phase; do not train on it alone.
- `assertion_*`: conservative assertion-positive subset. Oversample with
  negative examples from `full_*`; do not train on it alone.
- `assertion_synthetic_*`: balanced single-entity present/negated/historical/
  family examples. Mix with natural examples; do not use it as the only GRPO
  corpus.

Every family has `train`, `validation`, and `test` splits. Natural corpora are
split by SHA-256 of normalized source text. Synthetic assertion examples are
grouped by `(type, normalized entity)` before splitting, so the same entity
cannot cross partitions. Never train on `validation` or `test`.

`entities` contains private reward metadata:

- `_candidate_supervised`: whether candidate reward is safe for that entity.
- `_candidate_source`: `reviewed_alias`, `kb_semantic`, `not_linkable`, or an
  unsupervised reason.
- `_assertion_supervised`: whether assertion reward is enabled.

These private keys are deliberately absent from the assistant completion.

## Recommended curriculum

1. SFT on `extraction_train.jsonl`; select checkpoint using
   `extraction_validation.jsonl`.
2. GRPO on `full_train.jsonl` with `scripts/reward_v2.py`.
3. Short specialization phase using a mixture approximating:
   `full:linking:assertion:assertion_synthetic = 4:2:1:1`.
4. Select only against validation splits. Run test splits once after choices
   are frozen.

Do not continue training with the old exact-key reward. The local trainer now
defaults to reward v2:

```bash
python3 src/train/grpo_train.py \
  --model bibocat/qwen3-ner-grpo-final \
  --dataset data_train/grpo_v2/full_train.jsonl \
  --reward-version v2 \
  --output models/qwen3-ner-grpo-v2 \
  --learning_rate 5e-6 \
  --epochs 1 \
  --num_generations 4 \
  --max_seq_length 2048 \
  --max_new_tokens 512 \
  --use-vllm
```

For a faster run, keep `--use-vllm`, use `--bf16` on supported GPUs, and use
`--num_generations 2` only for smoke tests or speed-focused experiments. The
trainer passes `max_new_tokens` to TRL as `max_completion_length`, preventing
unbounded rollouts.

Start with one epoch and evaluate before increasing it. The candidate and
assertion subsets are small enough to overfit quickly.

## Rebuild

```bash
python3 scripts/build_grpo_v2_dataset.py
pytest -q tests/test_reward_v2.py tests/test_pipeline_regressions.py
```

See `manifest.json` for counts and SHA-256 checksums.
