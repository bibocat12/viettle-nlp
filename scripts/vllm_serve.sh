#!/bin/bash
# Serve the GRPO model via vLLM with OpenAI-compatible API
# Run this on your server with GPU

export HF_TOKEN=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

vllm serve bibocat/qwen3-ner-grpo-final \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3-ner-grpo \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32
