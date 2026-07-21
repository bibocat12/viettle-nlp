#!/bin/bash
# Serve the GRPO model via vLLM with OpenAI-compatible API
# Run this on your server with GPU

vllm serve bibocat/qwen3-ner-grpo-final \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name bibocat/qwen3-ner-grpo-final \
  --dtype bfloat16 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32
