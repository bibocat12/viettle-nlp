#!/usr/bin/env python3
"""
Viettle NLP - CLI for medical NER pipeline.

Usage:
  # Process single file (rules only)
  python3 cli.py input/1.txt

  # Recommended: self-hosted Qwen3 GRPO through vLLM
  python3 cli.py input/1.txt --llm vllm \
    --served-model bibocat/qwen3-ner-grpo-v2-merged

  # Process single file with local model on GPU
  python3 cli.py input/1.txt --llm local --gpu

  # Process single file with custom model path
  python3 cli.py input/1.txt --llm local --model-path ./models/Qwen2.5-7B-Vietnamese-Medical-NER-GRPO

  # Process all files in input/ with RunPod LLM
  python3 cli.py input/ --llm runpod --output outputs_new

  # Batch process with parallel workers
  python3 cli.py input/ --llm runpod --parallel 4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# transformers 5.x validates repo IDs strictly;
# monkey-patch to allow local absolute paths like /kaggle/...
import huggingface_hub.utils._validators as _hub_val
_hub_val.validate_repo_id = lambda repo_id: None

os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def setup_device(use_gpu: bool):
    """Configure device for GPU if available."""
    import torch
    if use_gpu and torch.cuda.is_available():
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    elif use_gpu:
        print("  GPU requested but not available, using CPU")
    return "cpu"


def process_file(note_id: str, text: str, kb, llm=None, on_log=None) -> dict:
    """Process a single file through the pipeline."""
    from orchestrator import run_pipeline
    return run_pipeline(note_id, text, kb=kb, llm=llm, on_log=on_log)


def main():
    parser = argparse.ArgumentParser(description="Viettle NLP - Medical NER Pipeline CLI")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("--output", "-o", default=None, help="Output directory (default: outputs/ or outputs_new/)")
    parser.add_argument("--llm", choices=["none", "vllm", "runpod", "local"], default="none",
                        help="LLM mode: none, vllm (OpenAI-compatible), runpod, or local")
    parser.add_argument("--model", choices=["qwen", "healthbert", "deberta"], default="qwen",
                        help="Model name (default: qwen)")
    parser.add_argument("--model-path", "-m", default="./models",
                        help="Base directory for ALL models (Qwen, ViHealthBERT, ViPubMedDeBERTa). "
                             "Default: ./models")
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
                        help="vLLM OpenAI-compatible base URL")
    parser.add_argument("--served-model", default=os.getenv("VLLM_MODEL", "bibocat/qwen3-ner-grpo-v2-merged"),
                        help="Model ID exposed by vLLM")
    parser.add_argument(
        "--strategy", choices=["pipeline", "grpo-v2-fusion"], default="pipeline",
        help="Processing strategy; grpo-v2-fusion preserves an anchor submission and adds safe consensus proposals",
    )
    parser.add_argument("--anchor-zip", default=None,
                        help="Fixed baseline ZIP required by --strategy grpo-v2-fusion")
    parser.add_argument("--passes", type=int, choices=[1, 3, 7], default=3,
                        help="Extraction passes used by grpo-v2-fusion (default: 3)")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse cached chunk responses and completed fusion records")
    parser.add_argument("--validation", default=None,
                        help="Optional GRPO-v2 JSONL validation split used to calibrate fusion/link thresholds")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N numbered files (useful for a smoke run)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available (only for local model)")
    parser.add_argument("--parallel", "-p", type=int, default=1, help="Parallel workers for batch processing")
    parser.add_argument(
        "--multiscale",
        action="store_true",
        help="Add non-overlapping recall proposals from smaller 500-character chunks",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed logs")
    args = parser.parse_args()
    if args.multiscale:
        os.environ["NER_MULTISCALE"] = "1"

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    if args.strategy == "grpo-v2-fusion":
        if args.llm != "vllm":
            parser.error("--strategy grpo-v2-fusion requires --llm vllm")
        if not args.anchor_zip:
            parser.error("--strategy grpo-v2-fusion requires --anchor-zip")
        anchor_zip = Path(args.anchor_zip)
        if not anchor_zip.exists():
            parser.error(f"anchor ZIP not found: {anchor_zip}")
        output_dir = Path(args.output) if args.output else Path("lb4_grpo_v2_fusion/output")
        from grpo_v2_fusion import run_fusion_batch
        result = run_fusion_batch(
            input_path=input_path,
            output_dir=output_dir,
            anchor_zip=anchor_zip,
            base_url=args.base_url,
            model=args.served_model,
            workers=args.parallel,
            passes=args.passes,
            resume=args.resume,
            validation_path=Path(args.validation) if args.validation else None,
            limit=args.limit,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # Setup device
    device = setup_device(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if args.gpu else ""

    # Set models directory for ALL models (embedding_linking, healthbert read this)
    models_dir = os.path.abspath(args.model_path)
    os.environ["VIETTLE_MODELS_DIR"] = models_dir

    # Load KB
    print("Loading knowledge base...")
    from kb import KnowledgeBase
    kb = KnowledgeBase()
    kb.load()
    print(f"  ICD-10: {len(kb.icd10)} codes, RxNorm: {len(kb.rxnorm_concepts)} concepts")

    # Setup LLM
    llm = None
    if args.llm == "vllm":
        print(f"Connecting to vLLM model {args.served_model} at {args.base_url}...")
        from vllm_llm import get_vllm_llm
        llm = get_vllm_llm(base_url=args.base_url, model=args.served_model)
        if not llm.is_available():
            print(f"  vLLM unavailable: {llm.last_error}")
            sys.exit(2)
        print("  vLLM ready")
    elif args.llm == "runpod":
        print("Loading RunPod LLM...")
        from runpod_llm import get_runpod_llm
        llm = get_runpod_llm()
        if llm.is_available():
            print("  RunPod LLM ready")
        else:
            print("  RunPod LLM unavailable (check .env)")
            llm = None
    elif args.llm == "local":
        # Resolve model name to path
        model_dirs = {
            "qwen": "Qwen2.5-7B-Vietnamese-Medical-NER-GRPO",
            "healthbert": "vihealthbert-base-word",
            "deberta": "vipubmed-deberta-base",
        }
        model_dir = model_dirs.get(args.model, model_dirs["qwen"])
        full_path = os.path.join(args.model_path, model_dir)
        print(f"Loading local LLM: {args.model} from {full_path}...")
        from llm import LLMEngine
        llm = LLMEngine(model_path=full_path)
        if args.gpu and torch.cuda.is_available():
            llm.device = "cuda"
        llm.load()
        print("  Local LLM ready")

    # Pre-compute embeddings for linking (load model once)
    if llm:
        print("Loading ViPubMedDeBERTa for embedding linking...")
        from embedding_linking import rerank_icd10, rerank_rxnorm
        _load = __import__('embedding_linking', fromlist=['_load_model'])._load_model
        _load()
        print("  Embedding model ready")

    # Determine files to process
    if input_path.is_file():
        files = [input_path]
        default_output = "outputs_new"
    else:
        files = sorted(input_path.glob("*.txt"), key=lambda p: int(p.stem))
        default_output = "outputs_new"

    output_dir = Path(args.output) if args.output else Path(default_output)
    output_dir.mkdir(exist_ok=True)

    total = len(files)
    print(f"\nProcessing {total} file(s)...")
    print(f"  Output: {output_dir.resolve()}")
    print(f"  LLM: {args.llm}")
    print(f"  Parallel: {args.parallel}\n")

    logs_buffer = []
    def on_log(entry):
        if args.verbose:
            agent = entry.get("agent", "")
            action = entry.get("action", "")
            print(f"    [{agent}] {action}")
        logs_buffer.append(entry)

    total_mentions = 0
    t_start = time.time()

    if args.parallel > 1 and len(files) > 1:
        # Parallel batch mode
        def _process(f):
            text = f.read_text()
            result = process_file(f.stem, text, kb=kb, llm=llm, on_log=on_log)
            out = result.get("mentions", [])
            (output_dir / f"{f.stem}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
            return f.stem, len(out), result.get("stats", {})

        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures = {ex.submit(_process, f): f for f in files}
            for future in as_completed(futures):
                fid, count, stats = future.result()
                total_mentions += count
                elapsed = time.time() - t_start
                print(f"  [{fid}] {count} mentions ({elapsed:.0f}s)")
    else:
        # Sequential mode
        for i, f in enumerate(files):
            t0 = time.time()
            text = f.read_text()
            result = process_file(f.stem, text, kb=kb, llm=llm, on_log=on_log)
            out = result.get("mentions", [])
            stats = result.get("stats", {})
            total_mentions += len(out)

            (output_dir / f"{f.stem}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))

            elapsed = time.time() - t0
            print(f"  [{f.stem}] {len(out)} mentions ({elapsed:.0f}s)"
                  f" | Qwen={stats.get('ner_qwen_mentions', '-')}"
                  f" HB={stats.get('ner_healthbert_mentions', '-')}"
                  f" Rules={stats.get('rule_mentions', '-')}")

    t_total = time.time() - t_start
    print(f"\nDone. {total} files, {total_mentions} total mentions in {t_total:.0f}s ({t_total/max(total,1):.1f}s/file)")
    print(f"Output saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    # Import torch at module level for GPU detection
    import torch
    main()
