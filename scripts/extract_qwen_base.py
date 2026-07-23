#!/usr/bin/env python3
"""Run only the 1200-character Qwen extraction pass for a directory."""

import argparse
import json
import os
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ner_tool import ner_and_type
from vllm_llm import VLLMLLM


def extract(index: int, inputs: Path, output: Path, base_url: str, model: str) -> tuple[int, int]:
    raw = (inputs / f"{index}.txt").read_text(encoding="utf-8")
    normalized = unicodedata.normalize("NFC", raw).replace("\u00a0", " ")
    llm = VLLMLLM(base_url=base_url, model=model, timeout=240, max_retries=3)
    mentions = ner_and_type(normalized, llm=llm)["mentions"]
    for mention in mentions:
        for field in ("source", "type_candidates"):
            mention.pop(field, None)
        mention.setdefault("assertions", [])
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{index}.json").write_text(
        json.dumps(mentions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return index, len(mentions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="bibocat/qwen3-ner-grpo-final")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    os.environ.pop("NER_MULTISCALE", None)
    total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                extract, index, args.inputs, args.output, args.base_url, args.model
            ): index
            for index in range(1, 101)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            index, count = future.result()
            total += count
            print(f"[{completed}/100] file={index} mentions={count}", flush=True)
    print(f"Completed 100 files; base-Qwen mentions={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
