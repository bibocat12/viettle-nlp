#!/usr/bin/env python3
"""Per-entity grounded agentic repair for Qwen checkpoints that ignore batch schemas."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from build_submission import validate_submission
from grpo_v2_fusion import _strict_assertions
from kb import KnowledgeBase
from medical_web_search import MedicalWebSearch
from run_agentic_repair import REVIEW_ITEM_SCHEMA, apply_reviews, review_request
from tools import validate_output
from vllm_llm import VLLMLLM


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="bibocat/qwen3-ner-grpo-v2-merged")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--web-grounding", action="store_true", help="Add cached RxNav/NCBI MeSH evidence; never emits web-only codes")
    args = parser.parse_args()

    client = VLLMLLM(base_url=args.base_url, model=args.model, timeout=120, max_retries=2)
    if not client.is_available():
        raise RuntimeError(client.last_error)
    kb = KnowledgeBase(); kb.load()
    args.output.mkdir(parents=True, exist_ok=True)
    run_dir = args.output.parent; cache_dir = run_dir / "cache"; audit_path = run_dir / "audit.jsonl"
    web = MedicalWebSearch(run_dir / "web_cache") if args.web_grounding else None
    lock = threading.Lock()

    def process(record: int) -> tuple[int, int, int]:
        destination = args.output / f"{record}.json"
        if args.resume and destination.exists():
            values = json.loads(destination.read_text(encoding="utf-8"))
            return record, len(values), 0
        entities = json.loads((args.source / f"{record}.json").read_text(encoding="utf-8"))
        text = (args.inputs / f"{record}.txt").read_text(encoding="utf-8")
        record_changes = []
        reviewed = 0
        for index, original in enumerate(entities):
            system, user, allowed_zero = review_request(text, [original], kb)
            if not json.loads(user).get("mentions"):
                continue
            system = system.replace(
                "Chỉ trả một JSON array, mỗi input id tối đa một object.",
                "Chỉ trả một JSON object cho input id này.",
            )
            if web is not None:
                if original["type"] == "THUỐC":
                    evidence = web.drug(original["text"], allowed_zero.get(0, set()))
                elif original["type"] == "CHẨN_ĐOÁN":
                    evidence = web.diagnosis(original["text"])
                else:
                    evidence = []
                if evidence:
                    user += "\nPublic terminology evidence (not a code authority): " + json.dumps(evidence, ensure_ascii=False)
            reviewed += 1
            cache_path = cache_dir / str(record) / f"{index}.jsonl"
            response = None
            if args.resume and cache_path.exists():
                try:
                    response = json.loads(cache_path.read_text(encoding="utf-8").splitlines()[-1]).get("response")
                except (OSError, json.JSONDecodeError):
                    response = None
            if response is None:
                try:
                    response = client.generate_json(system, user, max_tokens=256, schema=REVIEW_ITEM_SCHEMA)
                except Exception as exc:
                    response = {}
                    record_changes.append({"action": "review_failed", "id": index, "error": str(exc)})
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({"record": record, "entity": index, "response": response}, ensure_ascii=False) + "\n", encoding="utf-8")
            # Agent assertion labels are advisory.  Require the deterministic
            # clause/section scope checker to independently support every
            # proposed positive label.  Do not clear an existing assertion
            # merely because the model returned an empty list.
            if isinstance(response, dict) and response.get("assertion_reviewed") is True:
                proposed = response.get("assertions", [])
                supported = set(_strict_assertions(text, original))
                current = set(original.get("assertions", []))
                if (
                    not isinstance(proposed, list)
                    or not proposed
                    or not set(proposed) <= supported
                    or (current and not set(proposed))
                ):
                    response = dict(response)
                    response["assertion_reviewed"] = False
            # The one-item request uses local id=0; remap audit back to the
            # immutable source entity index after applying it.
            repaired, changes = apply_reviews([original], response, allowed_zero, kb)
            entities[index] = repaired[0]
            for change in changes:
                change["id"] = index
                record_changes.append(change)
        errors = validate_output(entities)["errors"]
        if errors:
            raise RuntimeError(f"record {record}: {errors[:3]}")
        destination.write_text(json.dumps(entities, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with lock:
            with audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"record": record, "reviewed": reviewed, "changes": record_changes}, ensure_ascii=False) + "\n")
        return record, len(entities), len(record_changes)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(process, record) for record in range(1, 101)]
        for completed, future in enumerate(as_completed(futures), 1):
            record, count, changes = future.result()
            print(f"[{completed}/100] {record}: mentions={count} changes={changes}", flush=True)

    errors = validate_submission(args.output, args.inputs)
    if errors:
        raise RuntimeError("; ".join(errors[:10]))
    with zipfile.ZipFile(run_dir / "output.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for record in range(1, 101):
            archive.write(args.output / f"{record}.json", f"output/{record}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
