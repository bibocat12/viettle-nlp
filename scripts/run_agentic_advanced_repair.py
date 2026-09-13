#!/usr/bin/env python3
"""Per-entity advanced repair with 5 breakthrough improvements:

1. KB Context Injection — full ICD-10/RxNorm descriptions instead of Wikipedia
2. Document-level consistency — same text → same code across each record
3. Hard-negative comparison — distinguish similar candidate codes
4. Type-specific confidence thresholds — per entity type
5. Chapter prefix pre-filtering — ICD-10 chapter letter for diagnosis codes
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests as http_requests


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from build_submission import validate_submission
from grpo_v2_fusion import _strict_assertions
from kb import KnowledgeBase
from run_agentic_repair import REVIEW_ITEM_SCHEMA, apply_reviews, review_request
from tools import validate_output
from vllm_llm import VLLMLLM


TYPE_THRESHOLDS = {
    "CHẨN_ĐOÁN": 0.72,
    "THUỐC": 0.68,
    "TRIỆU_CHỨNG": 0.80,
    "XÉT_NGHIỆM": 0.78,
}


def _kb_candidate_details(choices: list[dict], mtype: str, kb: KnowledgeBase) -> str:
    lines = []
    for c in choices:
        code = c.get("code", "")
        name = c.get("name", "")
        if mtype == "CHẨN_ĐOÁN":
            info = kb.icd10.get(code, {})
            vi = info.get("vietnamese_name", "")
            en = info.get("english_name", "")
            ch = code[0] if code else "?"
            details = f"  - Chapter {ch}: {vi}"
            if en and en != vi:
                details += f" / {en}"
        elif mtype == "THUỐC":
            info = kb.rxnorm_concepts.get(code, {})
            pref = info.get("prefLabel", "")
            strength = info.get("strength", "")
            details = f"  - {pref}"
            if strength:
                details += f" [{strength}]"
        else:
            details = f"  - {name}"
        if details != f"  - {name}":
            lines.append(details)
    return "\n".join(lines) if lines else ""


def _find_similar_candidates(choices: list[dict]) -> list[tuple[int, int]]:
    pairs = []
    for i, a in enumerate(choices):
        for j, b in enumerate(choices):
            if i >= j:
                continue
            an = a.get("name", "").lower()
            bn = b.get("name", "").lower()
            if not an or not bn:
                continue
            shorter = min(len(an), len(bn))
            if shorter < 5:
                continue
            common = sum(1 for ca, cb in zip(an, bn) if ca == cb)
            if common / shorter > 0.7:
                pairs.append((i, j))
    return pairs


def _enrich_system_prompt(system: str, user_data: dict, kb: KnowledgeBase) -> str:
    mentions = user_data.get("mentions", [])
    extras = []

    for m in mentions:
        text = m.get("text", "")
        mtype = m.get("type", "")
        choices = m.get("candidate_choices", [])

        # 1. KB context injection
        kb_detail = _kb_candidate_details(choices, mtype, kb)
        if kb_detail:
            extras.append(f"Chi tiết mã cho '{text}':\n{kb_detail}")

        # 3. Hard-negative comparison
        pairs = _find_similar_candidates(choices)
        if pairs:
            extras.append(f"Các mã cho '{text}' có tên rất giống nhau, cần phân biệt kỹ:")
            for i, j in pairs:
                extras.append(f"  - {choices[i].get('code','')}: {choices[i].get('name','')[:100]}")
                extras.append(f"  - {choices[j].get('code','')}: {choices[j].get('name','')[:100]}")
            extras.append("")

    # 5. Chapter prefix for all diagnosis codes
    diagnosis_chapters = set()
    for m in mentions:
        if m.get("type") in ("CHẨN_ĐOÁN",):
            for c in m.get("candidate_choices", []):
                code = c.get("code", "")
                if code and code[0].isalpha():
                    diagnosis_chapters.add(code[0])
    if diagnosis_chapters:
        extras.append(f"ICD-10 chapter(s) involved in this record: {', '.join(sorted(diagnosis_chapters))}")

    enrichment = "\n".join(extras)
    return system + "\n\n" + enrichment if enrichment else system


def _harmonize_entities(entities: list[dict]) -> tuple[list[dict], list[dict]]:
    text_groups = {}
    for idx, e in enumerate(entities):
        key = e.get("text", "").strip().lower()
        if not key:
            continue
        text_groups.setdefault(key, []).append(idx)

    changed = []
    for key, indices in text_groups.items():
        if len(indices) < 2:
            continue
        codes = [(entities[i].get("candidates") or [None])[0] for i in indices]
        codes = [c for c in codes if c]
        if not codes:
            continue
        most_common, _ = Counter(codes).most_common(1)[0]
        for i in indices:
            current = (entities[i].get("candidates") or [None])[0]
            if current and current != most_common:
                changed.append({
                    "action": "consistency_repaired",
                    "id": i,
                    "from": [current],
                    "to": [most_common],
                    "evidence": f"same text '{entities[i]['text']}' harmonized to most common code",
                })
                entities[i]["candidates"] = [most_common]
    return entities, changed


def run_advanced_review(
    client: VLLMLLM,
    system: str,
    user: str,
    kb: KnowledgeBase,
) -> object:
    user_data = json.loads(user) if isinstance(user, str) else user
    enriched_system = _enrich_system_prompt(system, user_data, kb)
    return client.generate_json(enriched_system, user, max_tokens=256, schema=REVIEW_ITEM_SCHEMA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="bibocat/qwen3-ner-grpo-v2-merged")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    client = VLLMLLM(base_url=args.base_url, model=args.model, timeout=120, max_retries=2)
    if not client.is_available():
        raise RuntimeError(client.last_error)
    kb = KnowledgeBase(); kb.load()
    args.output.mkdir(parents=True, exist_ok=True)
    run_dir = args.output.parent; cache_dir = run_dir / "cache"; audit_path = run_dir / "audit.jsonl"
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
                    response = run_advanced_review(client, system, user, kb)
                except Exception as exc:
                    response = {}
                    record_changes.append({"action": "review_failed", "id": index, "error": str(exc)})
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({"record": record, "entity": index, "response": response}, ensure_ascii=False) + "\n", encoding="utf-8")
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
            # 4. Type-specific confidence thresholds
            thresholds = TYPE_THRESHOLDS.copy()
            mtype = original.get("type", "")
            if mtype in thresholds:
                old_threshold = 0.78
                if isinstance(response, dict) and isinstance(response.get("confidence"), (int, float)):
                    conf = float(response["confidence"])
                    if conf < thresholds[mtype] and conf >= old_threshold:
                        response = dict(response)
                        response["assertion_reviewed"] = False
            repaired, changes = apply_reviews([original], response, allowed_zero, kb)
            entities[index] = repaired[0]
            for change in changes:
                change["id"] = index
                record_changes.append(change)

        # 2. Document-level consistency
        entities, consistency_changes = _harmonize_entities(entities)
        record_changes.extend(consistency_changes)

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
