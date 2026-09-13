#!/usr/bin/env python3
"""Per-entity guarded repair with web search tool for entity confirmation."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests as http_requests


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from build_submission import validate_submission
from grpo_v2_fusion import _strict_assertions
from kb import KnowledgeBase
from run_agentic_repair import REVIEW_ITEM_SCHEMA, _run_review_loop, apply_reviews, review_request
from tools import validate_output
from vllm_llm import VLLMLLM


WIKI_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search Wikipedia for medical term information to confirm entity codes.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (medical term, disease name, drug name)",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def _wiki_opensearch(query: str, lang: str = "en") -> list[str]:
    try:
        resp = http_requests.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params={"action": "opensearch", "search": query, "limit": 2, "format": "json"},
            headers={"User-Agent": "ViettleNLP/1.0"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        titles = data[1] if len(data) > 1 else []
        descriptions = data[2] if len(data) > 2 else []
        urls = data[3] if len(data) > 3 else []
        results = []
        for title, desc, url in zip(titles, descriptions, urls):
            results.append(f"{title}: {desc}" if desc else title)
            results.append(f"  ({url})")
            if desc:
                page_resp = http_requests.get(
                    f"https://{lang}.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "prop": "extracts",
                        "exintro": True,
                        "explaintext": True,
                        "titles": title,
                        "format": "json",
                    },
                    headers={"User-Agent": "ViettleNLP/1.0"},
                    timeout=10,
                )
                page_resp.raise_for_status()
                pages = page_resp.json().get("query", {}).get("pages", {})
                for pid, pdata in pages.items():
                    if pid != "-1":
                        extract = (pdata.get("extract") or "")[:600]
                        if extract:
                            results.append(f"  Details: {extract}")
        return results
    except Exception:
        return []


def _web_search(query: str) -> str:
    results = _wiki_opensearch(query, "en") + _wiki_opensearch(query, "vi")
    seen = set()
    deduped = []
    for line in results:
        key = line.lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(line)
    return "\n".join(deduped[:10]) or "No results found."


def _run_guarded_review(
    client: VLLMLLM,
    system: str,
    user: str,
    max_tool_rounds: int = 3,
) -> object:
    user_data = json.loads(user) if isinstance(user, str) else user
    mentions = user_data.get("mentions", [])
    web_contexts = []
    for mention in mentions:
        text = mention.get("text", "")
        code = mention.get("current_candidate", "")
        choices = mention.get("candidate_choices", [])
        queries = [text, code]
        for choice in choices:
            if isinstance(choice, dict) and choice.get("name"):
                name = choice["name"][:80]
                queries.append(name)
        for query in queries:
            if not query:
                continue
            result = _web_search(query)
            if result and "No results" not in result and "Search error" not in result:
                web_contexts.append(f"Web info for '{text}':\n{result}")
                break
    web_section = (
        "\n\nThông tin tra cứu từ Wikipedia:\n" + "\n---\n".join(web_contexts)
    ) if web_contexts else ""
    enriched_system = system + web_section
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
                    response = _run_guarded_review(client, system, user)
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
