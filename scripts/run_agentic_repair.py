#!/usr/bin/env python3
"""Grounded agentic candidate/assertion repair for an existing submission.

The reviewer is deliberately unable to add or delete spans, or alter type and
position.  It may only select a code supplied by local KB retrieval and revise
an assertion list for a supplied source span.  Every response is cached per
record so a stopped run resumes safely.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests as http_requests


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kb import KnowledgeBase
from linker_agent import _search_icd10, _search_rxnorm
from tools import validate_output
from vllm_llm import VLLMLLM


ASSERTIONS = {"isNegated", "isFamily", "isHistorical"}
REVIEW_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "candidate": {"type": "string"},
        "assertions": {"type": "array", "items": {"type": "string", "enum": sorted(ASSERTIONS)}},
        "assertion_reviewed": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "candidate_evidence": {"type": "string", "maxLength": 180},
        "assertion_evidence": {"type": "string", "maxLength": 180},
    },
    "required": ["id", "candidate", "assertions", "assertion_reviewed", "confidence"],
    "additionalProperties": False,
}
REVIEW_SCHEMA = {"type": "array", "items": REVIEW_ITEM_SCHEMA}

WEBFETCH_TOOL = {
    "type": "function",
    "function": {
        "name": "webfetch",
        "description": "Fetch content from a URL to verify or look up medical knowledge not in the local KB.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch",
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def _run_review_loop(
    client: VLLMLLM,
    system: str,
    user: str,
    max_tool_rounds: int = 3,
) -> object:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    for _ in range(max_tool_rounds):
        try:
            msg = client.generate_with_tools(messages, tools=[WEBFETCH_TOOL])
        except Exception:
            return client.generate_json(system, user, max_tokens=1024, schema=REVIEW_SCHEMA)
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc["function"]["name"] == "webfetch":
                    args = json.loads(tc["function"]["arguments"])
                    url = args.get("url", "")
                    try:
                        resp = http_requests.get(url, timeout=15)
                        result = resp.text[:6000]
                    except Exception as e:
                        result = f"Error fetching {url}: {e}"
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tc],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result[:4000],
                    })
            continue
        content = (msg.get("content") or "").strip()
        if content:
            return client.parse_json(content)
    return client.generate_json(system, user, max_tokens=1024, schema=REVIEW_SCHEMA)


def code_exists(code: str, mtype: str, kb: KnowledgeBase) -> bool:
    return code in kb.icd10 if mtype == "CHẨN_ĐOÁN" else code in kb.rxnorm_concepts


def candidates(entity: dict, kb: KnowledgeBase) -> list[dict]:
    mtype = entity["type"]
    if mtype not in {"CHẨN_ĐOÁN", "THUỐC"}:
        return []
    raw = _search_icd10(entity["text"], kb, top_k=8) if mtype == "CHẨN_ĐOÁN" else _search_rxnorm(entity["text"], kb, top_k=8)
    values = []
    existing = [str(code) for code in entity.get("candidates", [])[:1]]
    for code in existing + [str(value.get("code", "")) for value in raw]:
        if not code_exists(code, mtype, kb) or any(value["code"] == code for value in values):
            continue
        if mtype == "CHẨN_ĐOÁN":
            info = kb.icd10[code]
            name = info.get("vietnamese_name") or info.get("english_name", "")
        else:
            name = kb.rxnorm_concepts[code].get("prefLabel", "")
        values.append({"code": code, "name": name[:120]})
    return values[:8]


def review_request(text: str, entities: list[dict], kb: KnowledgeBase) -> tuple[str, str, dict[int, set[str]]]:
    items, allowed = [], {}
    for index, entity in enumerate(entities):
        start, end = entity["position"]
        choices = candidates(entity, kb)
        context = text[max(0, start - 140):min(len(text), end + 140)]
        assertion_relevant = bool(entity.get("assertions")) or bool(re.search(
            r"\b(?:không|chưa|phủ nhận|âm tính|tiền sử|gia đình|bố|mẹ|cha|"
            r"trước đây|đã từng|trước nhập viện)\b",
            context.casefold(),
        ))
        candidate_relevant = len(choices) >= 2 or (choices and not entity.get("candidates"))
        if not candidate_relevant and not assertion_relevant:
            continue
        items.append({
            "id": index,
            "text": entity["text"],
            "type": entity["type"],
            "current_candidate": (entity.get("candidates") or [""])[0],
            "candidate_choices": choices,
            "current_assertions": entity.get("assertions", []),
            "context": context,
            "review_candidate": candidate_relevant,
            "review_assertion": assertion_relevant,
        })
        allowed[index] = {choice["code"] for choice in choices}
    system = """Bạn là reviewer y khoa cực kỳ thận trọng.
Bạn KHÔNG được thêm/xóa thực thể hay thay đổi text, type, position.
Với mỗi id: chỉ chọn candidate từ candidate_choices; để candidate rỗng nếu
không chắc hoặc không có lựa chọn. Chỉ đặt assertion_reviewed=true khi ngữ cảnh
xác nhận rõ phủ định, tiền sử gia đình, hoặc tiền sử; nếu không chắc giữ nguyên
bằng assertion_reviewed=false. Hãy suy luận nội bộ, nhưng chỉ trả
candidate_evidence/assertion_evidence là bằng chứng ngắn, kiểm chứng được từ
tên mã hoặc đoạn context (tối đa 12 từ), không trả chuỗi suy luận từng bước.
Chỉ trả một JSON array, mỗi input id tối đa một object. confidence phản ánh độ
chắc chắn tổng thể; dưới 0.78 nếu còn mơ hồ. Không giải thích ngoài JSON."""
    return system, json.dumps({"mentions": items}, ensure_ascii=False), allowed


def apply_reviews(entities: list[dict], response: object, allowed: dict[int, set[str]], kb: KnowledgeBase) -> tuple[list[dict], list[dict]]:
    reviews = response if isinstance(response, list) else response.get("reviews", []) if isinstance(response, dict) else []
    # Some Qwen/vLLM structured-output versions return one valid review item
    # directly instead of wrapping it in the requested ``reviews`` object.
    # Accept that compatible form rather than silently discarding the review.
    if isinstance(response, dict) and "id" in response:
        reviews = [response]
    audit = []
    for review in reviews if isinstance(reviews, list) else []:
        if not isinstance(review, dict) or not isinstance(review.get("id"), int):
            continue
        index = review["id"]
        if not 0 <= index < len(entities):
            continue
        entity = entities[index]
        confidence = float(review.get("confidence", 0.0)) if isinstance(review.get("confidence", 0.0), (int, float)) else 0.0
        if confidence < 0.78:
            continue
        code = str(review.get("candidate", ""))
        candidate_evidence = str(review.get("candidate_evidence", "")).strip()
        if code and candidate_evidence and code in allowed.get(index, set()) and code_exists(code, entity["type"], kb):
            if entity.get("candidates") != [code]:
                audit.append({
                    "action": "candidate_repaired", "id": index,
                    "from": entity.get("candidates", []), "to": [code],
                    "evidence": str(review.get("candidate_evidence", ""))[:180],
                })
                entity["candidates"] = [code]
        labels = review.get("assertions")
        if review.get("assertion_reviewed") is True and isinstance(labels, list) and all(label in ASSERTIONS for label in labels):
            labels = list(dict.fromkeys(labels))
            if entity.get("assertions", []) != labels:
                audit.append({
                    "action": "assertion_repaired", "id": index,
                    "from": entity.get("assertions", []), "to": labels,
                    "evidence": str(review.get("assertion_evidence", ""))[:180],
                })
                entity["assertions"] = labels
    return entities, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Validated source output directory")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="bibocat/qwen3-ner-grpo-v2-merged")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    client = VLLMLLM(base_url=args.base_url, model=args.model, timeout=120, max_retries=2)
    if not client.is_available():
        raise RuntimeError(client.last_error)
    kb = KnowledgeBase(); kb.load()
    args.output.mkdir(parents=True, exist_ok=True)
    run_dir = args.output.parent; cache = run_dir / "cache"; audit_path = run_dir / "audit.jsonl"
    lock = threading.Lock()

    def process(index: int) -> tuple[int, int]:
        destination = args.output / f"{index}.json"
        if args.resume and destination.exists():
            return index, len(json.loads(destination.read_text(encoding="utf-8")))
        entities = json.loads((args.source / f"{index}.json").read_text(encoding="utf-8"))
        text = (args.inputs / f"{index}.txt").read_text(encoding="utf-8")
        cache_path = cache / f"{index}.jsonl"
        if args.resume and cache_path.exists():
            response = json.loads(cache_path.read_text(encoding="utf-8").splitlines()[-1]).get("response", {})
            _, _, allowed = review_request(text, entities, kb)
        else:
            system, user, allowed = review_request(text, entities, kb)
            try:
                response = _run_review_loop(client, system, user)
            except Exception as exc:
                response = {"reviews": []}
                with lock:
                    audit_path.parent.mkdir(parents=True, exist_ok=True)
                    audit_path.open("a", encoding="utf-8").write(json.dumps({"record": index, "error": str(exc)}) + "\n")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"record": index, "response": response}, ensure_ascii=False) + "\n", encoding="utf-8")
        result, changes = apply_reviews(entities, response, allowed, kb)
        errors = validate_output(result)["errors"]
        if errors:
            raise RuntimeError(f"record {index}: {errors[:3]}")
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with lock:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.open("a", encoding="utf-8").write(json.dumps({"record": index, "changes": changes}, ensure_ascii=False) + "\n")
        return index, len(result)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(process, index) for index in range(1, 101)]
        for completed, future in enumerate(as_completed(futures), 1):
            index, count = future.result(); print(f"[{completed}/100] {index}: {count}", flush=True)

    from scripts.build_submission import validate_submission
    errors = validate_submission(args.output, args.inputs)
    if errors:
        raise RuntimeError("; ".join(errors[:10]))
    import zipfile
    with zipfile.ZipFile(run_dir / "output.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(1, 101): archive.write(args.output / f"{index}.json", f"output/{index}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
