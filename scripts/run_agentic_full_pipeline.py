#!/usr/bin/env python3
"""Run a bounded, evidence-gated agentic review over a scored anchor.

Seven-pass extraction is treated only as proposal evidence.  The scored anchor
is never edited in place, and malformed/unfinished tool conversations are a
no-op.  Candidate changes require two independent reviewers; type repairs and
deletions require the same dual agreement plus deterministic training gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import unicodedata
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from build_submission import validate_submission
from kb import KnowledgeBase
from linker_agent import _search_icd10, _search_rxnorm
from vllm_llm import VLLMLLM


VALID_TYPES = {
    "CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "THUỐC",
    "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM",
}
PROMPT_VERSION = "agentic-full-v1"
ROLE_NAMES = ("terminology_specialist", "adversarial_conservator")
TYPE_CODE_OVERRIDES = {
    "nhịp tim chậm": "R00.1",
    "trầm cảm": "F32.9",
    "tiền sản giật": "O14.9",
}
DELETE_ALLOWLIST = {"viêm", "rượu"}


def normalise(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value)).casefold().split()).strip(" .,:;()")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_aliases() -> tuple[dict[str, str], dict[str, str]]:
    def merged(names: tuple[str, ...]) -> dict[str, str]:
        result: dict[str, str] = {}
        for name in names:
            path = ROOT / "database-stuff" / name
            if path.exists():
                result.update({normalise(k): str(v) for k, v in read_json(path).items()})
        return result
    return (
        merged(("icd10_aliases.json", "icd10_aliases_round2.json")),
        merged(("rxnorm_aliases.json", "rxnorm_aliases_round2.json")),
    )


def row_text(row: dict) -> str:
    for message in row.get("messages", []):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return str(row.get("text", ""))


def load_training_stats(path: Path) -> tuple[dict[str, Counter], dict[tuple[str, str], Counter]]:
    type_votes: defaultdict[str, Counter] = defaultdict(Counter)
    candidate_votes: defaultdict[tuple[str, str], Counter] = defaultdict(Counter)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for entity in row.get("entities", []):
                surface = normalise(entity.get("entity") or entity.get("text") or "")
                mention_type = str(entity.get("type", ""))
                if not surface or mention_type not in VALID_TYPES:
                    continue
                type_votes[surface][mention_type] += 1
                codes = list(map(str, entity.get("candidates", [])))
                if entity.get("_candidate_supervised") is not False and len(codes) == 1:
                    candidate_votes[(surface, mention_type)][codes[0]] += 1
    return dict(type_votes), dict(candidate_votes)


def bounded_count(text: str, surface: str) -> int:
    return len(re.findall(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", normalise(text)))


def low_precision_surfaces(path: Path, wanted: set[str], type_votes: dict[str, Counter]) -> dict[str, dict]:
    occurrences = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            text = row_text(json.loads(line))
            for surface in wanted:
                occurrences[surface] += bounded_count(text, surface)
    result = {}
    for surface in wanted:
        labels = sum(type_votes.get(surface, {}).values())
        total = occurrences[surface]
        result[surface] = {
            "labels": labels,
            "occurrences": total,
            "empirical_precision": labels / total if total else 0.0,
        }
    return result


def load_seven_support(path: Path | None) -> tuple[dict[tuple[int, int, int, str], list[str]], dict]:
    support: dict[tuple[int, int, int, str], list[str]] = {}
    pass_ids: set[str] = set()
    records: set[int] = set()
    if path is None:
        return support, {"records": 0, "passes": [], "strong_exact_proposals": 0}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            record = int(row["record"])
            records.add(record)
            for event in row.get("events", []):
                if event.get("pass_id"):
                    pass_ids.add(str(event["pass_id"]))
            for decision in row.get("decisions", []):
                position = decision.get("position")
                mention_type = decision.get("type")
                values = sorted(set(map(str, decision.get("support", []))))
                if isinstance(position, list) and len(position) == 2 and mention_type in VALID_TYPES:
                    key = (record, int(position[0]), int(position[1]), str(mention_type))
                    if len(values) > len(support.get(key, [])):
                        support[key] = values
    return support, {
        "records": len(records), "passes": sorted(pass_ids),
        "strong_exact_proposals": sum(len(values) >= 5 for values in support.values()),
    }


@dataclass(frozen=True)
class ReviewCase:
    case_id: str
    kind: str
    record: int
    position: tuple[int, int]
    text: str
    current_type: str
    proposed_type: str
    current_candidate: str
    proposed_candidate: str
    evidence: str
    training: dict
    context: str

    def payload(self) -> dict:
        return {
            "case_id": self.case_id,
            "kind": self.kind,
            "record": self.record,
            "position": list(self.position),
            "text": self.text,
            "current_type": self.current_type,
            "proposed_type": self.proposed_type,
            "current_candidate": self.current_candidate,
            "proposed_candidate": self.proposed_candidate,
            "evidence": self.evidence,
            "training": self.training,
            "context": self.context,
        }


def stable_case_id(kind: str, record: int, position: tuple[int, int], proposal: str) -> str:
    raw = json.dumps([kind, record, position, proposal], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_cases(
    anchor: dict[int, list[dict]],
    texts: dict[int, str],
    corrections: list[dict],
    max_tier: str,
    type_votes: dict[str, Counter],
    deletion_stats: dict[str, dict],
    seven_support: dict[tuple[int, int, int, str], list[str]],
) -> list[ReviewCase]:
    cases: list[ReviewCase] = []
    permitted = {"A"} if max_tier == "A" else {"A", "B"}
    correction_keys = set()
    for correction in corrections:
        if correction.get("tier") not in permitted:
            continue
        record = int(correction["record"])
        position = tuple(map(int, correction["position"]))
        entity = next((e for e in anchor[record] if tuple(e["position"]) == position), None)
        if entity is None or entity.get("type") not in {"CHẨN_ĐOÁN", "THUỐC"}:
            raise RuntimeError(f"invalid reviewed correction target: record={record} position={position}")
        proposed = str(correction.get("candidate", ""))
        current = str((entity.get("candidates") or [""])[0])
        start, end = position
        cases.append(ReviewCase(
            case_id=stable_case_id("candidate", record, position, proposed),
            kind="candidate", record=record, position=position, text=entity["text"],
            current_type=entity["type"], proposed_type=entity["type"],
            current_candidate=current, proposed_candidate=proposed,
            evidence=str(correction.get("evidence", "")), training={},
            context=texts[record][max(0, start - 120):min(len(texts[record]), end + 120)],
        ))
        correction_keys.add((record, position))

    for record, entities in anchor.items():
        for entity in entities:
            surface = normalise(entity["text"])
            votes = type_votes.get(surface)
            if votes:
                proposed_type, support = votes.most_common(1)[0]
                total = sum(votes.values())
                purity = support / total
                if proposed_type != entity["type"] and support >= 5 and purity >= 0.95:
                    start, end = map(int, entity["position"])
                    cases.append(ReviewCase(
                        case_id=stable_case_id("type", record, (start, end), proposed_type),
                        kind="type", record=record, position=(start, end), text=entity["text"],
                        current_type=entity["type"], proposed_type=proposed_type,
                        current_candidate=str((entity.get("candidates") or [""])[0]),
                        proposed_candidate="", evidence="exact training surface majority",
                        training={
                            "votes": dict(votes), "support": support, "total": total, "purity": purity,
                            "seven_pass_support": seven_support.get((record, start, end, proposed_type), []),
                        },
                        context=texts[record][max(0, start - 120):min(len(texts[record]), end + 120)],
                    ))
            stats = deletion_stats.get(surface)
            if (
                surface in DELETE_ALLOWLIST and stats and stats["labels"] >= 3
                and stats["empirical_precision"] <= 0.10
            ):
                start, end = map(int, entity["position"])
                cases.append(ReviewCase(
                    case_id=stable_case_id("delete", record, (start, end), "DELETE"),
                    kind="delete", record=record, position=(start, end), text=entity["text"],
                    current_type=entity["type"], proposed_type="", current_candidate="",
                    proposed_candidate="", evidence="deterministic exclusion + low empirical precision",
                    training={**stats, "seven_pass_support": seven_support.get((record, start, end, entity["type"]), [])},
                    context=texts[record][max(0, start - 120):min(len(texts[record]), end + 120)],
                ))
    unique = {case.case_id: case for case in cases}
    return sorted(unique.values(), key=lambda c: (c.record, c.position, c.kind))


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_case", "description": "Inspect the immutable case and its local clinical context.",
            "parameters": {"type": "object", "properties": {"case_id": {"type": "string"}}, "required": ["case_id"], "additionalProperties": False},
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_training_evidence", "description": "Get exact-surface label/candidate evidence from the training corpus.",
            "parameters": {"type": "object", "properties": {"case_id": {"type": "string"}}, "required": ["case_id"], "additionalProperties": False},
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_icd10", "description": "Search the local ICD-10 terminology for a diagnosis surface.",
            "parameters": {"type": "object", "properties": {"case_id": {"type": "string"}, "query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 8}}, "required": ["case_id", "query", "top_k"], "additionalProperties": False},
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_rxnorm", "description": "Search the local RxNorm terminology for a medication surface.",
            "parameters": {"type": "object", "properties": {"case_id": {"type": "string"}, "query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 8}}, "required": ["case_id", "query", "top_k"], "additionalProperties": False},
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_code", "description": "Look up the proposed code in local terminology and reviewed evidence.",
            "parameters": {"type": "object", "properties": {"case_id": {"type": "string"}, "code": {"type": "string"}}, "required": ["case_id", "code"], "additionalProperties": False},
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_terminology", "description": "Query a fixed authoritative public terminology endpoint using only the code, never the clinical note.",
            "parameters": {"type": "object", "properties": {"case_id": {"type": "string"}, "source": {"type": "string", "enum": ["rxnav", "icd10"]}, "code": {"type": "string"}}, "required": ["case_id", "source", "code"], "additionalProperties": False},
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_decision", "description": "Submit the terminal bounded decision after inspecting required evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "action": {"type": "string", "enum": ["KEEP", "UPDATE_CANDIDATE", "CHANGE_TYPE", "DELETE"]},
                    "candidate": {"type": "string"},
                    "entity_type": {"type": "string", "enum": sorted(VALID_TYPES)},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
                    "rationale": {"type": "string", "maxLength": 180},
                },
                "required": ["case_id", "action", "candidate", "entity_type", "confidence", "evidence_ids", "rationale"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]


class ToolExecutor:
    def __init__(self, kb: KnowledgeBase, type_votes: dict[str, Counter], candidate_votes: dict[tuple[str, str], Counter], aliases: tuple[dict[str, str], dict[str, str]], web_cache: Path):
        self.kb = kb
        self.type_votes = type_votes
        self.candidate_votes = candidate_votes
        self.icd_aliases, self.rx_aliases = aliases
        self.web_cache = web_cache
        self.lock = threading.Lock()

    def _guard(self, case: ReviewCase, args: dict) -> str | None:
        return None if args.get("case_id") == case.case_id else "case_id mismatch"

    def execute(self, name: str, args: dict, case: ReviewCase) -> dict:
        error = self._guard(case, args)
        if error:
            return {"error": error}
        if name == "inspect_case":
            return {"evidence_id": "case", **case.payload()}
        if name == "get_training_evidence":
            surface = normalise(case.text)
            return {
                "evidence_id": "training",
                "type_votes": dict(self.type_votes.get(surface, {})),
                "candidate_votes": dict(self.candidate_votes.get((surface, case.current_type), {})),
                "gate": case.training,
            }
        if name == "search_icd10":
            values = _search_icd10(str(args.get("query", "")), self.kb, top_k=int(args.get("top_k", 8)))
            return {"evidence_id": "icd_search", "results": values[:8]}
        if name == "search_rxnorm":
            values = _search_rxnorm(str(args.get("query", "")), self.kb, top_k=int(args.get("top_k", 8)))
            return {"evidence_id": "rx_search", "results": values[:8]}
        if name == "lookup_code":
            code = str(args.get("code", ""))
            if case.current_type == "THUỐC":
                info = self.kb.rxnorm_concepts.get(code, {})
                aliases = [surface for surface, value in self.rx_aliases.items() if value == code][:6]
            else:
                info = self.kb.icd10.get(code, {})
                aliases = [surface for surface, value in self.icd_aliases.items() if value == code][:6]
            return {"evidence_id": "code", "code": code, "local": info, "reviewed_aliases": aliases, "ledger_evidence": case.evidence}
        if name == "web_terminology":
            return self._web(case, str(args.get("source", "")), str(args.get("code", "")))
        return {"error": f"unknown tool {name}"}

    def _web(self, case: ReviewCase, source: str, code: str) -> dict:
        if code != case.proposed_candidate or not code:
            return {"error": "only the proposed non-empty code may be queried"}
        if source == "rxnav" and (case.current_type != "THUỐC" or not code.isdigit()):
            return {"error": "rxnav requires a medication RxCUI"}
        if source == "icd10" and case.current_type != "CHẨN_ĐOÁN":
            return {"error": "icd10 source requires a diagnosis"}
        cache = self.web_cache / f"{source}-{hashlib.sha256(code.encode()).hexdigest()[:20]}.json"
        try:
            data = read_json(cache)
        except (OSError, json.JSONDecodeError):
            try:
                if source == "rxnav":
                    response = requests.get(f"https://rxnav.nlm.nih.gov/REST/rxcui/{code}/properties.json", timeout=12, headers={"User-Agent": "ViettleNLP/1.0"})
                else:
                    response = requests.get("https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search", params={"sf": "code,name", "terms": code}, timeout=12, headers={"User-Agent": "ViettleNLP/1.0"})
                response.raise_for_status()
                data = response.json()
            except Exception as exc:
                data = {"error": str(exc)}
            with self.lock:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return {"evidence_id": "web", "source": source, "code": code, "result": data}


def assistant_message(message: dict) -> dict:
    result = {"role": "assistant", "content": message.get("content")}
    if message.get("tool_calls"):
        result["tool_calls"] = message["tool_calls"]
    return result


def expected_action(case: ReviewCase) -> str:
    return {"candidate": "UPDATE_CANDIDATE", "type": "CHANGE_TYPE", "delete": "DELETE"}[case.kind]


def decision_matches(case: ReviewCase, decision: dict) -> bool:
    if decision.get("case_id") != case.case_id or decision.get("action") != expected_action(case):
        return False
    if not isinstance(decision.get("confidence"), (int, float)):
        return False
    if case.kind == "candidate":
        return str(decision.get("candidate", "")) == case.proposed_candidate and decision.get("entity_type") == case.current_type
    if case.kind == "type":
        return decision.get("entity_type") == case.proposed_type
    return decision.get("entity_type") == case.current_type


def required_tools(case: ReviewCase) -> set[str]:
    if case.kind == "candidate":
        required = {"inspect_case", "lookup_code"}
        if case.proposed_candidate:
            required.add("web_terminology")
        return required
    return {"inspect_case", "get_training_evidence"}


def review_one(client: VLLMLLM, executor: ToolExecutor, case: ReviewCase, role: str, cache_dir: Path, resume: bool, max_rounds: int) -> dict:
    cache_key = hashlib.sha256(json.dumps({"version": PROMPT_VERSION, "model": client.model, "role": role, "case": case.payload()}, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    cache_path = cache_dir / case.kind / f"{cache_key}.json"
    if resume and cache_path.exists():
        try:
            return read_json(cache_path)
        except (OSError, json.JSONDecodeError):
            pass
    role_text = (
        "You are the primary medical terminology specialist." if role == "terminology_specialist"
        else "You are an adversarial conservative reviewer; reject any unsupported change."
    )
    required = sorted(required_tools(case))
    system = f"""{role_text}
Review exactly one immutable Vietnamese medical NER case. Use native tools, not
free-form answers. You MUST call {', '.join(required)} before submit_decision.
For candidate cases, UPDATE_CANDIDATE means the ledger proposal is correct;
KEEP means preserve the anchor. For type cases, CHANGE_TYPE only when context
and exact training evidence agree. DELETE only for a clear non-entity.
Reason internally. The terminal rationale must be short, verifiable evidence,
not chain-of-thought. Never invent a different code/type or alter boundaries."""
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps({"case_id": case.case_id, "kind": case.kind, "required_tools": required}, ensure_ascii=False)},
    ]
    called: set[str] = set()
    transcript: list[dict] = []
    terminal: dict = {}
    error = ""
    for _ in range(max_rounds):
        try:
            message = client.generate_with_tools(messages, tools=TOOLS, max_tokens=512)
        except Exception as exc:
            error = str(exc)
            break
        messages.append(assistant_message(message))
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            messages.append({"role": "user", "content": "Use the required native tools, then submit_decision."})
            continue
        for call in tool_calls:
            name = str(call.get("function", {}).get("name", ""))
            try:
                args = json.loads(call.get("function", {}).get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            if name == "submit_decision":
                terminal = args if isinstance(args, dict) else {}
                missing = required_tools(case) - called
                result = {"accepted_protocol": not missing, "missing_required_tools": sorted(missing)}
            else:
                result = executor.execute(name, args if isinstance(args, dict) else {}, case)
                if not result.get("error"):
                    called.add(name)
            transcript.append({"tool": name, "arguments": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": json.dumps(result, ensure_ascii=False)[:8000]})
        if terminal and required_tools(case) <= called:
            break
    value = {
        "case_id": case.case_id, "role": role, "called_tools": sorted(called),
        "decision": terminal, "protocol_ok": required_tools(case) <= called,
        "decision_matches": decision_matches(case, terminal), "error": error,
        "transcript": transcript,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(cache_path)
    return value


def package(output_dir: Path, inputs: Path, zip_path: Path) -> None:
    errors = validate_submission(output_dir, inputs)
    if errors:
        raise RuntimeError("; ".join(errors[:10]))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for record in range(1, 101):
            archive.write(output_dir / f"{record}.json", f"output/{record}.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training", type=Path, default=ROOT / "data_train/grpo_v2/extraction_train.jsonl")
    parser.add_argument("--seven-audit", type=Path, help="Existing deterministic seven-pass proposal audit to reuse")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="bibocat/qwen3-ner-grpo-v2-merged")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-tier", choices=("A", "B"), default="A")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-cases", type=int)
    args = parser.parse_args()

    client = VLLMLLM(base_url=args.base_url, model=args.model, timeout=120, max_retries=2)
    if not client.is_available():
        raise RuntimeError(client.last_error)
    kb = KnowledgeBase(); kb.load()
    aliases = load_aliases()
    type_votes, candidate_votes = load_training_stats(args.training)
    deletion_stats = low_precision_surfaces(args.training, DELETE_ALLOWLIST, type_votes)
    seven_support, seven_summary = load_seven_support(args.seven_audit)
    texts = {record: (args.inputs / f"{record}.txt").read_text(encoding="utf-8") for record in range(1, 101)}
    anchor = {record: read_json(args.anchor / f"{record}.json") for record in range(1, 101)}
    corrections = read_json(args.corrections)
    cases = build_cases(anchor, texts, corrections, args.max_tier, type_votes, deletion_stats, seven_support)
    if args.limit_cases is not None:
        cases = cases[:max(0, args.limit_cases)]

    run_dir = args.output.parent
    cache_dir = run_dir / "cache"
    executor = ToolExecutor(kb, type_votes, candidate_votes, aliases, run_dir / "web_cache")
    reviews: dict[str, list[dict]] = {}
    lock = threading.Lock()

    def process(case: ReviewCase) -> tuple[ReviewCase, list[dict]]:
        values = [review_one(client, executor, case, role, cache_dir, args.resume, max(3, args.max_rounds)) for role in ROLE_NAMES]
        return case, values

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(process, case) for case in cases]
        for completed, future in enumerate(as_completed(futures), 1):
            case, values = future.result()
            with lock:
                reviews[case.case_id] = values
            approvals = sum(
                value.get("protocol_ok") and value.get("decision_matches")
                and float(value.get("decision", {}).get("confidence", 0)) >= (0.98 if case.kind == "delete" else 0.95)
                for value in values
            )
            print(f"[{completed}/{len(cases)}] record={case.record} kind={case.kind} approvals={approvals}/2", flush=True)

    candidate_entities = {record: json.loads(json.dumps(values, ensure_ascii=False)) for record, values in anchor.items()}
    accepted: list[dict] = []
    rejected: list[dict] = []
    case_by_id = {case.case_id: case for case in cases}
    for case in cases:
        values = reviews.get(case.case_id, [])
        threshold = 0.98 if case.kind == "delete" else 0.95
        approved = len(values) == 2 and all(
            value.get("protocol_ok") and value.get("decision_matches")
            and float(value.get("decision", {}).get("confidence", 0)) >= threshold
            for value in values
        )
        event = {"case": case.payload(), "approved": approved, "reviews": values}
        (accepted if approved else rejected).append(event)
        if not approved or case.kind != "candidate":
            continue
        entity = next(e for e in candidate_entities[case.record] if tuple(e["position"]) == case.position and e["type"] == case.current_type)
        entity["candidates"] = [case.proposed_candidate] if case.proposed_candidate else []

    candidate_dir = run_dir / "candidate_output"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for record in range(1, 101):
        (candidate_dir / f"{record}.json").write_text(json.dumps(candidate_entities[record], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    package(candidate_dir, args.inputs, run_dir / "candidate_output.zip")

    final_entities = {record: json.loads(json.dumps(values, ensure_ascii=False)) for record, values in candidate_entities.items()}
    for event in accepted:
        case = case_by_id[event["case"]["case_id"]]
        if case.kind == "candidate":
            continue
        entities = final_entities[case.record]
        entity = next((e for e in entities if tuple(e["position"]) == case.position and e["type"] == case.current_type), None)
        if entity is None:
            continue
        if case.kind == "delete":
            entities.remove(entity)
            continue
        entity["type"] = case.proposed_type
        if case.proposed_type in {"CHẨN_ĐOÁN", "THUỐC"}:
            code = TYPE_CODE_OVERRIDES.get(normalise(entity["text"]), "")
            entity["candidates"] = [code] if code else []
        else:
            entity.pop("candidates", None)

    args.output.mkdir(parents=True, exist_ok=True)
    for record in range(1, 101):
        final_entities[record].sort(key=lambda e: (e["position"][0], e["position"][1], e["type"]))
        (args.output / f"{record}.json").write_text(json.dumps(final_entities[record], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    package(args.output, args.inputs, run_dir / "output.zip")

    audit_path = run_dir / "audit.jsonl"
    audit_path.write_text("".join(json.dumps(event, ensure_ascii=False) + "\n" for event in accepted + rejected), encoding="utf-8")
    summary = {
        "strategy": PROMPT_VERSION, "model": args.model, "workers": args.workers,
        "seven_pass_reused": seven_summary,
        "cases": len(cases), "accepted": len(accepted), "rejected": len(rejected),
        "accepted_by_kind": dict(Counter(event["case"]["kind"] for event in accepted)),
        "tool_calls": sum(len(review.get("transcript", [])) for event in accepted + rejected for review in event["reviews"]),
        "web_calls": sum(sum(step.get("tool") == "web_terminology" for step in review.get("transcript", [])) for event in accepted + rejected for review in event["reviews"]),
        "candidate_zip": str(run_dir / "candidate_output.zip"), "full_zip": str(run_dir / "output.zip"),
        "anchor_invariants": "candidate variant changes candidates only; full variant permits dual-approved type/delete transactions; assertions and positions are frozen",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
