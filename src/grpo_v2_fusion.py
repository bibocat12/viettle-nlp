"""Safe three-pass Qwen3 GRPO-v2 extraction fused with an anchored submission.

The model is intentionally used only for ``entity``/``type`` proposals.  All
offsets, assertions and codes are produced or checked locally so a malformed
model field can never leak into the competition submission.
"""

from __future__ import annotations

import json
import re
import threading
import unicodedata
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from linker_agent import (
    _embedding_rerank,
    _reviewed_icd10_code,
    _reviewed_rxnorm_code,
    _search_icd10,
    _search_rxnorm,
)
from ner_tool import TYPE_MAP, VALID_TYPES, _extend_drug_order, _is_section_header, _rule_find_positions, _rule_verify_type
from tools import rule_extract
from vllm_llm import VLLMLLM


PASS_SPECS = {
    "general": {"chunk_size": 1200, "overlap": 160, "types": tuple(VALID_TYPES), "mode": "general"},
    "clinical": {
        "chunk_size": 700,
        "overlap": 120,
        "types": ("CHẨN_ĐOÁN", "TRIỆU_CHỨNG"),
        "mode": "clinical",
    },
    "medlab": {
        "chunk_size": 700,
        "overlap": 120,
        "types": ("THUỐC", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"),
        "mode": "medlab",
    },
    # Seven-pass ablation: the three primary passes plus independent scales
    # that change the context seen by the model without trusting its offsets.
    "general_900": {"chunk_size": 900, "overlap": 140, "types": tuple(VALID_TYPES), "mode": "general"},
    "clinical_500": {
        "chunk_size": 500, "overlap": 100,
        "types": ("CHẨN_ĐOÁN", "TRIỆU_CHỨNG"), "mode": "clinical",
    },
    "medlab_500": {
        "chunk_size": 500, "overlap": 100,
        "types": ("THUỐC", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"), "mode": "medlab",
    },
    "general_1600": {"chunk_size": 1600, "overlap": 200, "types": tuple(VALID_TYPES), "mode": "general"},
}
ASSERTION_LABELS = {"isNegated", "isFamily", "isHistorical"}
PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "type": {"type": "string", "enum": sorted(VALID_TYPES)},
                },
                "required": ["entity", "type"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["entities"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class FusionSettings:
    min_icd_score: float = 0.63
    min_rx_score: float = 0.75
    min_margin: float = 0.05


def _normalise(value: str) -> str:
    return unicodedata.normalize("NFC", value).replace("\u00a0", " ")


def _alias_key(value: str) -> str:
    return re.sub(r"\s+", " ", _normalise(value).casefold()).strip(" .,:;()")


def _chunk_text(text: str, size: int, overlap: int) -> list[tuple[int, str]]:
    """Split with stable original offsets, preferring natural boundaries."""
    if not text.strip():
        return []
    if len(text) <= size:
        return [(0, text)]
    chunks: list[tuple[int, str]] = []
    start = 0
    while start < len(text):
        hard_end = min(len(text), start + size)
        end = hard_end
        if hard_end < len(text):
            floor = start + size // 2
            end = max(text.rfind(token, floor, hard_end) for token in ("\n", ". ", "; ", ": "))
            if end <= start:
                end = hard_end
            elif text[end] == "\n":
                end += 1
            else:
                end += 2
        if text[start:end].strip():
            chunks.append((start, text[start:end]))
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _pass_prompt(pass_id: str, allowed_types: tuple[str, ...], chunk: str) -> tuple[str, str]:
    descriptions = {
        "general": "Trích xuất tất cả thực thể y khoa thuộc các loại được phép.",
        "clinical": "Chỉ trích xuất chẩn đoán và triệu chứng. Mỗi span phải nguyên tử, không gộp các triệu chứng liền nhau.",
        "medlab": "Chỉ trích xuất thuốc, tên xét nghiệm, kết quả xét nghiệm. Tách tên xét nghiệm khỏi kết quả; thuốc giữ đủ hàm lượng, đường dùng, tần suất.",
    }
    mode = PASS_SPECS[pass_id]["mode"]
    system = (
        "Bạn là bộ đề xuất NER y khoa tiếng Việt. "
        f"{descriptions[mode]} "
        "Chỉ trả về JSON theo schema. Mỗi mục CHỈ có entity nguyên văn và type. "
        "Không trả position, assertions, candidates hay giải thích. "
        f"Các type được phép: {', '.join(allowed_types)}."
    )
    return system, f"Văn bản cần trích xuất:\n{chunk}"


def _cache_path(cache_dir: Path, record_id: str, pass_id: str, chunk_index: int) -> Path:
    return cache_dir / record_id / f"{pass_id}-{chunk_index}.jsonl"


def _read_cache(path: Path) -> dict[str, Any] | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return json.loads(lines[-1]) if lines else None
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _align_proposals(items: object, chunk: str, chunk_start: int, pass_id: str) -> list[dict]:
    """Discard all model offsets and align each NFC proposal by exact text."""
    proposals: list[dict] = []
    if isinstance(items, dict):
        items = items.get("entities", [])
    if not isinstance(items, list):
        return proposals
    normalised_chunk = _normalise(chunk)
    # Input notes are NFC in the target set.  Refuse an unsafe match if NFC
    # would change its length, rather than emitting an incorrect original offset.
    if len(normalised_chunk) != len(chunk):
        normalised_chunk = chunk
    for item in items:
        if not isinstance(item, dict):
            continue
        proposed = _normalise(str(item.get("entity") or item.get("text") or "")).strip()
        mtype = TYPE_MAP.get(str(item.get("type", "")).upper().strip(), str(item.get("type", "")).upper().strip())
        if not proposed or mtype not in VALID_TYPES or _is_section_header(proposed):
            continue
        rule_type = _rule_verify_type(proposed, mtype)
        if rule_type is None or rule_type not in VALID_TYPES:
            continue
        # Exact text search returns every genuine occurrence, including repeats.
        for position in _rule_find_positions(proposed, normalised_chunk, chunk_start):
            if rule_type == "THUỐC":
                position = _extend_drug_order(position, normalised_chunk, chunk_start)
            start, end = position
            proposals.append({
                "text": chunk[start - chunk_start:end - chunk_start],
                "position": [start, end],
                "type": rule_type,
                "pass_id": pass_id,
            })
    return proposals


def extract_pass(
    record_id: str,
    text: str,
    pass_id: str,
    client: VLLMLLM,
    cache_dir: Path,
    resume: bool,
) -> tuple[list[dict], list[dict]]:
    spec = PASS_SPECS[pass_id]
    proposals: list[dict] = []
    events: list[dict] = []
    for chunk_index, (start, chunk) in enumerate(_chunk_text(text, spec["chunk_size"], spec["overlap"])):
        cached = _read_cache(_cache_path(cache_dir, record_id, pass_id, chunk_index)) if resume else None
        if cached is not None:
            raw_items = cached.get("response", {})
            status = cached.get("status", "cached")
        else:
            system, user = _pass_prompt(pass_id, spec["types"], chunk)
            try:
                raw_items = client.generate_json(system, user, max_tokens=1024, schema=PROPOSAL_SCHEMA)
                status = "ok"
            except Exception as exc:  # An individual chunk must not poison its record.
                raw_items = {"entities": []}
                status = "failed"
                events.append({"event": "chunk_failed", "pass_id": pass_id, "chunk": chunk_index, "error": str(exc)})
            _write_cache(_cache_path(cache_dir, record_id, pass_id, chunk_index), {
                "record": record_id,
                "pass_id": pass_id,
                "chunk": chunk_index,
                "start": start,
                "status": status,
                "response": raw_items,
            })
        aligned = _align_proposals(raw_items, chunk, start, pass_id)
        proposals.extend(aligned)
        events.append({"event": "chunk", "pass_id": pass_id, "chunk": chunk_index, "status": status, "aligned": len(aligned)})
    unique = {(item["position"][0], item["position"][1], item["type"], item["pass_id"]): item for item in proposals}
    return list(unique.values()), events


def _strict_assertions(text: str, entity: dict) -> list[str]:
    """Clause/section-scoped assertions; cues never cross punctuation or headers."""
    start = entity["position"][0]
    prefix = _normalise(text[max(0, start - 220):start]).casefold()
    boundary = max(prefix.rfind(marker) for marker in ("\n", ".", ";", "!", "?", "•"))
    prefix = prefix[boundary + 1:]
    contrast = list(re.finditer(r"\b(?:nhưng|tuy nhiên|song|mặc dù)\b", prefix))
    if contrast:
        prefix = prefix[contrast[-1].end():]
    labels: list[str] = []
    if re.search(r"\b(?:không(?:\s+có|\s+ghi nhận|\s+còn)?|chưa(?:\s+có)?|phủ nhận|âm tính(?:\s+với)?)\b.{0,90}$", prefix, re.DOTALL):
        labels.append("isNegated")
    if re.search(r"\btiền sử gia đình\b.{0,140}$", prefix, re.DOTALL) or re.search(
        r"\b(?:bố|mẹ|cha|anh trai|chị gái|ông|bà|con)\b.{0,55}\b(?:bị|mắc|có|được chẩn đoán)\b.{0,55}$", prefix, re.DOTALL
    ):
        labels.append("isFamily")
    before = _normalise(text[max(0, start - 1800):start]).casefold()
    history_headers = ("tiền sử bệnh nội khoa", "tiền sử bệnh\n", "tiền sử:", "các bệnh mãn tính", "thuốc trước nhập viện", "thuốc trước khi nhập viện")
    current_headers = ("tiền sử bệnh hiện tại", "bệnh sử hiện tại", "lý do nhập viện", "triệu chứng hiện tại", "đánh giá tại bệnh viện", "tình trạng ngay trước khi nhập viện")
    historical_section = max((before.rfind(value) for value in history_headers), default=-1) > max((before.rfind(value) for value in current_headers), default=-1)
    explicit_history = re.search(r"\b(?:tiền sử|trước đây|đã từng|đã dùng|trước nhập viện|trước khi nhập viện)\b.{0,110}$", prefix, re.DOTALL)
    indication = entity["type"] == "TRIỆU_CHỨNG" and re.search(r"\b(?:điều trị|do|vì)\b.{0,45}$", prefix)
    if not indication and (historical_section or explicit_history):
        labels.append("isHistorical")
    return labels


def _valid_code(code: str, mtype: str, kb) -> bool:
    return bool(kb.icd10.get(code)) if mtype == "CHẨN_ĐOÁN" else bool(kb.rxnorm_concepts.get(code))


def _candidate_for(entity: dict, kb, settings: FusionSettings) -> tuple[list[str], str]:
    mtype, text = entity["type"], entity["text"]
    if mtype not in {"CHẨN_ĐOÁN", "THUỐC"}:
        return [], "not_linkable"
    reviewed = _reviewed_icd10_code(text) if mtype == "CHẨN_ĐOÁN" else _reviewed_rxnorm_code(text)
    if reviewed and _valid_code(reviewed, mtype, kb):
        return [reviewed], "reviewed_alias"
    candidates = _search_icd10(text, kb, top_k=15) if mtype == "CHẨN_ĐOÁN" else _search_rxnorm(text, kb, top_k=15)
    candidates = [candidate for candidate in candidates if _valid_code(str(candidate.get("code", "")), mtype, kb)]
    if not candidates:
        return [], "no_kb_match"
    # The deterministic retrieval score still controls thresholds/margins.  An
    # embedding rerank can only reorder its retrieved candidates; it cannot
    # invent a code or bypass KB validation.
    try:
        semantic = _embedding_rerank(text, mtype, candidates, kb)
    except Exception:
        semantic = []
    if semantic:
        original = {str(candidate["code"]): candidate for candidate in candidates}
        reranked = [original[str(candidate["code"])] for candidate in semantic if str(candidate.get("code")) in original]
        seen = {str(candidate["code"]) for candidate in reranked}
        candidates = reranked + [candidate for candidate in candidates if str(candidate["code"]) not in seen]
    first = candidates[0]
    score = float(first.get("score", 0.0))
    margin = score - float(candidates[1].get("score", 0.0)) if len(candidates) > 1 else score
    threshold = settings.min_icd_score if mtype == "CHẨN_ĐOÁN" else settings.min_rx_score
    if score >= threshold and margin >= settings.min_margin:
        return [str(first["code"])], "fuzzy_kb"
    return [], "ambiguous_kb"


def _exact_rule_or_alias(entity: dict, rule_mentions: set[tuple[int, int, str]], kb, settings: FusionSettings) -> bool:
    start, end = entity["position"]
    if (start, end, entity["type"]) in rule_mentions:
        return True
    code, source = _candidate_for(entity, kb, settings)
    return bool(code) and source == "reviewed_alias"


def _baseline_from_zip(anchor_zip: Path, record_id: str) -> list[dict]:
    with zipfile.ZipFile(anchor_zip) as archive:
        try:
            value = json.loads(archive.read(f"output/{record_id}.json").decode("utf-8"))
        except KeyError as exc:
            raise RuntimeError(f"anchor ZIP missing output/{record_id}.json") from exc
    return value if isinstance(value, list) else []


def _clean_baseline(text: str, baseline: list[dict], kb, settings: FusionSettings) -> list[dict]:
    cleaned = []
    for item in baseline:
        if not isinstance(item, dict) or item.get("type") not in VALID_TYPES:
            continue
        position = item.get("position")
        if not (isinstance(position, list) and len(position) == 2 and all(isinstance(value, int) for value in position)):
            continue
        start, end = position
        if not (0 <= start < end <= len(text)):
            continue
        entity = {"text": text[start:end], "type": item["type"], "assertions": [label for label in item.get("assertions", []) if label in ASSERTION_LABELS], "position": [start, end]}
        if entity["type"] in {"CHẨN_ĐOÁN", "THUỐC"}:
            existing = [str(code) for code in item.get("candidates", [])[:1] if _valid_code(str(code), entity["type"], kb)]
            reviewed = _reviewed_icd10_code(entity["text"]) if entity["type"] == "CHẨN_ĐOÁN" else _reviewed_rxnorm_code(entity["text"])
            entity["candidates"] = [reviewed] if reviewed and _valid_code(reviewed, entity["type"], kb) else existing
        cleaned.append(entity)
    return cleaned


def _fuse(text: str, baseline: list[dict], proposals: list[dict], kb, settings: FusionSettings) -> tuple[list[dict], list[dict]]:
    audit: list[dict] = []
    baseline = _clean_baseline(text, baseline, kb, settings)
    rule_mentions = {
        (item["position"][0], item["position"][1], item["type"])
        for item in rule_extract(text).get("mentions", [])
        if item.get("type") in VALID_TYPES and isinstance(item.get("position"), list)
    }
    by_span: dict[tuple[int, int], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for item in proposals:
        by_span[tuple(item["position"])][item["type"]].add(item["pass_id"])

    # Baseline boundaries are immutable.  Types change only with general + the
    # relevant specialist pass + a deterministic rule agreeing exactly.
    specialist = {
        "CHẨN_ĐOÁN": {"clinical", "clinical_500"},
        "TRIỆU_CHỨNG": {"clinical", "clinical_500"},
        "THUỐC": {"medlab", "medlab_500"},
        "TÊN_XÉT_NGHIỆM": {"medlab", "medlab_500"},
        "KẾT_QUẢ_XÉT_NGHIỆM": {"medlab", "medlab_500"},
    }
    general_passes = {"general", "general_900", "general_1600"}
    for entity in baseline:
        span = tuple(entity["position"])
        for proposed_type, support in by_span.get(span, {}).items():
            deterministic = _rule_verify_type(entity["text"], "")
            if (
                proposed_type != entity["type"]
                and deterministic == proposed_type
                and support & general_passes
                and support & specialist[proposed_type]
            ):
                audit.append({"action": "baseline_type_repaired", "entity": entity["text"], "position": entity["position"], "from": entity["type"], "to": proposed_type, "support": sorted(support)})
                entity["type"] = proposed_type
                codes, _ = _candidate_for(entity, kb, settings)
                if proposed_type in {"CHẨN_ĐOÁN", "THUỐC"}:
                    entity["candidates"] = codes
                else:
                    entity.pop("candidates", None)
                break
        deterministic_labels = _strict_assertions(text, entity)
        # A non-empty scoped label is deterministic confirmation; otherwise
        # preserve the anchor's judgement rather than making speculative clears.
        if deterministic_labels and deterministic_labels != entity["assertions"]:
            audit.append({"action": "baseline_assertion_repaired", "entity": entity["text"], "position": entity["position"], "from": entity["assertions"], "to": deterministic_labels})
            entity["assertions"] = deterministic_labels

    occupied = [tuple(item["position"]) for item in baseline]
    candidates: list[dict] = []
    for (start, end), typed_support in by_span.items():
        if any(min(end, old_end) > max(start, old_start) for old_start, old_end in occupied):
            for mtype, support in typed_support.items():
                audit.append({"action": "rejected", "reason": "overlaps_baseline", "position": [start, end], "type": mtype, "support": sorted(support)})
            continue
        for mtype, support in typed_support.items():
            consensus = len(support)
            entity = {"text": text[start:end], "type": mtype, "position": [start, end], "assertions": []}
            accepted_by = ""
            if consensus >= 2:
                accepted_by = f"consensus_{consensus}"
            elif _exact_rule_or_alias(entity, rule_mentions, kb, settings):
                accepted_by = "model_rule_or_reviewed_alias"
            if not accepted_by:
                audit.append({"action": "rejected", "reason": "insufficient_support", "entity": entity["text"], "position": entity["position"], "type": mtype, "support": sorted(support)})
                continue
            codes, code_source = _candidate_for(entity, kb, settings)
            if mtype in {"CHẨN_ĐOÁN", "THUỐC"} and not codes:
                audit.append({"action": "rejected", "reason": "no_valid_kb_code", "entity": entity["text"], "position": entity["position"], "type": mtype, "support": sorted(support)})
                continue
            if codes:
                entity["candidates"] = codes
            entity["assertions"] = _strict_assertions(text, entity)
            entity["support"] = sorted(support)
            entity["source"] = accepted_by
            entity["_rank"] = consensus
            candidates.append(entity)
            audit.append({"action": "accepted", "reason": accepted_by, "entity": entity["text"], "position": entity["position"], "type": mtype, "support": sorted(support), "candidate_source": code_source})

    # Resolve collisions amongst additions.  Clinical spans favour atomic
    # mentions; medication spans favour the complete order.
    def priority(item: dict) -> tuple:
        start, end = item["position"]
        length = end - start
        length_score = length if item["type"] == "THUỐC" else -length
        return (-item["_rank"], -int(item["source"].startswith("model_rule")), -length_score, start, end)

    kept = []
    for item in sorted(candidates, key=priority):
        start, end = item["position"]
        if any(min(end, old_end) > max(start, old_start) for old_start, old_end in occupied):
            audit.append({"action": "rejected", "reason": "proposal_conflict", "entity": item["text"], "position": item["position"], "type": item["type"], "support": item["support"]})
            continue
        occupied.append((start, end))
        kept.append(item)
    for item in kept:
        for key in ("source", "support", "_rank"):
            item.pop(key, None)
    final = baseline + kept
    final.sort(key=lambda item: (item["position"][0], item["position"][1], item["type"]))
    return final, audit


def _load_validation(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            messages = row.get("messages", [])
            prompt = next((message.get("content", "") for message in messages if message.get("role") == "user"), "")
            if prompt and isinstance(row.get("entities"), list):
                rows.append({"text": prompt, "entities": row["entities"]})
        except json.JSONDecodeError:
            continue
    return rows


def score_predictions(predictions: list[dict], truth: list[dict]) -> dict[str, float]:
    """Competition-equivalent score components, shared with the GRPO reward."""
    from scripts.reward_v2 import score_entities
    return score_entities(predictions, truth)


def _append_audit(path: Path, lock: threading.Lock, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def run_fusion_batch(
    input_path: Path,
    output_dir: Path,
    anchor_zip: Path,
    base_url: str,
    model: str,
    workers: int = 2,
    passes: int = 3,
    resume: bool = False,
    validation_path: Path | None = None,
    limit: int | None = None,
) -> dict:
    """Run the anchored strategy and produce output/, metrics.json, audit.jsonl, ZIP."""
    from kb import KnowledgeBase
    from scripts.build_submission import validate_submission

    if not input_path.is_dir():
        raise ValueError("grpo-v2-fusion expects a directory containing numbered .txt files")
    files = sorted(input_path.glob("*.txt"), key=lambda value: int(value.stem))
    if limit is not None:
        files = files[:max(0, limit)]
    if not files:
        raise ValueError(f"no .txt files in {input_path}")
    client = VLLMLLM(base_url=base_url, model=model, timeout=120, max_retries=2)
    if not client.is_available():
        raise RuntimeError(f"vLLM unavailable: {client.last_error}")
    kb = KnowledgeBase()
    kb.load()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir.parent
    cache_dir = run_dir / "cache"
    audit_path = run_dir / "audit.jsonl"
    if not resume and audit_path.exists():
        audit_path.unlink()
    settings = FusionSettings()
    pass_ids = (
        ("general",)
        if passes == 1
        else ("general", "clinical", "medlab")
        if passes == 3
        else (
            "general", "clinical", "medlab", "general_900", "clinical_500",
            "medlab_500", "general_1600",
        )
    )
    lock = threading.Lock()
    totals = defaultdict(int)

    def process(source: Path) -> tuple[str, int, int]:
        record_id = source.stem
        destination = output_dir / f"{record_id}.json"
        if resume and destination.exists():
            _append_audit(audit_path, lock, {"record": record_id, "action": "resumed_output"})
            return record_id, len(json.loads(destination.read_text(encoding="utf-8"))), 0
        text = source.read_text(encoding="utf-8")
        baseline = _baseline_from_zip(anchor_zip, record_id)
        all_proposals, pass_events = [], []
        for pass_id in pass_ids:
            pass_proposals, events = extract_pass(record_id, text, pass_id, client, cache_dir, resume)
            all_proposals.extend(pass_proposals)
            pass_events.extend(events)
        result, decisions = _fuse(text, baseline, all_proposals, kb, settings)
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _append_audit(audit_path, lock, {"record": record_id, "events": pass_events, "decisions": decisions, "baseline": len(baseline), "proposals": len(all_proposals), "output": len(result)})
        return record_id, len(result), len(result) - len(baseline)

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 2))) as executor:
        futures = {executor.submit(process, source): source for source in files}
        for future in as_completed(futures):
            record_id, count, added = future.result()
            totals["records"] += 1
            totals["mentions"] += count
            totals["added"] += added
            print(f"[{totals['records']}/{len(files)}] record={record_id} mentions={count} added={added}", flush=True)

    full_submission = len(files) == 100 and limit is None
    if full_submission:
        errors = validate_submission(output_dir, input_path)
        if errors:
            raise RuntimeError("fusion output invalid: " + "; ".join(errors[:10]))
        zip_path = run_dir / "output.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for source in sorted(output_dir.glob("*.json"), key=lambda value: int(value.stem)):
                archive.write(source, arcname=f"output/{source.name}")
        with zipfile.ZipFile(zip_path) as archive:
            zip_errors = [name for name in archive.namelist() if not re.fullmatch(r"output/(?:[1-9]|[1-9][0-9]|100)\.json", name)]
            if len(archive.namelist()) != 100 or zip_errors:
                raise RuntimeError("packaged ZIP does not contain exactly output/1.json … output/100.json")
    else:
        from tools import validate_output
        errors = []
        for source in files:
            output = json.loads((output_dir / f"{source.stem}.json").read_text(encoding="utf-8"))
            errors.extend(validate_output(output)["errors"])
        if errors:
            raise RuntimeError("fusion smoke output invalid: " + "; ".join(errors[:10]))
        zip_path = None

    metrics = {
        "strategy": "grpo-v2-fusion",
        "model": model,
        "base_url": base_url,
        "passes": list(pass_ids),
        "records": totals["records"],
        "mentions": totals["mentions"],
        "added": totals["added"],
        "validation": None,
        "submission_valid": full_submission,
        "zip": str(zip_path) if zip_path else None,
    }
    # Validation data is an optional explicit calibration input.  Scoring the
    # anchor against it is not meaningful (different records), so report the
    # scorer contract and leave production threshold selection deterministic.
    if validation_path:
        rows = _load_validation(validation_path)
        metrics["validation"] = {
            "path": str(validation_path),
            "records": len(rows),
            "score_formula": "0.3 * text + 0.3 * assertions + 0.4 * candidates",
            "selected_settings": settings.__dict__,
            "note": "Validation split registered for reproducible ablations; use --strategy on a held-out anchor set for scored candidate threshold selection.",
        }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics
