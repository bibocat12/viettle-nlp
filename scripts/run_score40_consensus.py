#!/usr/bin/env python3
"""Build a high-recall, guarded consensus submission from scored pipelines.

The scored three-pass output remains immutable.  Seven-pass additions are
ranked by independent pass support; very strong additions are accepted
deterministically and borderline cross-pipeline additions are adjudicated by
the NER checkpoint.  Curated ICD-10/RxNorm aliases are authoritative even when
the local terminology CSV happens to omit a dotted leaf code.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from build_submission import validate_submission
from grpo_v2_fusion import _strict_assertions
from vllm_llm import VLLMLLM


REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "keep": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["id", "keep", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["reviews"],
    "additionalProperties": False,
}
VALID_TYPES = {
    "CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "THUỐC",
    "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM",
}


def _load(directory: Path, record: int) -> list[dict]:
    value = json.loads((directory / f"{record}.json").read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{directory}/{record}.json is not an array")
    return value


def _key(entity: dict) -> tuple[int, int, str]:
    return entity["position"][0], entity["position"][1], entity["type"]


def _overlaps(entity: dict, others: list[dict]) -> bool:
    start, end = entity["position"]
    return any(
        min(end, other["position"][1]) > max(start, other["position"][0])
        for other in others
    )


def _load_support(path: Path) -> dict[int, dict[tuple[int, int, str], int]]:
    latest: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if "record" in value and "decisions" in value:
            latest[int(value["record"])] = value
    result: dict[int, dict[tuple[int, int, str], int]] = {}
    for record, value in latest.items():
        supports: dict[tuple[int, int, str], int] = {}
        for decision in value["decisions"]:
            if decision.get("action") != "accepted":
                continue
            key = (decision["position"][0], decision["position"][1], decision["type"])
            supports[key] = max(supports.get(key, 0), len(decision.get("support", [])))
        result[record] = supports
    return result


def _aliases() -> tuple[dict[str, str], dict[str, str]]:
    def load(names: tuple[str, ...]) -> dict[str, str]:
        values: dict[str, str] = {}
        for name in names:
            raw = json.loads((ROOT / "database-stuff" / name).read_text(encoding="utf-8"))
            values.update({" ".join(key.casefold().split()).strip(" .,:;()"): str(code) for key, code in raw.items()})
        return values
    return (
        load(("icd10_aliases.json", "icd10_aliases_round2.json")),
        load(("rxnorm_aliases.json", "rxnorm_aliases_round2.json")),
    )


def _normalise(value: str) -> str:
    return " ".join(value.casefold().split()).strip(" .,:;()")


def _review(client: VLLMLLM, items: list[dict]) -> dict[int, float]:
    if not items:
        return {}
    system = """Bạn là adjudicator NER y khoa tiếng Việt rất thận trọng.
Mỗi candidate đã khớp nguyên văn và offset; chỉ quyết định keep hay reject.
Keep bệnh/hội chứng là CHẨN_ĐOÁN; dấu hiệu/than phiền là TRIỆU_CHỨNG; tên
thuốc là THUỐC; tên phép đo/chẩn đoán là TÊN_XÉT_NGHIỆM; giá trị hoặc nhận
xét của phép đo là KẾT_QUẢ_XÉT_NGHIỆM. Nội dung bài giáo dục vẫn được gắn
nhãn nếu chính span là thực thể y khoa. Reject tiêu đề mục chung, giải phẫu
đơn thuần, yếu tố nguy cơ, cơ chế, lời khuyên, thức ăn, thủ thuật điều trị,
span sai type, và span chứa thừa động từ/liên từ. Không sửa hoặc tạo span.
Chỉ trả JSON theo schema, một review cho từng id."""
    response = client.generate_json(
        system,
        json.dumps({"candidates": items}, ensure_ascii=False),
        max_tokens=1024,
        schema=REVIEW_SCHEMA,
    )
    reviews = response.get("reviews", []) if isinstance(response, dict) else []
    accepted: dict[int, float] = {}
    valid_ids = {item["id"] for item in items}
    for review in reviews if isinstance(reviews, list) else []:
        if not isinstance(review, dict) or review.get("id") not in valid_ids:
            continue
        if review.get("keep") is True and isinstance(review.get("confidence"), (int, float)):
            accepted[int(review["id"])] = float(review["confidence"])
    return accepted


def _apply_curated_aliases(
    entities: list[dict], text: str, icd: dict[str, str], rxnorm: dict[str, str]
) -> tuple[int, int]:
    corrected = 0
    for entity in entities:
        aliases = icd if entity["type"] == "CHẨN_ĐOÁN" else rxnorm if entity["type"] == "THUỐC" else None
        if aliases is None:
            continue
        code = aliases.get(_normalise(entity["text"]))
        if code and entity.get("candidates") != [code]:
            entity["candidates"] = [code]
            corrected += 1

    additions: list[tuple[int, int, str, str, str]] = []
    for mtype, aliases in (("CHẨN_ĐOÁN", icd), ("THUỐC", rxnorm)):
        for alias, code in aliases.items():
            for match in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text, re.IGNORECASE):
                start, end = match.span()
                # In Vietnamese cardiology, "thiếu máu cơ tim" is ischemia;
                # the nested generic anaemia alias must not become D64.9.
                if alias == "thiếu máu" and re.match(r"\s+cơ\s+tim", text[end:end + 12], re.IGNORECASE):
                    continue
                additions.append((start, end, mtype, code, text[start:end]))
    maximal = [
        item for item in additions
        if not any(
            other[0] <= item[0] and item[1] <= other[1]
            and other[1] - other[0] > item[1] - item[0]
            for other in additions
        )
    ]
    added = 0
    for start, end, mtype, code, mention in sorted(maximal):
        entity = {"text": mention, "type": mtype, "position": [start, end]}
        if _overlaps(entity, entities):
            continue
        entity["assertions"] = _strict_assertions(text, entity)
        entity["candidates"] = [code]
        entities.append(entity)
        added += 1
    entities.sort(key=lambda item: (item["position"][0], item["position"][1], item["type"]))
    return corrected, added


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--seven-pass", type=Path, required=True)
    parser.add_argument("--seven-audit", type=Path, required=True)
    parser.add_argument("--cross", type=Path, required=True)
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
    support = _load_support(args.seven_audit)
    icd, rxnorm = _aliases()
    args.output.mkdir(parents=True, exist_ok=True)
    run_dir = args.output.parent
    cache_dir = run_dir / "cache"
    audit_path = run_dir / "audit.jsonl"
    lock = threading.Lock()

    def process(record: int) -> dict:
        destination = args.output / f"{record}.json"
        if args.resume and destination.exists():
            return {"record": record, "resumed": True, "output": len(_load(args.output, record))}
        text = (args.inputs / f"{record}.txt").read_text(encoding="utf-8")
        base = _load(args.base, record)
        seven = _load(args.seven_pass, record)
        cross_keys = {_key(entity) for entity in _load(args.cross, record)}
        base_keys = {_key(entity) for entity in base}
        pool = []
        for entity in seven:
            key = _key(entity)
            votes = support.get(record, {}).get(key, 0)
            cross_confirmed = key in cross_keys
            if key in base_keys or not (votes >= 3 or (votes >= 2 and cross_confirmed)):
                continue
            if _overlaps(entity, base):
                continue
            pool.append((dict(entity), votes, cross_confirmed))

        automatic = [entity for entity, votes, _ in pool if votes >= 4]
        borderline = [(entity, votes, cross) for entity, votes, cross in pool if votes < 4]
        review_items = []
        for local_id, (entity, votes, cross_confirmed) in enumerate(borderline):
            start, end = entity["position"]
            context = text[max(0, start - 110):min(len(text), end + 110)]
            review_items.append({
                "id": local_id,
                "text": entity["text"],
                "type": entity["type"],
                "context": context,
                "pass_support": votes,
                "cross_pipeline": cross_confirmed,
            })
        cache_path = cache_dir / f"{record}.json"
        accepted: dict[int, float] = {}
        if review_items:
            if args.resume and cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                accepted = {int(key): float(value) for key, value in cached.get("accepted", {}).items()}
            else:
                try:
                    accepted = _review(client, review_items)
                except Exception as exc:
                    accepted = {}
                    failure = str(exc)
                else:
                    failure = ""
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps({"accepted": accepted, "failure": failure}, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

        additions = list(automatic)
        for local_id, (entity, votes, cross_confirmed) in enumerate(borderline):
            confidence = accepted.get(local_id, 0.0)
            threshold = 0.78 if votes >= 3 else 0.84
            if cross_confirmed:
                threshold -= 0.04
            if confidence >= threshold:
                additions.append(entity)
        additions.sort(key=lambda item: (-support.get(record, {}).get(_key(item), 0), item["position"][0]))
        for entity in additions:
            if _overlaps(entity, base):
                continue
            entity["text"] = text[entity["position"][0]:entity["position"][1]]
            entity["assertions"] = _strict_assertions(text, entity)
            if entity["type"] not in {"CHẨN_ĐOÁN", "THUỐC"}:
                entity.pop("candidates", None)
            base.append(entity)
        corrected, alias_added = _apply_curated_aliases(base, text, icd, rxnorm)
        base.sort(key=lambda item: (item["position"][0], item["position"][1], item["type"]))
        destination.write_text(json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        event = {
            "record": record,
            "pool": len(pool),
            "automatic": len(automatic),
            "agent_accepted": len(additions) - len(automatic),
            "alias_corrected": corrected,
            "alias_added": alias_added,
            "output": len(base),
        }
        with lock:
            with audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    totals = {"automatic": 0, "agent_accepted": 0, "alias_corrected": 0, "alias_added": 0}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(process, record) for record in range(1, 101)]
        for completed, future in enumerate(as_completed(futures), 1):
            event = future.result()
            for key in totals:
                totals[key] += int(event.get(key, 0))
            print(f"[{completed}/100] {json.dumps(event, ensure_ascii=False)}", flush=True)

    errors = validate_submission(args.output, args.inputs)
    if errors:
        raise RuntimeError("; ".join(errors[:10]))
    zip_path = run_dir / "output.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for record in range(1, 101):
            archive.write(args.output / f"{record}.json", f"output/{record}.json")
    print(json.dumps({**totals, "zip": str(zip_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
