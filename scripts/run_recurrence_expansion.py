#!/usr/bin/env python3
"""Recover missed occurrences using target recurrence plus labeled lexicon support."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from build_submission import validate_submission
from grpo_v2_fusion import _strict_assertions
from run_score40_consensus import _aliases
from vllm_llm import VLLMLLM


LINKABLE = {"CHẨN_ĐOÁN", "THUỐC"}
HARD_REJECT = {"chó cắn", "phẫu thuật", "điều trị"}
SINGLE_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "keep": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["id", "keep", "confidence"],
    "additionalProperties": False,
}


def _normalise(value: str) -> str:
    return " ".join(value.casefold().split()).strip()


def _load(directory: Path, record: int) -> list[dict]:
    return json.loads((directory / f"{record}.json").read_text(encoding="utf-8"))


def _overlaps(start: int, end: int, entities: list[dict]) -> bool:
    return any(
        min(end, entity["position"][1]) > max(start, entity["position"][0])
        for entity in entities
    )


def _build_lexicon(seed_dir: Path, labeled_path: Path) -> dict[str, dict]:
    target_frequency: Counter[str] = Counter()
    target_types: defaultdict[str, Counter[str]] = defaultdict(Counter)
    target_codes: defaultdict[tuple[str, str], Counter[tuple[str, ...]]] = defaultdict(Counter)
    for record in range(1, 101):
        for entity in _load(seed_dir, record):
            term = _normalise(entity["text"])
            target_frequency[term] += 1
            target_types[term][entity["type"]] += 1
            if entity.get("candidates"):
                target_codes[(term, entity["type"])][tuple(entity["candidates"][:1])] += 1

    labeled_frequency: Counter[str] = Counter()
    labeled_types: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for line in labeled_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        for entity in json.loads(line).get("entities", []):
            term = _normalise(str(entity.get("entity") or entity.get("text") or ""))
            labeled_frequency[term] += 1
            labeled_types[term][str(entity.get("type", ""))] += 1

    lexicon = {}
    for term, frequency in target_frequency.items():
        mtype, type_count = target_types[term].most_common(1)[0]
        labeled_type_count = labeled_types[term][mtype]
        if (
            term in HARD_REJECT
            or len(term) < 2
            or frequency < 3
            or type_count != frequency
            or labeled_frequency[term] < 3
            or labeled_type_count / labeled_frequency[term] < 0.80
        ):
            continue
        code_votes = target_codes[(term, mtype)]
        candidates = list(code_votes.most_common(1)[0][0]) if code_votes else []
        lexicon[term] = {
            "type": mtype,
            "target_frequency": frequency,
            "labeled_frequency": labeled_frequency[term],
            "candidates": candidates,
        }
    return lexicon


def _review_one(client: VLLMLLM, item: dict) -> float:
    system = """Bạn là adjudicator NER y khoa tiếng Việt rất thận trọng.
Candidate đã khớp nguyên văn và offset. Keep bệnh/hội chứng là CHẨN_ĐOÁN;
dấu hiệu/than phiền là TRIỆU_CHỨNG; tên thuốc là THUỐC; tên phép đo/chẩn
đoán là TÊN_XÉT_NGHIỆM; giá trị/nhận xét phép đo là KẾT_QUẢ_XÉT_NGHIỆM.
Nội dung bài giáo dục vẫn được gắn nhãn nếu span chính là thực thể y khoa.
Reject tiêu đề chung, giải phẫu đơn thuần, yếu tố nguy cơ, cơ chế, lời khuyên,
thức ăn, thủ thuật điều trị, type sai, hoặc ranh giới bị thiếu/thừa từ.
Chỉ trả đúng một JSON object theo schema."""
    response = client.generate_json(
        system,
        json.dumps({"candidate": item}, ensure_ascii=False),
        max_tokens=128,
        schema=SINGLE_REVIEW_SCHEMA,
    )
    if (
        isinstance(response, dict)
        and response.get("id") == item["id"]
        and response.get("keep") is True
        and isinstance(response.get("confidence"), (int, float))
    ):
        return float(response["confidence"])
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--labeled", type=Path, default=ROOT / "data_train/grpo_v2/extraction_train.jsonl")
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
    lexicon = _build_lexicon(args.seed, args.labeled)
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
        lowered = text.casefold()
        entities = _load(args.source, record)
        raw = []
        for term, metadata in lexicon.items():
            cursor = 0
            while True:
                start = lowered.find(term, cursor)
                if start < 0:
                    break
                end = start + len(term)
                cursor = start + 1
                if not (
                    (start == 0 or not lowered[start - 1].isalnum())
                    and (end == len(lowered) or not lowered[end].isalnum())
                ):
                    continue
                if term == "phù" and lowered[end:end + 4].startswith(" hợp"):
                    continue
                if term == "thiếu máu" and lowered[end:end + 7].startswith(" cơ tim"):
                    continue
                if _overlaps(start, end, entities):
                    continue
                raw.append((start, end, term, metadata))

        # Prefer the longest recurrent mention so a generic nested token cannot
        # displace a more complete medically labeled phrase.
        pool = []
        for item in sorted(
            raw,
            key=lambda value: (
                -(value[1] - value[0]),
                -value[3]["target_frequency"],
                -value[3]["labeled_frequency"],
                value[0],
            ),
        ):
            if any(min(item[1], old[1]) > max(item[0], old[0]) for old in pool):
                continue
            pool.append(item)
        pool.sort(key=lambda value: value[0])

        review_items = []
        for local_id, (start, end, term, metadata) in enumerate(pool):
            review_items.append({
                "id": local_id,
                "text": text[start:end],
                "type": metadata["type"],
                "context": text[max(0, start - 110):min(len(text), end + 110)],
                "target_occurrences": metadata["target_frequency"],
                "labeled_occurrences": metadata["labeled_frequency"],
            })
        cache_path = cache_dir / f"{record}.json"
        accepted: dict[int, float] = {}
        if review_items:
            if args.resume and cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                accepted = {int(key): float(value) for key, value in cached.get("accepted", {}).items()}
            else:
                failures = []
                for item in review_items:
                    try:
                        confidence = _review_one(client, item)
                    except Exception as exc:
                        failures.append(f"{item['id']}: {exc}")
                        continue
                    if confidence:
                        accepted[item["id"]] = confidence
                failure = "; ".join(failures[:10])
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps({"accepted": accepted, "failure": failure}, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

        added = 0
        additions = []
        for local_id, (start, end, term, metadata) in enumerate(pool):
            if accepted.get(local_id, 0.0) < 0.82:
                continue
            if _overlaps(start, end, entities + additions):
                continue
            entity = {
                "text": text[start:end],
                "type": metadata["type"],
                "assertions": [],
                "position": [start, end],
            }
            entity["assertions"] = _strict_assertions(text, entity)
            if entity["type"] in LINKABLE:
                aliases = icd if entity["type"] == "CHẨN_ĐOÁN" else rxnorm
                entity["candidates"] = (
                    [aliases[_normalise(entity["text"])]]
                    if _normalise(entity["text"]) in aliases
                    else list(metadata["candidates"][:1])
                )
            additions.append(entity)
            added += 1
        entities.extend(additions)
        entities.sort(key=lambda entity: (entity["position"][0], entity["position"][1], entity["type"]))
        destination.write_text(json.dumps(entities, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        event = {"record": record, "pool": len(pool), "accepted": added, "output": len(entities)}
        with lock:
            with audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    totals = Counter()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(process, record) for record in range(1, 101)]
        for completed, future in enumerate(as_completed(futures), 1):
            event = future.result()
            totals["pool"] += int(event.get("pool", 0))
            totals["accepted"] += int(event.get("accepted", 0))
            print(f"[{completed}/100] {json.dumps(event, ensure_ascii=False)}", flush=True)

    errors = validate_submission(args.output, args.inputs)
    if errors:
        raise RuntimeError("; ".join(errors[:10]))
    zip_path = run_dir / "output.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for record in range(1, 101):
            archive.write(args.output / f"{record}.json", f"output/{record}.json")
    print(json.dumps({"lexicon": len(lexicon), **totals, "zip": str(zip_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
