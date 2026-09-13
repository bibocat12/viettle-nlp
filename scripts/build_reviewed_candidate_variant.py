#!/usr/bin/env python3
"""Apply an exact, independently reviewed candidate correction ledger.

This builder deliberately cannot add, delete, retype, move, or change the
assertions of an entity.  It is used as the candidate-only safety ablation for
the agentic run.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from build_submission import validate_submission


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tier", choices=("A", "B"), default="A")
    args = parser.parse_args()

    ledger = json.loads(args.corrections.read_text(encoding="utf-8"))
    if not isinstance(ledger, list):
        raise ValueError("correction ledger must be a JSON array")
    permitted = {"A"} if args.max_tier == "A" else {"A", "B"}
    by_record: dict[int, list[dict]] = {}
    seen: set[tuple[int, int, int]] = set()
    for row in ledger:
        if not isinstance(row, dict) or row.get("tier") not in permitted:
            continue
        record = int(row["record"])
        start, end = map(int, row["position"])
        key = (record, start, end)
        if key in seen:
            raise ValueError(f"duplicate correction target: {key}")
        seen.add(key)
        by_record.setdefault(record, []).append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    audit: list[dict] = []
    for record in range(1, 101):
        text = (args.inputs / f"{record}.txt").read_text(encoding="utf-8")
        entities = json.loads((args.source / f"{record}.json").read_text(encoding="utf-8"))
        targets = {
            tuple(entity["position"]): entity
            for entity in entities
            if entity.get("type") in {"CHẨN_ĐOÁN", "THUỐC"}
        }
        for correction in by_record.get(record, []):
            position = tuple(correction["position"])
            entity = targets.get(position)
            if entity is None:
                raise RuntimeError(f"record {record}: missing linkable target at {position}")
            start, end = position
            if entity.get("text") != text[start:end]:
                raise RuntimeError(f"record {record}: source text mismatch at {position}")
            before = list(map(str, entity.get("candidates", [])))
            code = str(correction.get("candidate", ""))
            after = [code] if code else []
            if before == after:
                stats["already_applied"] += 1
                continue
            entity["candidates"] = after
            stats["cleared" if not after else "updated"] += 1
            audit.append({
                "record": record,
                "position": list(position),
                "text": entity["text"],
                "type": entity["type"],
                "from": before,
                "to": after,
                "tier": correction["tier"],
                "evidence": correction.get("evidence", ""),
            })
        (args.output / f"{record}.json").write_text(
            json.dumps(entities, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    errors = validate_submission(args.output, args.inputs)
    if errors:
        raise RuntimeError("; ".join(errors[:10]))
    run_dir = args.output.parent
    (run_dir / "audit.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit),
        encoding="utf-8",
    )
    zip_path = run_dir / "output.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for record in range(1, 101):
            archive.write(args.output / f"{record}.json", f"output/{record}.json")
    print(json.dumps({**stats, "changes": len(audit), "zip": str(zip_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
