#!/usr/bin/env python3
"""Combine a reviewed candidate-only source with dual-approved agent edits."""

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
from run_agentic_full_pipeline import TYPE_CODE_OVERRIDES, normalise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--agent-audit", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    approved = []
    with args.agent_audit.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                event = json.loads(line)
                if event.get("approved") and event.get("case", {}).get("kind") in {"type", "delete"}:
                    approved.append(event["case"])
    by_record: dict[int, list[dict]] = {}
    for case in approved:
        by_record.setdefault(int(case["record"]), []).append(case)

    args.output.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    applied = []
    for record in range(1, 101):
        entities = json.loads((args.candidate_source / f"{record}.json").read_text(encoding="utf-8"))
        for case in by_record.get(record, []):
            position = tuple(case["position"])
            entity = next((value for value in entities if tuple(value["position"]) == position and value["type"] == case["current_type"]), None)
            if entity is None:
                raise RuntimeError(f"record {record}: missing approved target {position}")
            if case["kind"] == "delete":
                entities.remove(entity)
                stats["deleted"] += 1
                applied.append(case)
                continue
            entity["type"] = case["proposed_type"]
            if entity["type"] in {"CHẨN_ĐOÁN", "THUỐC"}:
                code = TYPE_CODE_OVERRIDES.get(normalise(entity["text"]), "")
                entity["candidates"] = [code] if code else []
            else:
                entity.pop("candidates", None)
            stats["retyped"] += 1
            applied.append(case)
        entities.sort(key=lambda value: (value["position"][0], value["position"][1], value["type"]))
        (args.output / f"{record}.json").write_text(
            json.dumps(entities, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    errors = validate_submission(args.output, args.inputs)
    if errors:
        raise RuntimeError("; ".join(errors[:10]))
    run_dir = args.output.parent
    (run_dir / "agent_transactions.jsonl").write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in applied), encoding="utf-8"
    )
    zip_path = run_dir / "output.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for record in range(1, 101):
            archive.write(args.output / f"{record}.json", f"output/{record}.json")
    print(json.dumps({**stats, "zip": str(zip_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
