#!/usr/bin/env python3
"""Keep the earliest mention for each normalized (surface, type) per record."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from build_submission import validate_submission


def _normalise(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split()).strip(" .,:;()")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    for record in range(1, 101):
        entities = json.loads((args.source / f"{record}.json").read_text(encoding="utf-8"))
        entities.sort(key=lambda e: (e["position"][0], e["position"][1], e["type"]))
        seen = set()
        kept = []
        for entity in entities:
            key = (_normalise(entity["text"]), entity["type"])
            if key in seen:
                stats["duplicates_removed"] += 1
                stats[f"removed_{entity['type']}"] += 1
                continue
            seen.add(key)
            kept.append(entity)
        (args.output / f"{record}.json").write_text(
            json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    errors = validate_submission(args.output, args.inputs)
    if errors:
        raise RuntimeError("; ".join(errors[:10]))
    zip_path = args.output.parent / "output.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for record in range(1, 101):
            archive.write(args.output / f"{record}.json", f"output/{record}.json")
    print(json.dumps({**stats, "zip": str(zip_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
