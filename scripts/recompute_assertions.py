#!/usr/bin/env python3
"""Recompute scoped assertions for a generated output directory."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agents.worker_agents import AssertionAgent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "input")
    args = parser.parse_args()
    agent = AssertionAgent()
    changed_files = changed_entities = 0
    for path in sorted(args.output_dir.glob("*.json"), key=lambda value: int(value.stem)):
        input_path = args.input_dir / f"{path.stem}.txt"
        text = input_path.read_text(encoding="utf-8")
        entities = json.loads(path.read_text(encoding="utf-8"))
        changed = 0
        for entity in entities:
            if entity.get("type") not in ("CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"):
                assertions = []
            else:
                values = agent._rule_assert(
                    text,
                    entity["position"][0],
                    entity["position"][1],
                    entity.get("type", ""),
                )
                assertions = [
                    name for name in ("isNegated", "isFamily", "isHistorical")
                    if values.get(name)
                ]
            if entity.get("assertions") != assertions:
                entity["assertions"] = assertions
                changed += 1
        if changed:
            path.write_text(
                json.dumps(entities, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            changed_files += 1
            changed_entities += changed
    print(f"Recomputed {changed_entities} assertions in {changed_files} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
