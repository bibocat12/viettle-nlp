#!/usr/bin/env python3
"""Deterministically repair offsets, assertions, and unsafe links in a submission.

The input extraction is retained.  This command is deliberately non-generative:
it relocates every mention against the original note, re-evaluates assertion
scope, and takes links only from a validated anchor submission or a curated
local alias.  It never changes the source directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from build_submission import validate_submission
from grpo_v2_fusion import _strict_assertions
from kb import KnowledgeBase
from linker_agent import _reviewed_icd10_code, _reviewed_rxnorm_code
from position_fix import fix_positions


LINKABLE_TYPES = {"CHẨN_ĐOÁN", "THUỐC"}


def _load_entities(directory: Path, record: int) -> list[dict]:
    value = json.loads((directory / f"{record}.json").read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{directory}/{record}.json must contain an array")
    return value


def _anchor_links(directory: Path, record: int) -> dict[tuple[int, int, str], list[str]]:
    links: dict[tuple[int, int, str], list[str]] = {}
    for entity in _load_entities(directory, record):
        position = entity.get("position")
        if (
            isinstance(position, list)
            and len(position) == 2
            and all(isinstance(item, int) for item in position)
            and entity.get("type") in LINKABLE_TYPES
        ):
            links[(position[0], position[1], entity["type"])] = list(entity.get("candidates", [])[:1])
    return links


def _curated_link(entity: dict, kb: KnowledgeBase) -> list[str]:
    if entity["type"] == "CHẨN_ĐOÁN":
        code = _reviewed_icd10_code(entity["text"])
        return [code] if code and code in kb.icd10 else []
    if entity["type"] == "THUỐC":
        code = _reviewed_rxnorm_code(entity["text"])
        return [code] if code and code in kb.rxnorm_concepts else []
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Unmodified source output directory")
    parser.add_argument("--anchor", type=Path, required=True, help="Validated high-quality output directory")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    kb = KnowledgeBase(); kb.load()
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {"input": 0, "relocated": 0, "dropped": 0, "anchor_links": 0, "curated_links": 0}

    for record in range(1, 101):
        text = (args.inputs / f"{record}.txt").read_text(encoding="utf-8")
        source = _load_entities(args.source, record)
        counts["input"] += len(source)
        fixed = fix_positions([dict(entity) for entity in source], text)
        counts["dropped"] += len(source) - len(fixed)
        counts["relocated"] += sum(bool(entity.pop("position_fixed", False)) for entity in fixed)
        anchors = _anchor_links(args.anchor, record)
        cleaned: list[dict] = []
        seen: set[tuple[int, int, str]] = set()
        for entity in fixed:
            mtype = entity.get("type")
            position = entity.get("position")
            if mtype not in {"CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "THUỐC", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}:
                continue
            if not (isinstance(position, list) and len(position) == 2):
                continue
            key = (position[0], position[1], mtype)
            if key in seen:
                continue
            seen.add(key)
            entity = {
                "text": text[position[0]:position[1]],
                "type": mtype,
                "assertions": _strict_assertions(text, entity),
                "position": position,
            }
            if mtype in LINKABLE_TYPES:
                if key in anchors:
                    entity["candidates"] = anchors[key]
                    counts["anchor_links"] += bool(anchors[key])
                else:
                    entity["candidates"] = _curated_link(entity, kb)
                    counts["curated_links"] += bool(entity["candidates"])
            cleaned.append(entity)
        cleaned.sort(key=lambda entity: (entity["position"][0], entity["position"][1], entity["type"]))
        (args.output / f"{record}.json").write_text(
            json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    errors = validate_submission(args.output, args.inputs)
    if errors:
        raise RuntimeError("; ".join(errors[:10]))
    zip_path = args.output.parent / "output.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for record in range(1, 101):
            archive.write(args.output / f"{record}.json", f"output/{record}.json")
    print(json.dumps({**counts, "zip": str(zip_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
