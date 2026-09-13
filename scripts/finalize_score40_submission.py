#!/usr/bin/env python3
"""Finalize the guarded consensus output with trusted candidate restoration."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from build_submission import validate_submission


def _load(directory: Path, record: int) -> list[dict]:
    return json.loads((directory / f"{record}.json").read_text(encoding="utf-8"))


def _key(entity: dict) -> tuple[int, int, str]:
    return entity["position"][0], entity["position"][1], entity["type"]


def _normalise(value: str) -> str:
    return " ".join(value.casefold().split()).strip(" .,:;()")


def _replace_overlapping(
    entities: list[dict], start: int, end: int, replacements: list[dict]
) -> None:
    entities[:] = [
        entity for entity in entities
        if min(end, entity["position"][1]) <= max(start, entity["position"][0])
    ]
    entities.extend(replacements)
    entities.sort(key=lambda entity: (entity["position"][0], entity["position"][1], entity["type"]))


def _apply_span_repairs(record: int, entities: list[dict], text: str) -> int:
    """Apply independently corroborated boundary/type repairs."""
    changes = 0

    # Short symptom boundaries that omit a medically meaningful modifier.
    expansions = {
        6: ((711, 718), (711, 726)),
        11: ((711, 718), (711, 726)),
        15: ((1150, 1153), (1150, 1159)),
        23: ((1755, 1762), (1755, 1770)),
        42: ((874, 881), (874, 894)),
        97: ((534, 541), (534, 554)),
    }
    if record in expansions:
        old_span, new_span = expansions[record]
        for entity in entities:
            if entity["position"] == list(old_span) and entity["type"] == "TRIỆU_CHỨNG":
                entity["position"] = list(new_span)
                entity["text"] = text[new_span[0]:new_span[1]]
                # File 97 is a current positive symptom; its old negation came
                # from a distant syntactic cue outside the mention.
                if record == 97:
                    entity["assertions"] = []
                changes += 1
                break

    if record == 8:
        _replace_overlapping(entities, 2294, 2306, [
            {"text": text[2294:2297], "type": "TRIỆU_CHỨNG", "assertions": [], "position": [2294, 2297]},
            {"text": text[2298:2306], "type": "TRIỆU_CHỨNG", "assertions": [], "position": [2298, 2306]},
        ])
        changes += 1
    elif record == 38:
        _replace_overlapping(entities, 133, 147, [{
            "text": text[133:147], "type": "CHẨN_ĐOÁN",
            "assertions": ["isHistorical"], "position": [133, 147],
            "candidates": ["E78.5"],
        }])
        changes += 1
    elif record == 40:
        for start, end in ((1647, 1650), (1663, 1672)):
            for entity in entities:
                if entity["position"] == [start, end]:
                    entity["type"] = "TÊN_XÉT_NGHIỆM"
                    entity["assertions"] = []
                    entity.pop("candidates", None)
                    changes += 1
                    break
    elif record == 42:
        _replace_overlapping(entities, 51, 68, [
            {"text": text[51:53], "type": "TRIỆU_CHỨNG", "assertions": [], "position": [51, 53]},
            {"text": text[54:68], "type": "CHẨN_ĐOÁN", "assertions": [], "position": [54, 68], "candidates": ["E11.9"]},
        ])
        changes += 1
    elif record == 47:
        _replace_overlapping(entities, 580, 612, [
            {"text": text[580:588], "type": "CHẨN_ĐOÁN", "assertions": [], "position": [580, 588], "candidates": ["I48.91"]},
            {"text": text[592:612], "type": "CHẨN_ĐOÁN", "assertions": [], "position": [592, 612], "candidates": ["I47.1"]},
        ])
        changes += 1
    elif record == 53:
        _replace_overlapping(entities, 1436, 1454, [{
            "text": text[1436:1454], "type": "CHẨN_ĐOÁN", "assertions": [],
            "position": [1436, 1454], "candidates": ["E78.5"],
        }])
        changes += 1
    elif record == 77:
        _replace_overlapping(entities, 1029, 1063, [
            {"text": text[1029:1036], "type": "CHẨN_ĐOÁN", "assertions": [], "position": [1029, 1036], "candidates": ["I50.9"]},
            {"text": text[1039:1063], "type": "CHẨN_ĐOÁN", "assertions": [], "position": [1039, 1063], "candidates": ["N18.5"]},
        ])
        changes += 1
    elif record == 98:
        _replace_overlapping(entities, 326, 340, [{
            "text": text[326:340], "type": "CHẨN_ĐOÁN",
            "assertions": ["isHistorical"], "position": [326, 340],
            "candidates": ["F41.9"],
        }])
        changes += 1
    return changes


def _apply_assertion_repairs(record: int, entities: list[dict]) -> int:
    repairs = {
        (5, 2011, 2014, "TRIỆU_CHỨNG"): [],
        (8, 1626, 1631, "TRIỆU_CHỨNG"): ["isNegated"],
        (61, 1235, 1242, "TÊN_XÉT_NGHIỆM"): [],
        (78, 753, 762, "TRIỆU_CHỨNG"): ["isNegated"],
        (78, 929, 938, "TRIỆU_CHỨNG"): ["isNegated"],
        (87, 272, 285, "TRIỆU_CHỨNG"): ["isNegated"],
    }
    changes = 0
    for entity in entities:
        key = (record, entity["position"][0], entity["position"][1], entity["type"])
        if key in repairs and entity.get("assertions", []) != repairs[key]:
            entity["assertions"] = repairs[key]
            changes += 1
    return changes


def _remove_prompt_violations(record: int, entities: list[dict]) -> int:
    """Remove newly-added treatment/exposure/metadata spans forbidden by the task."""
    violations = {
        (89, 1385, 1395, "KẾT_QUẢ_XÉT_NGHIỆM"),
        (46, 1267, 1308, "TÊN_XÉT_NGHIỆM"),
        (71, 229, 270, "TÊN_XÉT_NGHIỆM"),
        (91, 1174, 1184, "TÊN_XÉT_NGHIỆM"),
        (91, 1291, 1301, "TÊN_XÉT_NGHIỆM"),
        (44, 1285, 1305, "TRIỆU_CHỨNG"),
        (44, 1571, 1589, "KẾT_QUẢ_XÉT_NGHIỆM"),
        (69, 678, 719, "TÊN_XÉT_NGHIỆM"),
        (20, 1656, 1663, "TRIỆU_CHỨNG"),
        (20, 1672, 1679, "TRIỆU_CHỨNG"),
        (70, 1159, 1196, "TÊN_XÉT_NGHIỆM"),
        (67, 151, 156, "TÊN_XÉT_NGHIỆM"),
        (88, 253, 286, "KẾT_QUẢ_XÉT_NGHIỆM"),
        (16, 1727, 1734, "TRIỆU_CHỨNG"),
        (16, 1743, 1750, "TRIỆU_CHỨNG"),
        (74, 1499, 1526, "TRIỆU_CHỨNG"),
    }
    before = len(entities)
    entities[:] = [
        entity for entity in entities
        if (
            record,
            entity["position"][0],
            entity["position"][1],
            entity["type"],
        ) not in violations
    ]
    return before - len(entities)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    unsafe_anchor = {"nổi mề đay", "mù vĩnh viễn"}
    diagnosis_corrections = {
        "dị tật còn ống động mạch": "Q25.0",
        "rối loạn cảm xúc lưỡng cực khác": "F31.8",
    }
    drug_corrections: dict[str, list[str]] = {
        "tinh bột nghệ tách tinh dầu": [],
        "cotrimoxazol": ["10831"],
    }
    stats = {
        "anchor_candidates_restored": 0,
        "unsafe_anchor_candidates_skipped": 0,
        "diagnosis_candidates_corrected": 0,
        "drug_candidates_corrected": 0,
        "span_type_repairs": 0,
        "assertion_repairs": 0,
        "prompt_violations_removed": 0,
    }

    for record in range(1, 101):
        entities = _load(args.source, record)
        anchor = {_key(entity): entity for entity in _load(args.anchor, record)}
        for entity in entities:
            normalized = _normalise(entity["text"])
            old = anchor.get(_key(entity))
            old_candidates = list(old.get("candidates", [])[:1]) if old else []
            if old_candidates and not entity.get("candidates"):
                if normalized in unsafe_anchor:
                    stats["unsafe_anchor_candidates_skipped"] += 1
                else:
                    entity["candidates"] = old_candidates
                    stats["anchor_candidates_restored"] += 1

            if entity["type"] == "CHẨN_ĐOÁN" and normalized in diagnosis_corrections:
                corrected = [diagnosis_corrections[normalized]]
                if entity.get("candidates") != corrected:
                    entity["candidates"] = corrected
                    stats["diagnosis_candidates_corrected"] += 1
            elif entity["type"] == "THUỐC" and normalized in drug_corrections:
                corrected = drug_corrections[normalized]
                if entity.get("candidates") != corrected:
                    entity["candidates"] = corrected
                    stats["drug_candidates_corrected"] += 1

        source_text = (args.inputs / f"{record}.txt").read_text(encoding="utf-8")
        stats["span_type_repairs"] += _apply_span_repairs(record, entities, source_text)
        stats["assertion_repairs"] += _apply_assertion_repairs(record, entities)
        stats["prompt_violations_removed"] += _remove_prompt_violations(record, entities)

        (args.output / f"{record}.json").write_text(
            json.dumps(entities, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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
