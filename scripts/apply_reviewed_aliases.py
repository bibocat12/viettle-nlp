#!/usr/bin/env python3
"""Apply reviewed diagnosis aliases to an already generated output directory."""

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from linker_agent import _reviewed_icd10_code


def load_type_aliases() -> dict:
    path = ROOT / "database-stuff" / "type_aliases.json"
    values = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {" ".join(key.casefold().split()).strip(" .,:;()"): value for key, value in values.items()}


def load_rxnorm_aliases() -> dict:
    values = {}
    for filename in ("rxnorm_aliases.json", "rxnorm_aliases_round2.json"):
        path = ROOT / "database-stuff" / filename
        if path.exists():
            values.update(json.loads(path.read_text(encoding="utf-8")))
    return {" ".join(key.casefold().split()).strip(" .,:;()"): str(value) for key, value in values.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "input")
    args = parser.parse_args()
    type_aliases = load_type_aliases()
    rxnorm_aliases = load_rxnorm_aliases()
    changed_files = changed_entities = 0
    for path in sorted(args.output_dir.glob("*.json"), key=lambda value: int(value.stem)):
        entities = json.loads(path.read_text(encoding="utf-8"))
        source_text = (args.input_dir / f"{path.stem}.txt").read_text(encoding="utf-8")
        changed = 0
        kept_entities = []
        for entity in entities:
            if entity.get("type") == "THUỐC":
                original = entity.get("text", "")
                leading = re.match(
                    r"^(?:được\s+)?(?:điều\s+trị\s+)?(?:bằng\s+cách\s+tăng\s+liều|"
                    r"bằng|uống|dùng|tiêm)\s+",
                    original,
                    flags=re.IGNORECASE,
                )
                if leading:
                    entity["text"] = original[leading.end():]
                    entity["position"][0] += leading.end()
                    changed += 1
                normalized_drug = " ".join(entity.get("text", "").casefold().split()).strip(" .,:;()")
                if re.match(r"^(?:from|còn|từ|xuống|lên|khoảng)\s+\d+(?:[.,]\d+)?\s*(?:mg|ml|mcg|iu|u)$", normalized_drug):
                    changed += 1
                    continue
                alias_hits = []
                for alias, code in sorted(rxnorm_aliases.items(), key=lambda item: len(item[0]), reverse=True):
                    for hit in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized_drug):
                        if not any(hit.start() < old_end and old_start < hit.end() for old_start, old_end, _ in alias_hits):
                            alias_hits.append((hit.start(), hit.end(), code))
                if len(alias_hits) >= 2:
                    for hit_start, hit_end, code in sorted(alias_hits):
                        absolute_start = entity["position"][0] + hit_start
                        kept_entities.append({
                            "text": source_text[absolute_start:absolute_start + (hit_end - hit_start)],
                            "type": "THUỐC",
                            "assertions": list(entity.get("assertions", [])),
                            "position": [absolute_start, absolute_start + (hit_end - hit_start)],
                            "candidates": [code],
                        })
                    changed += 1
                    continue
                reviewed_rxnorm = rxnorm_aliases.get(normalized_drug)
                if reviewed_rxnorm and entity.get("candidates") != [reviewed_rxnorm]:
                    entity["candidates"] = [reviewed_rxnorm]
                    changed += 1
            normalized_text = " ".join(entity.get("text", "").casefold().split()).strip(" .,:;()")
            reviewed_type = type_aliases.get(normalized_text)
            if reviewed_type and entity.get("type") != reviewed_type:
                entity["type"] = reviewed_type
                entity.pop("candidates", None)
                changed += 1
            if entity.get("type") == "CHẨN_ĐOÁN":
                code = _reviewed_icd10_code(entity.get("text", ""))
                if code and entity.get("candidates") != [code]:
                    entity["candidates"] = [code]
                    changed += 1
            kept_entities.append(entity)
        entities = kept_entities
        occupied = {
            (tuple(entity.get("position", [])), entity.get("type"))
            for entity in entities
        }
        for alias, reviewed_type in sorted(type_aliases.items(), key=lambda item: len(item[0]), reverse=True):
            for match in re.finditer(re.escape(alias), source_text, flags=re.IGNORECASE):
                key = ((match.start(), match.end()), reviewed_type)
                if key in occupied:
                    continue
                overlaps_same_type = any(
                    entity.get("type") == reviewed_type
                    and min(match.end(), entity.get("position", [0, 0])[1])
                    > max(match.start(), entity.get("position", [0, 0])[0])
                    for entity in entities
                )
                if overlaps_same_type and alias != "ercp":
                    continue
                added = {
                    "text": source_text[match.start():match.end()],
                    "type": reviewed_type,
                    "assertions": [],
                    "position": [match.start(), match.end()],
                }
                if reviewed_type == "CHẨN_ĐOÁN":
                    code = _reviewed_icd10_code(added["text"])
                    added["candidates"] = [code] if code else []
                entities.append(added)
                occupied.add(key)
                changed += 1
        for alias, code in sorted(rxnorm_aliases.items(), key=lambda item: len(item[0]), reverse=True):
            for match in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", source_text, flags=re.IGNORECASE):
                if any(
                    entity.get("type") == "THUỐC"
                    and min(match.end(), entity.get("position", [0, 0])[1])
                    > max(match.start(), entity.get("position", [0, 0])[0])
                    for entity in entities
                ):
                    continue
                left = source_text[max(0, match.start() - 24):match.start()]
                dose_prefix = re.search(
                    r"\d+(?:[.,]\d+)?\s*(?:mg|mcg|g|ml|meq)"
                    r"(?:\s+(?:po|iv|im|sc|sl))?\s+$",
                    left,
                    flags=re.IGNORECASE,
                )
                start = match.start() - len(dose_prefix.group(0)) if dose_prefix else match.start()
                added = {
                    "text": source_text[start:match.end()],
                    "type": "THUỐC",
                    "assertions": [],
                    "position": [start, match.end()],
                    "candidates": [code],
                }
                entities.append(added)
                changed += 1
        filtered_entities = []
        for entity in entities:
            if entity.get("type") != "THUỐC":
                filtered_entities.append(entity)
                continue
            normalized_drug = " ".join(entity.get("text", "").casefold().split()).strip(" .,:;()")
            contains_reviewed_name = any(
                re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized_drug)
                for alias in rxnorm_aliases
            )
            start, end = entity.get("position", [0, 0])
            overlaps_better_drug = any(
                other is not entity
                and other.get("type") == "THUỐC"
                and other.get("candidates")
                and min(end, other.get("position", [0, 0])[1])
                > max(start, other.get("position", [0, 0])[0])
                for other in entities
            )
            looks_like_orphan_dose = bool(re.match(
                r"^(?:from|còn|từ|xuống|lên|khoảng|thêm|nhận|cho)\b.*"
                r"\d+(?:[.,]\d+)?\s*(?:mg|ml|mcg|g|meq|iu|u)\b",
                normalized_drug,
            ))
            generic_without_code = (
                normalized_drug in {"thuốc giảm đau opioid", "kháng sinh"}
                and not entity.get("candidates")
            )
            if not contains_reviewed_name and (
                overlaps_better_drug or looks_like_orphan_dose or generic_without_code
            ):
                changed += 1
                continue
            filtered_entities.append(entity)
        entities = filtered_entities
        entities.sort(key=lambda item: tuple(item.get("position", [0, 0])))
        if changed:
            path.write_text(
                json.dumps(entities, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            changed_files += 1
            changed_entities += changed
    print(f"Applied reviewed aliases to {changed_entities} entities in {changed_files} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
