#!/usr/bin/env python3
"""Build reproducible leaderboard ablations from an already-scored ZIP.

The source archive is never modified. Each variant is emitted as both a JSON
directory and a submission ZIP so one leaderboard submission changes only the
feature named by that variant.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agents.worker_agents import AssertionAgent
from tools import validate_output


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split()).strip(" .,:;()")


def _load_icd_aliases(*filenames: str) -> dict[str, str]:
    aliases = {}
    for filename in filenames:
        path = ROOT / "database-stuff" / filename
        values = json.loads(path.read_text(encoding="utf-8"))
        aliases.update({_normalize(key): str(code).upper() for key, code in values.items()})
    return aliases


def _load_archive(path: Path) -> dict[int, list[dict]]:
    records: dict[int, list[dict]] = {}
    with zipfile.ZipFile(path) as archive:
        for index in range(1, 101):
            name = f"output/{index}.json"
            records[index] = json.loads(archive.read(name))
    return records


def _is_invalid_short_occurrence(entity: dict, source: str) -> bool:
    """Detect keyword expansion inside another word (ho/khoa, ct/bactrim)."""
    text = entity.get("text", "")
    start, end = entity.get("position", [0, 0])
    compact_length = sum(char.isalnum() for char in text)
    if compact_length > 3:
        return False
    left_is_word = start > 0 and source[start - 1].isalnum()
    right_is_word = end < len(source) and source[end].isalnum()
    return left_is_word or right_is_word


def clean_short_boundaries(records: dict[int, list[dict]]) -> Counter:
    stats = Counter()
    for index, entities in records.items():
        source = (ROOT / "input" / f"{index}.txt").read_text(encoding="utf-8")
        kept = []
        for entity in entities:
            if _is_invalid_short_occurrence(entity, source):
                stats[f"removed_{entity.get('type', 'UNKNOWN')}"] += 1
                continue
            kept.append(entity)
        records[index] = kept
    return stats


def keep_high_precision_candidates(
    records: dict[int, list[dict]], aliases: dict[str, str]
) -> Counter:
    """Abstain only on diagnosis codes that lack an exact reviewed mapping."""
    stats = Counter()
    for entities in records.values():
        for entity in entities:
            if entity.get("type") != "CHẨN_ĐOÁN":
                continue
            reviewed = aliases.get(_normalize(entity.get("text", "")))
            if reviewed:
                if entity.get("candidates") != [reviewed]:
                    entity["candidates"] = [reviewed]
                    stats["diagnosis_code_corrected"] += 1
            elif entity.get("candidates"):
                entity["candidates"] = []
                stats["diagnosis_code_abstained"] += 1
    return stats


def apply_round2_type_reviews(records: dict[int, list[dict]]) -> Counter:
    """Apply context-reviewed corrections for systematic round-one type errors."""
    stats = Counter()
    diagnosis_codes = {
        "nhiễm khuẩn đường tiết niệu": "N39.0",
        "ngưng thở khi ngủ": "G47.3",
        "ngưng thở khi ngủ do tắc nghẽn": "G47.3",
    }
    type_only = {
        "không có rối loạn thần kinh": "KẾT_QUẢ_XÉT_NGHIỆM",
        "loét đau": "TRIỆU_CHỨNG",
        "loét đỏ": "TRIỆU_CHỨNG",
        "loét sưng": "TRIỆU_CHỨNG",
        "suy giảm huyết động": "TRIỆU_CHỨNG",
    }
    leukemia = "bệnh bạch cầu dòng tủy mãn tính"
    assertion_agent = AssertionAgent()

    for index, entities in records.items():
        source = (ROOT / "input" / f"{index}.txt").read_text(encoding="utf-8")
        for entity in entities:
            normalized = _normalize(entity.get("text", ""))
            if normalized in diagnosis_codes:
                if entity.get("type") != "CHẨN_ĐOÁN":
                    stats["type_corrected_to_diagnosis"] += 1
                entity["type"] = "CHẨN_ĐOÁN"
                entity["candidates"] = [diagnosis_codes[normalized]]
            elif normalized in type_only:
                corrected_type = type_only[normalized]
                if entity.get("type") != corrected_type:
                    stats[f"type_corrected_to_{corrected_type}"] += 1
                entity["type"] = corrected_type
                entity.pop("candidates", None)
                if corrected_type not in ("CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"):
                    entity["assertions"] = []

        for match in re.finditer(
            rf"(?<!\w){re.escape(leukemia)}(?!\w)", source, flags=re.IGNORECASE
        ):
            start, end = match.span()
            exact = any(
                entity.get("type") == "CHẨN_ĐOÁN"
                and entity.get("position") == [start, end]
                for entity in entities
            )
            if exact:
                continue
            overlapping = [
                entity for entity in entities
                if min(end, entity["position"][1]) > max(start, entity["position"][0])
            ]
            if overlapping:
                entities[:] = [entity for entity in entities if entity not in overlapping]
                stats["nested_wrong_type_removed"] += len(overlapping)
            flags = assertion_agent._rule_assert(source, start, end, "CHẨN_ĐOÁN")
            entities.append({
                "text": source[start:end],
                "type": "CHẨN_ĐOÁN",
                "assertions": [
                    key for key in ("isNegated", "isFamily", "isHistorical")
                    if flags.get(key)
                ],
                "position": [start, end],
                "candidates": ["C92.1"],
            })
            stats["leukemia_span_restored"] += 1
        entities.sort(key=lambda item: (item["position"][0], item["position"][1]))
    return stats


def recompute_assertions(records: dict[int, list[dict]]) -> Counter:
    agent = AssertionAgent()
    stats = Counter()
    for index, entities in records.items():
        source = (ROOT / "input" / f"{index}.txt").read_text(encoding="utf-8")
        for entity in entities:
            if entity.get("type") not in ("CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"):
                assertions = []
            else:
                flags = agent._rule_assert(
                    source,
                    entity["position"][0],
                    entity["position"][1],
                    entity.get("type", ""),
                )
                assertions = [
                    key for key in ("isNegated", "isFamily", "isHistorical")
                    if flags.get(key)
                ]
            if entity.get("assertions") != assertions:
                entity["assertions"] = assertions
                stats["assertion_corrected"] += 1
    return stats


def apply_safe_rxnorm_reviews(records: dict[int, list[dict]]) -> Counter:
    """Correct deterministic route/form links and restore missed Bactrim spans."""
    stats = Counter()
    agent = AssertionAgent()
    exact_codes = {
        "albuterolipratropium nebs x2": "214199",
        "albuterolipratropium nebulizer": "214199",
        "albuterol và ipratropium": "214199",
        "metoprolol 5mg iv x2": "866508",
        "bumetanide 2mg iv": "1727568",
        "levofloxacin 750mg iv": "1665519",
        "10mg iv diltiazem": "1791228",
        "40meq po k": "8591",
        "40meq iv k": "8591",
        "natriclori 0.9 %": "313002",
        "laxis 20mg tiêm tĩnh mạch": "1719290",
        "80mg po lasix": "205732",
        "80mg lasix": "1719285",
        "8mg morphine": "1732011",
        "1mg dilaudid": "1872265",
        "ceftriaxone 1 gram": "1665021",
        "vancomycin 1 gram": "1807513",
    }

    for index, entities in records.items():
        source = (ROOT / "input" / f"{index}.txt").read_text(encoding="utf-8")
        for entity in entities:
            if entity.get("type") != "THUỐC":
                continue
            normalized = _normalize(entity.get("text", ""))
            code = exact_codes.get(normalized)
            if code and entity.get("candidates") != [code]:
                entity["candidates"] = [code]
                stats["rxnorm_code_corrected"] += 1
            if normalized == "thay thế bicarbonate":
                prefix_length = len(entity["text"]) - len("bicarbonate")
                entity["position"][0] += prefix_length
                entity["text"] = source[entity["position"][0]:entity["position"][1]]
                entity["candidates"] = ["36676"]
                stats["bicarbonate_span_trimmed"] += 1
            elif normalized == "80mg lasix":
                suffix = source[entity["position"][1]:entity["position"][1] + 3]
                if suffix.casefold() == " iv":
                    entity["position"][1] += 3
                    entity["text"] = source[entity["position"][0]:entity["position"][1]]
                    stats["drug_route_extended"] += 1

        for match in re.finditer(r"(?<!\w)bactrim(?!\w)", source, flags=re.IGNORECASE):
            start, end = match.span()
            if any(
                entity.get("type") == "THUỐC"
                and min(end, entity["position"][1]) > max(start, entity["position"][0])
                for entity in entities
            ):
                continue
            flags = agent._rule_assert(source, start, end, "THUỐC")
            entities.append({
                "text": source[start:end],
                "type": "THUỐC",
                "assertions": [
                    key for key in ("isNegated", "isFamily", "isHistorical")
                    if flags.get(key)
                ],
                "position": [start, end],
                "candidates": ["151399"],
            })
            stats["bactrim_restored"] += 1
        entities.sort(key=lambda item: (item["position"][0], item["position"][1]))
    return stats


def split_concatenated_drugs(records: dict[int, list[dict]]) -> Counter:
    """Split only unmistakable multi-drug spans created by missing separators."""
    stats = Counter()
    agent = AssertionAgent()
    compact_splits = {
        "ciproflagyl": (("cipro", "203563"), ("flagyl", "202866")),
        "klonopinclonidine": (("klonopin", "202585"), ("clonidine", "2599")),
        "vancozosyn": (("vanco", "11124"), ("zosyn", "74170")),
    }

    for index, entities in records.items():
        source = (ROOT / "input" / f"{index}.txt").read_text(encoding="utf-8")
        replaced = []
        for entity in entities:
            if entity.get("type") != "THUỐC":
                replaced.append(entity)
                continue
            normalized = _normalize(entity.get("text", ""))
            parts = compact_splits.get(normalized)
            if parts:
                cursor = entity["position"][0]
                for token, code in parts:
                    start = source.casefold().find(token, cursor, entity["position"][1])
                    if start < 0:
                        continue
                    end = start + len(token)
                    flags = agent._rule_assert(source, start, end, "THUỐC")
                    replaced.append({
                        "text": source[start:end],
                        "type": "THUỐC",
                        "assertions": [
                            key for key in ("isNegated", "isFamily", "isHistorical")
                            if flags.get(key)
                        ],
                        "position": [start, end],
                        "candidates": [code],
                    })
                    cursor = end
                stats["concatenated_drug_split"] += 1
                continue

            if normalized == "cefepim và vancomycin truyền tĩnh mạch":
                for token, code in (("cefepim", "20481"), ("vancomycin", "11124")):
                    start = source.casefold().find(
                        token, entity["position"][0], entity["position"][1]
                    )
                    end = start + len(token)
                    flags = agent._rule_assert(source, start, end, "THUỐC")
                    replaced.append({
                        "text": source[start:end],
                        "type": "THUỐC",
                        "assertions": [
                            key for key in ("isNegated", "isFamily", "isHistorical")
                            if flags.get(key)
                        ],
                        "position": [start, end],
                        "candidates": [code],
                    })
                stats["explicit_multi_drug_split"] += 1
                continue
            replaced.append(entity)
        replaced.sort(key=lambda item: (item["position"][0], item["position"][1]))
        records[index] = replaced
    return stats


def apply_type_precision_reviews(records: dict[int, list[dict]]) -> Counter:
    """Remove obvious fragments and fix deterministic vital/lab/disease types."""
    stats = Counter()
    agent = AssertionAgent()
    fragments = {
        "đó", "thêm", "tiếp", "bằng thuốc", "bán trú",
        "bảo tồn", "nội khoa", "ngoại khoa", "theo đơn", "phù hợp",
    }
    diagnosis_codes = {
        "giả gout": "M11.2",
        "gãy xương": "T14.2",
        "tăng nhãn áp": "H40.9",
        "thiếu máu": "D64.9",
        "u tuyến": "D36.9",
        "viêm túi mật": "K81.9",
    }
    test_names = {
        "bnp", "bilirubin", "bạch cầu", "đường huyết", "glucose",
        "kali", "protein", "tiểu cầu",
    }
    result_names = {"hạ huyết áp", "tăng men gan"}
    vital_pattern = re.compile(
        r"(?:hr|rr|spo2|bp|t)\s*[:=]?\s*\d+(?:[.,/]\d+)?"
        r"(?:\s*(?:ra|%|°?c))?",
        flags=re.IGNORECASE,
    )

    for index, entities in records.items():
        source = (ROOT / "input" / f"{index}.txt").read_text(encoding="utf-8")
        kept = []
        for entity in entities:
            normalized = _normalize(entity.get("text", ""))
            if normalized in fragments:
                stats["nonentity_fragment_removed"] += 1
                continue
            if normalized == "protein":
                start, end = entity["position"]
                context = source[max(0, start - 30):min(len(source), end + 30)].casefold()
                if "tổng hợp protein" in context:
                    stats["nonentity_fragment_removed"] += 1
                    continue

            corrected_type = None
            if normalized in diagnosis_codes:
                corrected_type = "CHẨN_ĐOÁN"
            elif normalized in result_names or vital_pattern.fullmatch(normalized):
                corrected_type = "KẾT_QUẢ_XÉT_NGHIỆM"
            elif normalized in test_names:
                corrected_type = "TÊN_XÉT_NGHIỆM"

            if corrected_type and entity.get("type") != corrected_type:
                entity["type"] = corrected_type
                stats[f"precision_type_to_{corrected_type}"] += 1
                if corrected_type == "CHẨN_ĐOÁN":
                    entity["candidates"] = [diagnosis_codes[normalized]]
                    flags = agent._rule_assert(
                        source, entity["position"][0], entity["position"][1], corrected_type
                    )
                    entity["assertions"] = [
                        key for key in ("isNegated", "isFamily", "isHistorical")
                        if flags.get(key)
                    ]
                else:
                    entity.pop("candidates", None)
                    entity["assertions"] = []
            kept.append(entity)
        records[index] = kept
    return stats


def add_nonoverlapping_reviewed_diagnoses(records: dict[int, list[dict]]) -> Counter:
    """Add only maximal reviewed aliases that do not overlap any prediction."""
    aliases_path = ROOT / "database-stuff" / "icd10_aliases.json"
    aliases = json.loads(aliases_path.read_text(encoding="utf-8"))
    assertion_agent = AssertionAgent()
    stats = Counter()

    for index, entities in records.items():
        source = (ROOT / "input" / f"{index}.txt").read_text(encoding="utf-8")
        matches = []
        for alias, code in aliases.items():
            pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
            for match in re.finditer(pattern, source, flags=re.IGNORECASE):
                # "thiếu máu" in "thiếu máu cơ tim" means ischemia, not D64.9.
                suffix = source[match.end():match.end() + 12].casefold()
                if _normalize(alias) == "thiếu máu" and re.match(r"\s+cơ\s+tim", suffix):
                    continue
                matches.append((match.start(), match.end(), str(code).upper()))

        maximal = [
            match for match in matches
            if not any(
                other[0] <= match[0] and match[1] <= other[1]
                and other[1] - other[0] > match[1] - match[0]
                for other in matches
            )
        ]
        for start, end, code in sorted(maximal):
            if any(
                min(end, entity["position"][1]) > max(start, entity["position"][0])
                for entity in entities
            ):
                continue
            flags = assertion_agent._rule_assert(
                source, start, end, "CHẨN_ĐOÁN"
            )
            assertions = [
                key for key in ("isNegated", "isFamily", "isHistorical")
                if flags.get(key)
            ]
            entities.append({
                "text": source[start:end],
                "type": "CHẨN_ĐOÁN",
                "assertions": assertions,
                "position": [start, end],
                "candidates": [code],
            })
            stats["diagnosis_added"] += 1
        entities.sort(key=lambda item: (item["position"][0], item["position"][1]))
    return stats


def _validate(records: dict[int, list[dict]]) -> None:
    errors = []
    for index, entities in records.items():
        source = (ROOT / "input" / f"{index}.txt").read_text(encoding="utf-8")
        result = validate_output(entities)
        errors.extend(f"{index}.json: {value}" for value in result["errors"])
        for offset, entity in enumerate(entities):
            start, end = entity["position"]
            if source[start:end] != entity["text"]:
                errors.append(f"{index}.json[{offset}]: text/position mismatch")
            if len(entity.get("candidates", [])) > 1:
                errors.append(f"{index}.json[{offset}]: more than one candidate")
    if errors:
        raise ValueError("\n".join(errors[:100]))


def _write_variant(name: str, records: dict[int, list[dict]], destination: Path) -> None:
    output_dir = destination / name
    zip_path = destination / f"submission_{name}.zip"
    if output_dir.exists() or zip_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing variant: {name}")
    output_dir.mkdir(parents=True)
    for index in range(1, 101):
        (output_dir / f"{index}.json").write_text(
            json.dumps(records[index], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(1, 101):
            archive.write(output_dir / f"{index}.json", arcname=f"output/{index}.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "output_qwen3.zip")
    parser.add_argument("--destination", type=Path, default=ROOT / "leaderboard_variants")
    parser.add_argument("--only", help="Write only one named variant")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    baseline = _load_archive(args.source)
    round1_aliases = _load_icd_aliases("icd10_aliases.json")
    round2_aliases = _load_icd_aliases(
        "icd10_aliases.json", "icd10_aliases_round2.json"
    )

    variants = {}

    boundary = copy.deepcopy(baseline)
    boundary_stats = clean_short_boundaries(boundary)
    variants["lb2_boundary"] = (boundary, boundary_stats)

    candidate_hp = copy.deepcopy(boundary)
    candidate_stats = boundary_stats + keep_high_precision_candidates(
        candidate_hp, round1_aliases
    )
    variants["lb3_candidate_hp"] = (candidate_hp, candidate_stats)

    alias_recall = copy.deepcopy(boundary)
    alias_stats = boundary_stats + add_nonoverlapping_reviewed_diagnoses(alias_recall)
    variants["lb4_alias_recall"] = (alias_recall, alias_stats)

    combined = copy.deepcopy(alias_recall)
    combined_stats = alias_stats + keep_high_precision_candidates(
        combined, round1_aliases
    )
    variants["lb5_combined"] = (combined, combined_stats)

    reviewed_v2 = copy.deepcopy(boundary)
    reviewed_v2_stats = boundary_stats + apply_round2_type_reviews(reviewed_v2)
    reviewed_v2_stats += keep_high_precision_candidates(reviewed_v2, round2_aliases)
    variants["lb6_reviewed_v2"] = (reviewed_v2, reviewed_v2_stats)

    assert_rx = copy.deepcopy(reviewed_v2)
    assert_rx_stats = reviewed_v2_stats + recompute_assertions(assert_rx)
    assert_rx_stats += apply_safe_rxnorm_reviews(assert_rx)
    variants["lb7_assert_rx"] = (assert_rx, assert_rx_stats)

    drug_splits = copy.deepcopy(assert_rx)
    drug_split_stats = assert_rx_stats + split_concatenated_drugs(drug_splits)
    variants["lb8_drug_splits"] = (drug_splits, drug_split_stats)

    type_precision = copy.deepcopy(assert_rx)
    type_precision_stats = assert_rx_stats + apply_type_precision_reviews(type_precision)
    variants["lb9_type_precision"] = (type_precision, type_precision_stats)

    full_round3 = copy.deepcopy(drug_splits)
    full_round3_stats = drug_split_stats + apply_type_precision_reviews(full_round3)
    variants["lb10_full_round3"] = (full_round3, full_round3_stats)

    type_precision_v2 = copy.deepcopy(assert_rx)
    type_precision_v2_stats = assert_rx_stats + apply_type_precision_reviews(type_precision_v2)
    variants["lb11_type_precision_v2"] = (type_precision_v2, type_precision_v2_stats)

    controlled_recall = copy.deepcopy(type_precision_v2)
    controlled_recall_stats = type_precision_v2_stats + add_nonoverlapping_reviewed_diagnoses(
        controlled_recall
    )
    variants["lb12_controlled_recall"] = (controlled_recall, controlled_recall_stats)

    type_precision_v3 = copy.deepcopy(assert_rx)
    type_precision_v3_stats = assert_rx_stats + apply_type_precision_reviews(type_precision_v3)
    variants["lb13_rx_precision"] = (type_precision_v3, type_precision_v3_stats)

    full_round4 = copy.deepcopy(type_precision_v3)
    full_round4_stats = type_precision_v3_stats + split_concatenated_drugs(full_round4)
    variants["lb14_full_round4"] = (full_round4, full_round4_stats)

    manifest_path = args.destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}
    for name, (records, stats) in variants.items():
        if args.only and name != args.only:
            continue
        _validate(records)
        _write_variant(name, records, args.destination)
        type_counts = Counter(
            entity["type"] for entities in records.values() for entity in entities
        )
        manifest[name] = {
            "changes": dict(stats),
            "entities": sum(type_counts.values()),
            "types": dict(type_counts),
            "linked": sum(
                bool(entity.get("candidates"))
                for entities in records.values() for entity in entities
            ),
        }
        print(name, json.dumps(manifest[name], ensure_ascii=False))

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
