#!/usr/bin/env python3
"""Build leakage-safe, competition-schema GRPO v2 datasets.

The builder never reads leaderboard inputs. It cleans the existing training
corpus, removes the unsupported THỦ_THUẬT label, adds conservative assertion
labels, validates offsets, and supervises candidates only when a reviewed
alias or KB semantic match makes the code trustworthy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
VALID_TYPES = {
    "THUỐC",
    "TRIỆU_CHỨNG",
    "CHẨN_ĐOÁN",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
}
VALID_ASSERTIONS = {"isNegated", "isFamily", "isHistorical"}

SYSTEM_PROMPT = """Bạn là hệ thống trích xuất thực thể y khoa tiếng Việt.

Chỉ sử dụng đúng 5 loại:
- THUỐC: tên thuốc, có thể gồm hàm lượng, dạng, đường dùng và tần suất.
- TRIỆU_CHỨNG: triệu chứng, dấu hiệu hoặc than phiền của bệnh nhân.
- CHẨN_ĐOÁN: bệnh, hội chứng hoặc chẩn đoán.
- TÊN_XÉT_NGHIỆM: tên xét nghiệm hoặc thủ thuật chẩn đoán.
- KẾT_QUẢ_XÉT_NGHIỆM: kết quả, chỉ số hoặc nhận xét của xét nghiệm.

Mỗi thực thể phải có:
- entity: chuỗi nguyên văn, khớp chính xác văn bản đầu vào.
- type: một trong 5 loại trên.
- position: [start, end] theo chỉ số ký tự, end là vị trí ngay sau ký tự cuối.
- assertions: chỉ gồm isNegated, isFamily, isHistorical; không có thì [].
- candidates: tối đa một mã ICD-10 cho CHẨN_ĐOÁN hoặc RxNorm cho THUỐC; loại khác dùng [].

Không trích xuất tiêu đề mục, metadata, thủ thuật điều trị/phẫu thuật hoặc cụm từ phi y khoa.
Giữ nguyên viết hoa, dấu tiếng Việt và ranh giới đầy đủ của thực thể.
Chỉ trả về một mảng JSON, không giải thích."""

ICD_OVERRIDES = {
    "benh dai": "A82.9",
    "may day vo can": "L50.1",
    "may day man": "L50.8",
    "may day man tinh": "L50.8",
    "tieu duong": "E11.9",
    "gian thung tinh": "I86.1",
    "nang mang nhen": "G93.0",
    "tram cam": "F32.9",
    "mun trung ca": "L70.9",
}

STOPWORDS = {
    "benh", "do", "cua", "va", "hoac", "khong", "xac", "dinh", "khac",
    "tinh", "trang", "kem", "voi", "tai", "vi", "tri", "loai", "thuoc",
}


def fold(value: str) -> str:
    value = value.casefold().replace("đ", "d")
    value = unicodedata.normalize("NFD", value)
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value).casefold()).strip()


def semantic_score(left: str, right: str) -> float:
    a = [token for token in fold(left).split() if token not in STOPWORDS]
    b = [token for token in fold(right).split() if token not in STOPWORDS]
    if not a or not b:
        return 0.0
    aset, bset = set(a), set(b)
    dice = 2 * len(aset & bset) / (len(aset) + len(bset))
    ratio = SequenceMatcher(None, " ".join(a), " ".join(b)).ratio()
    containment = 1.0 if " ".join(a) in " ".join(b) or " ".join(b) in " ".join(a) else 0.0
    return 0.55 * dice + 0.35 * ratio + 0.10 * containment


def load_json_aliases(names: tuple[str, ...]) -> dict[str, str]:
    values: dict[str, str] = {}
    for name in names:
        path = ROOT / "database-stuff" / name
        if path.exists():
            values.update(json.loads(path.read_text(encoding="utf-8")))
    return {fold(name): str(code) for name, code in values.items()}


def load_knowledge() -> tuple[dict, dict, dict, dict]:
    icd_aliases = load_json_aliases(("icd10_aliases.json", "icd10_aliases_round2.json"))
    icd_aliases.update(ICD_OVERRIDES)
    rx_aliases = load_json_aliases(("rxnorm_aliases.json", "rxnorm_aliases_round2.json"))

    icd_names: dict[str, str] = {}
    with (ROOT / "database-stuff" / "icd10_vietnamese.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            code = row.get("icd10_code", "").strip().upper()
            if code:
                icd_names[code] = " ".join(
                    value for value in (row.get("vietnamese_name", ""), row.get("english_name", "")) if value
                )

    rx_names: dict[str, str] = defaultdict(str)
    with (ROOT / "database-stuff" / "rxnorm_concepts.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            code = row.get("rxcui", "").strip()
            if code:
                rx_names[code] = " ".join(
                    value for value in (row.get("prefLabel", ""), row.get("altLabels", ""), row.get("strength", "")) if value
                )
    return icd_aliases, rx_aliases, icd_names, dict(rx_names)


def relocate(text: str, entity: str, anchor: int) -> tuple[int, int] | None:
    matches = [match.span() for match in re.finditer(re.escape(entity), text)]
    if not matches:
        matches = [match.span() for match in re.finditer(re.escape(entity), text, re.IGNORECASE)]
    return min(matches, key=lambda span: abs(span[0] - anchor)) if matches else None


def assertions_for(text: str, start: int, entity_type: str) -> list[str]:
    if entity_type not in {"THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN"}:
        return []
    prefix = unicodedata.normalize("NFC", text[max(0, start - 180):start]).casefold()
    boundary = max(prefix.rfind(token) for token in ("\n", ".", ";", "!", "?"))
    clause = prefix[boundary + 1:] if boundary >= 0 else prefix
    contrasts = list(re.finditer(r"\b(?:nhưng|tuy nhiên|song)\b", clause))
    if contrasts:
        clause = clause[contrasts[-1].end():]

    labels = []
    negated = re.search(
        r"(?:\bkhông(?:\s+có|\s+ghi nhận|\s+còn)?|\bchưa(?:\s+có)?|\bphủ nhận)\b"
        r"\s*$",
        clause,
        re.DOTALL,
    )
    negation_blocker = re.search(
        r"\b(?:điều trị|phòng ngừa|nguy cơ|làm giảm|gây ra|thực hiện|sử dụng)\b[^.!?;\n]*$",
        clause,
    )
    if negated and not negation_blocker:
        labels.append("isNegated")
    if re.search(
        r"\btiền sử gia đình(?:\s+(?:bị|mắc|có))?\s*$", clause, re.DOTALL
    ) or re.search(
        r"\b(?:bố|mẹ|cha|chị gái|anh trai|ông|bà)\b"
        r"(?:\s+của)?(?:\s+(?:bệnh nhân|người bệnh|tôi|em|bé))?.{0,25}"
        r"\b(?:bị|mắc|được chẩn đoán)\b.{0,35}$",
        clause,
        re.DOTALL,
    ):
        labels.append("isFamily")
    if re.search(
        r"(?:\btiền sử(?:\s+(?:bị|mắc|có))?|\bđã từng(?:\s+(?:bị|mắc|có))?|"
        r"\btrước đây(?:\s+đã)?|\bđã dùng|\btrước nhập viện|\btrước khi nhập viện)\s*$",
        clause,
        re.DOTALL,
    ):
        labels.append("isHistorical")
    return labels


DIAGNOSIS_EXACT = {
    "hiv", "hiv 1", "hiv 2", "hiv aids", "aids", "lao", "lao phoi",
    "lao mang phoi", "lao ngoai phoi", "lao hach", "lao xuong",
    "tieu duong", "dai thao duong", "tang huyet ap", "cao huyet ap",
    "rung nhi", "tram cam", "beo phi", "thieu mau", "nhip tim cham",
    "roi loan nhip tim", "nhoi mau co tim", "dot quy", "sot ret",
    "cuong giap", "tu ky", "cum", "rubella", "sui mao ga", "giang mai",
    "bach hau", "parkinson", "benh gout", "benh gut", "nao ung thuy",
}

SYMPTOM_EXACT = {
    "tieu chay", "tao bon", "mat ngu", "u tai", "di ung", "ho khan",
    "kho nuot", "chay mau", "va mo hoi", "hoi hop", "ha huyet ap",
    "huyet ap thap", "huyet ap cao", "dau that nguc", "suy nhuoc",
}


def corrected_type(value: str, entity_type: str, icd_aliases: dict, rx_aliases: dict) -> str:
    name = fold(value)
    if name in rx_aliases:
        return "THUỐC"
    if name in icd_aliases or name in DIAGNOSIS_EXACT:
        return "CHẨN_ĐOÁN"
    if name in SYMPTOM_EXACT:
        return "TRIỆU_CHỨNG"
    if name.startswith("dau ") or name in {"dau", "dau nhuc"}:
        return "TRIỆU_CHỨNG"
    if re.search(
        r"^(?:benh\b|viem\b|ung thu\b|hoi chung\b|lao(?:\b|_)|nhiem trung\b|"
        r"nhiem khuan\b|xo gan\b|xuat huyet nao\b|ap xe\b|thoat vi\b|u nang\b|"
        r"nhoi mau\b|roi loan\b|suy tim\b|suy ho hap\b|thuyen tac\b)",
        name,
    ):
        return "CHẨN_ĐOÁN"
    if re.search(r"(?:xet nghiem|sieu am|x quang|noi soi|dien tam do|ecg|mri|ct)\b", name):
        return "TÊN_XÉT_NGHIỆM"
    return entity_type


def low_quality_entity(value: str, entity_type: str, has_candidate: bool) -> bool:
    name = fold(value)
    tokens = name.split()
    if len(value.strip()) < 2 or not name:
        return True
    if name in {
        "benh nhan", "trieu chung", "dieu tri", "thuoc", "benh", "tinh trang",
        "phu hop", "hieu qua", "kip thoi", "tiep", "do", "them",
    }:
        return True
    if not has_candidate and (len(value) > 60 or len(tokens) > 9):
        return True
    if entity_type == "TRIỆU_CHỨNG" and re.search(
        r"\b(?:co the|gay ra|duoc ket hop|bao gom|mot nhom|tiep xuc voi|"
        r"neu khong|de giai phong|lam suy yeu|anh huong den)\b",
        name,
    ):
        return True
    return False


def trusted_candidates(
    entity: dict,
    icd_aliases: dict,
    rx_aliases: dict,
    icd_names: dict,
    rx_names: dict,
) -> tuple[list[str], bool, str]:
    entity_type = entity["type"]
    if entity_type not in {"CHẨN_ĐOÁN", "THUỐC"}:
        return [], True, "not_linkable"

    name = fold(entity["entity"])
    aliases = icd_aliases if entity_type == "CHẨN_ĐOÁN" else rx_aliases
    if name in aliases:
        return [aliases[name]], True, "reviewed_alias"

    existing = entity.get("candidates", [])
    if not isinstance(existing, list) or len(existing) != 1:
        return [], False, "missing"
    code = str(existing[0]).strip().upper() if entity_type == "CHẨN_ĐOÁN" else str(existing[0]).strip()
    if entity_type == "CHẨN_ĐOÁN":
        if not re.fullmatch(r"[A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?", code):
            return [], False, "invalid_code"
        label = icd_names.get(code, "")
        threshold = 0.52
    else:
        if not code.isdigit():
            return [], False, "invalid_code"
        label = rx_names.get(code, "")
        threshold = 0.42
    score = semantic_score(entity["entity"], label)
    if label and score >= threshold:
        return [code], True, "kb_semantic"
    return [], False, "semantic_reject"


def remove_candidate_leakage(text: str, codes: set[str]) -> str:
    cleaned = text
    for code in sorted(codes, key=len, reverse=True):
        escaped = re.escape(code)
        cleaned = re.sub(rf"\s*\(\s*{escaped}\s*\)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            rf"\s*(?:với\s+)?mã\s+ICD-?10\s+{escaped}\b[.,:]?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
    return re.sub(r" {2,}", " ", cleaned)


def clean_row(raw: dict, line_number: int, knowledge: tuple[dict, dict, dict, dict]) -> dict | None:
    messages = raw.get("messages", [])
    if len(messages) < 2 or not isinstance(messages[1].get("content"), str):
        return None
    original_text = unicodedata.normalize("NFC", messages[1]["content"])
    cleaned_entities = []
    icd_aliases, rx_aliases, _, _ = knowledge
    dropped_quality = 0
    type_repairs = 0
    for entity in raw.get("entities", []):
        if not isinstance(entity, dict) or entity.get("type") not in VALID_TYPES:
            continue
        value = entity.get("entity") or entity.get("text") or ""
        position = entity.get("position", [0, 0])
        if not value or not isinstance(position, list) or len(position) != 2:
            continue
        value = unicodedata.normalize("NFC", value)
        start, end = position
        if not (isinstance(start, int) and isinstance(end, int)):
            continue
        if not (0 <= start < end <= len(original_text) and original_text[start:end] == value):
            relocated = relocate(original_text, value, start)
            if relocated is None:
                continue
            start, end = relocated
        repaired_type = corrected_type(value, entity["type"], icd_aliases, rx_aliases)
        type_repairs += repaired_type != entity["type"]
        if low_quality_entity(value, repaired_type, bool(entity.get("candidates"))):
            dropped_quality += 1
            continue
        candidate_entity = {
            "entity": original_text[start:end],
            "type": repaired_type,
            "position": [start, end],
            "candidates": entity.get("candidates", []),
        }
        candidates, supervised, source = trusted_candidates(candidate_entity, *knowledge)
        candidate_entity["candidates"] = candidates
        candidate_entity["_candidate_supervised"] = supervised
        candidate_entity["_candidate_source"] = source
        cleaned_entities.append(candidate_entity)

    # Remove candidate codes shown verbatim in synthetic prompts. The model
    # must learn linking rather than copy a parenthesized answer.
    trusted_codes = {
        entity["candidates"][0]
        for entity in cleaned_entities
        if entity["_candidate_supervised"] and entity["candidates"]
    }
    text = remove_candidate_leakage(original_text, trusted_codes)

    final_entities = []
    seen = set()
    for entity in sorted(cleaned_entities, key=lambda item: item["position"]):
        value = entity["entity"]
        old_start = entity["position"][0]
        relocated = relocate(text, value, old_start)
        if relocated is None:
            continue
        start, end = relocated
        marker = (start, end, entity["type"])
        if marker in seen:
            continue
        seen.add(marker)
        assertions = assertions_for(text, start, entity["type"])
        final_entities.append({
            "entity": text[start:end],
            "type": entity["type"],
            "position": [start, end],
            "assertions": assertions,
            "candidates": entity["candidates"],
            "_assertion_supervised": True,
            "_candidate_supervised": entity["_candidate_supervised"],
            "_candidate_source": entity["_candidate_source"],
        })

    public_entities = [
        {key: value for key, value in entity.items() if not key.startswith("_")}
        for entity in final_entities
    ]
    if not final_entities:
        return None
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": json.dumps(public_entities, ensure_ascii=False)},
        ],
        "entities": final_entities,
        "metadata": {
            "schema_version": "grpo_v2",
            "source": "grpo_combined_cleaned",
            "source_line": line_number,
            "type_repairs": type_repairs,
            "quality_entities_dropped": dropped_quality,
            "candidate_fully_supervised": all(
                entity["_candidate_supervised"] for entity in final_entities
                if entity["type"] in {"CHẨN_ĐOÁN", "THUỐC"}
            ),
        },
    }


def split_name(text: str) -> str:
    digest = int(hashlib.sha256(compact(text).encode("utf-8")).hexdigest()[:8], 16) % 1000
    if digest < 900:
        return "train"
    if digest < 950:
        return "validation"
    return "test"


def build_assertion_synthetic(rows_by_split: dict[str, list]) -> dict[str, list]:
    """Create balanced high-precision assertion examples without test leakage."""
    pools: dict[str, dict[str, dict]] = {
        entity_type: {} for entity_type in ("CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "THUỐC")
    }
    for rows in rows_by_split.values():
        for row in rows:
            for entity in row["entities"]:
                entity_type = entity["type"]
                if entity_type not in pools:
                    continue
                name = entity["entity"].strip()
                if not (2 <= len(name) <= 40) or low_quality_entity(
                    name, entity_type, bool(entity.get("candidates"))
                ):
                    continue
                pools[entity_type].setdefault(fold(name), entity)

    selected = {}
    for entity_type, values in pools.items():
        ordered = sorted(
            values.values(),
            key=lambda entity: hashlib.sha256(
                f"{entity_type}\0{fold(entity['entity'])}".encode("utf-8")
            ).hexdigest(),
        )
        selected[entity_type] = ordered[:500]

    templates = {
        "present": {
            "CHẨN_ĐOÁN": ("Bệnh nhân hiện được chẩn đoán {entity}.", []),
            "TRIỆU_CHỨNG": ("Bệnh nhân hiện có {entity}.", []),
            "THUỐC": ("Bệnh nhân đang dùng {entity}.", []),
        },
        "negated": {
            "CHẨN_ĐOÁN": ("Bệnh nhân không có {entity}.", ["isNegated"]),
            "TRIỆU_CHỨNG": ("Không ghi nhận {entity} ở bệnh nhân.", ["isNegated"]),
        },
        "historical": {
            "CHẨN_ĐOÁN": ("Bệnh nhân có tiền sử {entity}.", ["isHistorical"]),
            "TRIỆU_CHỨNG": ("Trước đây bệnh nhân đã từng có {entity}.", ["isHistorical"]),
            "THUỐC": ("Bệnh nhân đã dùng {entity} trước nhập viện.", ["isHistorical"]),
        },
        "family": {
            "CHẨN_ĐOÁN": ("Mẹ của bệnh nhân mắc {entity}.", ["isFamily"]),
        },
    }
    output = {name: [] for name in ("train", "validation", "test")}
    for entity_type, entities in selected.items():
        for source in entities:
            group_key = f"{entity_type}\0{fold(source['entity'])}"
            split = split_name(group_key)
            for state, by_type in templates.items():
                if entity_type not in by_type:
                    continue
                template, assertions = by_type[entity_type]
                text = template.format(entity=source["entity"])
                start = text.index(source["entity"])
                candidate_supervised = bool(source.get("_candidate_supervised", False))
                candidates = list(source.get("candidates", [])) if candidate_supervised else []
                internal = {
                    "entity": text[start:start + len(source["entity"])],
                    "type": entity_type,
                    "position": [start, start + len(source["entity"])],
                    "assertions": list(assertions),
                    "candidates": candidates,
                    "_assertion_supervised": True,
                    "_candidate_supervised": candidate_supervised,
                    "_candidate_source": source.get("_candidate_source", "synthetic_masked"),
                }
                public = {key: value for key, value in internal.items() if not key.startswith("_")}
                output[split].append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": json.dumps([public], ensure_ascii=False)},
                    ],
                    "entities": [internal],
                    "metadata": {
                        "schema_version": "grpo_v2",
                        "source": "assertion_synthetic_v2",
                        "assertion_state": state,
                        "candidate_fully_supervised": candidate_supervised,
                    },
                })
    return output


def validate_row(row: dict) -> list[str]:
    errors = []
    text = row["messages"][1]["content"]
    public = json.loads(row["messages"][2]["content"])
    if len(public) != len(row["entities"]):
        errors.append("assistant/entity length mismatch")
    for index, entity in enumerate(row["entities"]):
        if entity["type"] not in VALID_TYPES:
            errors.append(f"entity {index}: invalid type")
        start, end = entity["position"]
        if text[start:end] != entity["entity"]:
            errors.append(f"entity {index}: offset mismatch")
        if not set(entity["assertions"]) <= VALID_ASSERTIONS:
            errors.append(f"entity {index}: invalid assertion")
        if len(entity["candidates"]) > 1:
            errors.append(f"entity {index}: too many candidates")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data_train" / "grpo_combined.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data_train" / "grpo_v2")
    args = parser.parse_args()

    knowledge = load_knowledge()
    rows_by_split: dict[str, list] = {name: [] for name in ("train", "validation", "test")}
    full_by_split: dict[str, list] = {name: [] for name in ("train", "validation", "test")}
    linking_by_split: dict[str, list] = {name: [] for name in ("train", "validation", "test")}
    assertion_by_split: dict[str, list] = {name: [] for name in ("train", "validation", "test")}
    seen_texts = set()
    stats = Counter()

    with args.input.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid_json"] += 1
                continue
            row = clean_row(raw, line_number, knowledge)
            if row is None:
                stats["invalid_row"] += 1
                continue
            text_key = compact(row["messages"][1]["content"])
            if text_key in seen_texts:
                stats["duplicates_removed"] += 1
                continue
            seen_texts.add(text_key)
            errors = validate_row(row)
            if errors:
                stats["validation_rejected"] += 1
                continue
            split = split_name(text_key)
            rows_by_split[split].append(row)
            if row["metadata"]["candidate_fully_supervised"]:
                full_by_split[split].append(row)
                if any(entity["candidates"] for entity in row["entities"]):
                    linking_by_split[split].append(row)
            if any(entity["assertions"] for entity in row["entities"]):
                assertion_by_split[split].append(row)
            stats["rows_clean"] += 1
            stats[f"split_{split}"] += 1
            stats["entities"] += len(row["entities"])
            stats["type_repairs"] += row["metadata"]["type_repairs"]
            stats["quality_entities_dropped"] += row["metadata"]["quality_entities_dropped"]
            for entity in row["entities"]:
                stats[f"type_{entity['type']}"] += 1
                stats["assertion_positive"] += bool(entity["assertions"])
                stats["candidate_supervised"] += entity["_candidate_supervised"]
                stats["candidate_positive"] += bool(entity["candidates"])
                stats[f"candidate_source_{entity['_candidate_source']}"] += 1

    args.output.mkdir(parents=True, exist_ok=True)
    assertion_synthetic = build_assertion_synthetic(rows_by_split)
    files = {}
    for family, groups in (
        ("extraction", rows_by_split),
        ("full", full_by_split),
        ("linking", linking_by_split),
        ("assertion", assertion_by_split),
        ("assertion_synthetic", assertion_synthetic),
    ):
        for split, rows in groups.items():
            path = args.output / f"{family}_{split}.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files[path.name] = {"rows": len(rows), "bytes": path.stat().st_size, "sha256": digest}

    manifest = {
        "schema_version": "grpo_v2",
        "source": str(args.input),
        "leaderboard_inputs_used": False,
        "split_policy": {
            "natural_corpora": "sha256(normalized_text): 90% train / 5% validation / 5% test",
            "assertion_synthetic": "sha256(type + normalized_entity): 90% train / 5% validation / 5% test",
        },
        "stats": dict(sorted(stats.items())),
        "files": files,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
