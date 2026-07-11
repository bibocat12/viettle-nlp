import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))


def preprocess_note(note_id: str, text: str) -> dict:
    """Normalize whitespace/punctuation while preserving exact character offsets."""
    import unicodedata

    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\u00a0", " ")

    sentences = []
    lines = normalized.split("\n")
    offset = 0
    for line in lines:
        stripped = line.strip()
        if stripped:
            start = normalized.find(stripped, offset)
            if start >= 0:
                sentences.append({
                    "text": stripped,
                    "char_start": start,
                    "char_end": start + len(stripped)
                })
                offset = start + len(stripped)
        else:
            offset += len(line) + 1

    sections = []
    section_patterns = [
        (r"(?:^|\n)\s*\d+\.\s*(.+?)(?:\n|$)", "section"),
        (r"(?:^|\n)\s*((?:Tiền sử|Bệnh sử|Kết quả|Đánh giá|Xử trí).+?)(?:\n|$)", "section"),
    ]
    for pattern, stype in section_patterns:
        for m in re.finditer(pattern, normalized):
            sections.append({
                "name": m.group(1).strip() if m.lastindex and m.lastindex >= 1 else stype,
                "char_start": m.start(),
                "char_end": m.end()
            })

    return {
        "note_id": note_id,
        "original_text": text,
        "normalized_text": normalized,
        "sentences": sentences,
        "sections": sections,
    }


NEGATION_TRIGGERS = [
    r"không\b", r"chưa\b", r"không có\b", r"chưa có\b", r"không ghi nhận\b",
    r"không có gì\b", r"không có triệu chứng\b", r"không có bất thường\b",
    r"không có gì đáng\b", r"không buồn nôn\b", r"không đánh trống ngực\b",
    r"không quầng sáng\b", r"không chóng mặt\b", r"không có biểu hiện\b",
    r"không có quá trình\b", r"không phàn nàn\b", r"không có gì bất thường\b",
    r"không có gì đáng chú ý\b", r"không sốt\b", r"không ho\b",
]

FAMILY_TRIGGERS = [
    r"bố\b", r"mẹ\b", r"cha\b", r"anh\b", r"chị\b", r"em\b",
    r"ông\b", r"bà\b", r"người nhà\b", r"gia đình\b", r"gia đình có\b",
    r"tiền sử gia đình\b", r"theo lời\b",
]

HISTORICAL_TRIGGERS = [
    r"tiền sử\b", r"trước đây\b", r"trước khi\b", r"trước nhập viện\b",
    r"trước đó\b", r"đã điều trị\b", r"đã dùng\b", r"đã chỉ định\b",
    r"đã sử dụng\b", r"được chỉ định\b", r"các bệnh đã điều trị\b",
    r"các bệnh lý mãn tính\b", r"bệnh lý nội khoa mạn\b", r"tái phát\b",
]

DRUG_PATTERN = re.compile(
    r"(?P<name>[A-Za-zÀ-ỹ]{3,25})"
    r"\s+(?P<dose>\d+(?:\.\d+)?)\s*(?P<unit>mg|ml|mcg|IU|U)\b"
    r"(?:\s+(?P<route>po|iv|im|sc|sl|tiêm\s+tĩnh\s+mạch|tiêm|uống|đặt\s+dưới\s+lưỡi|đặt|nhỏ|bôi|thoa))?"
    r"(?:\s+(?P<freq>bid|tid|qd|qid|prn|x\s*\d+))?",
    re.IGNORECASE
)

TEST_PATTERNS = [
    r"điện tâm đồ", r"ecg", r"ECG", r"chụp x-quang", r"x-quang",
    r"siêu âm tim qua thành ngực", r"siêu âm tim", r"siêu âm",
    r"chụp cắt lớp vi tính", r"CT", r"MRI", r"mri",
    r"monitor holter", r"holter",
    r"phân tích nước tiểu", r"nước tiểu",
    r"bảng công thức sinh hóa máu", r"bảng công thức máu",
    r"bảng chức năng gan", r"bản phân tích nước tiểu",
    r"troponin", r"creatinine", r"canxi", r"bilirubin",
    r"hemoglobin", r"Hb", r"hematocrit", r"Hct",
    r"tiểu cầu", r"bạch cầu", r"hồng cầu", r"đường huyết",
    r"glucose", r"BUN", r"ure", r"sodium", r"potassium", r"chloride",
    r"protein", r"albumin", r"canxi toàn phần", r"canxi ion hóa",
    # Medical procedures (removed — not test names)
]

LAB_VALUE_PATTERNS = [
    (r"(troponin)\s*(\d+\.?\d*)", "troponin"),
    (r"(canxi\s*(?:toàn phần)?)\s*(?:là|:)?\s*(\d+\.?\d*)", "canxi"),
    (r"(canxi\s*ion hóa)\s*(?:là|:)?\s*(\d+\.?\d*)", "canxi ion hóa"),
    (r"(creatinine|cr)\s*(\d+\.?\d*)", "creatinine"),
    (r"(Na|sodium)\s*(\d+\.?\d*)", "sodium"),
    (r"(K|potassium)\s*(\d+\.?\d*)", "potassium"),
    (r"(Hb|hemoglobin)\s*(\d+\.?\d*)", "hemoglobin"),
    (r"(Hct|hematocrit)\s*(\d+\.?\d*)", "hematocrit"),
    (r"(WBC|bạch cầu)\s*(\d+\.?\d*)", "WBC"),
    (r"(RBC|hồng cầu)\s*(\d+\.?\d*)", "RBC"),
    (r"(platelet|tiểu cầu)\s*(\d+\.?\d*)", "platelet"),
    (r"(glucose|đường huyết)\s*(?:là|:)?\s*(\d+\.?\d*)", "glucose"),
    (r"(bilirubin)\s*(\d+\.?\d*)", "bilirubin"),
]

NORMAL_RESULTS = [
    r"bình thường", r"không ghi nhận gì bất thường",
    r"không có gì đáng chú ý", r"không có gì bất thường",
    r"không có bất thường", r"không có quá trình cấp tính",
]


def _non_overlapping(matches: list) -> list:
    if not matches:
        return []
    matches.sort(key=lambda m: m["position"][0])
    result = [matches[0]]
    for m in matches[1:]:
        prev = result[-1]
        if m["position"][0] >= prev["position"][1]:
            result.append(m)
        else:
            cur_len = m["position"][1] - m["position"][0]
            prev_len = prev["position"][1] - prev["position"][0]
            if cur_len > prev_len:
                result[-1] = m
    return result


def rule_extract(text: str) -> dict:
    """Extract mentions using regex rules and detect assertion cues."""
    mentions = []
    assertion_cues = []

    # Drugs
    for m in DRUG_PATTERN.finditer(text):
        start, end = m.start(), m.end()
        name = m.group("name").strip()
        if len(name) < 3:
            continue
        dose = m.group("dose") or ""
        unit = m.group("unit") or ""
        drug_text = f"{name} {dose}{unit}".strip()
        mentions.append({
            "text": drug_text,
            "position": [start, end],
            "type": "THUỐC",
            "confidence": 0.98,
            "source": "rule",
        })

    # Test names
    for pattern in TEST_PATTERNS:
        for m in re.finditer(re.escape(pattern), text, re.IGNORECASE):
            start, end = m.start(), m.end()
            if end - start >= 2:
                mentions.append({
                    "text": text[start:end],
                    "position": [start, end],
                    "type": "TÊN_XÉT_NGHIỆM",
                    "confidence": 0.95,
                    "source": "rule",
                })

    # Lab values
    for pattern, test_name in LAB_VALUE_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            groups = m.groups()
            if len(groups) >= 2:
                value_text = groups[-1]
                last_idx = m.lastindex if m.lastindex is not None else len(groups)
                value_start = m.start(last_idx)
                value_end = m.end(last_idx)
                mentions.append({
                    "text": value_text,
                    "position": [value_start, value_end],
                    "type": "KẾT_QUẢ_XÉT_NGHIỆM",
                    "confidence": 0.95,
                    "source": "rule",
                })

    # Normal results
    for pattern in NORMAL_RESULTS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            mentions.append({
                "text": text[m.start():m.end()],
                "position": [m.start(), m.end()],
                "type": "KẾT_QUẢ_XÉT_NGHIỆM",
                "confidence": 0.90,
                "source": "rule",
            })

    mentions = _non_overlapping(mentions)

    # Assertion cues
    for m in re.finditer(r"không\b|chưa\b|không có\b|tiền sử\b|trước đây\b|bố\b|mẹ\b|người nhà\b|gia đình\b", text, re.IGNORECASE):
        cue = m.group()
        pos = [m.start(), m.end()]
        cue_lower = cue.lower()
        if any(re.match(p, cue_lower) for p in NEGATION_TRIGGERS):
            assertion_cues.append({"cue": cue, "position": pos, "label_hint": "isNegated"})
        if any(re.match(p, cue_lower) for p in FAMILY_TRIGGERS):
            assertion_cues.append({"cue": cue, "position": pos, "label_hint": "isFamily"})
        if any(re.match(p, cue_lower) for p in HISTORICAL_TRIGGERS):
            assertion_cues.append({"cue": cue, "position": pos, "label_hint": "isHistorical"})

    return {"mentions": mentions, "assertion_cues": assertion_cues}


def validate_output(predictions: list) -> dict:
    """Hard schema validator - no LLM, pure validation."""
    valid_types = {"THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}
    valid_assertions = {"isNegated", "isFamily", "isHistorical"}
    errors = []
    warnings = []

    for i, pred in enumerate(predictions):
        if "text" not in pred:
            errors.append(f"[{i}] missing 'text'")
        if "position" not in pred:
            errors.append(f"[{i}] missing 'position'")
        elif not isinstance(pred["position"], list) or len(pred["position"]) != 2:
            errors.append(f"[{i}] invalid 'position' format")
        elif pred["position"][1] <= pred["position"][0]:
            errors.append(f"[{i}] position end <= start")
        if pred.get("type") not in valid_types:
            errors.append(f"[{i}] invalid type: {pred.get('type')}")
        assertions = pred.get("assertions", {})
        for key in assertions:
            if key not in valid_assertions:
                errors.append(f"[{i}] invalid assertion key: {key}")
        if pred.get("type") in ("CHẨN_ĐOÁN", "THUỐC"):
            if not pred.get("candidates"):
                warnings.append(f"[{i}] {pred.get('type')} has no candidates")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
