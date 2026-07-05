import re
from typing import List, Dict, Tuple

NEGATION_TRIGGERS = [
    r"không\b", r"chưa\b", r"không có\b", r"chưa có\b", r"không ghi nhận\b",
    r"không có gì\b", r"không có triệu chứng\b",
    r"không có bất thường\b", r"không có gì đáng\b", r"không có cải thiện\b",
    r"không liên quan\b", r"không cảm giác\b", r"không buồn nôn\b",
    r"không đổ mồ hôi\b", r"không mất kiểm soát\b", r"không có khó chịu\b",
    r"không đánh trống ngực\b", r"không quầng sáng\b", r"không chóng mặt\b",
    r"không có biểu hiện\b", r"không có quá trình\b", r"không phàn nàn\b",
    r"không có gì bất thường\b", r"không có gì đáng chú ý\b",
    r"không sốt\b", r"không ho\b",
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
    r"các bệnh lý mãn tính\b", r"bệnh lý nội khoa mạn\b",
    r"các tập kinh\b", r"tái phát\b",
]

DRUG_PATTERN = re.compile(
    r"(?P<name>[A-Za-zÀ-ỹ]{3,25})"
    r"\s+(?P<dose>\d+(?:\.\d+)?)\s*(?P<unit>mg|ml|mcg|IU|U)\b"
    r"(?:\s+(?P<route>po|iv|im|sc|sl|tiêm\s+tĩnh\s+mạch|tiêm|uống|đặt\s+dưới\s+lưỡi|đặt|nhỏ|bôi|thoa))?"
    r"(?:\s+(?P<freq>bid|tid|qd|qid|prn|x\s*\d+))?",
    re.IGNORECASE
)

TEST_NAME_PATTERNS = [
    r"điện tâm đồ", r"ecg", r"ECG",
    r"chụp x-quang", r"x-quang", r"X-quang",
    r"siêu âm tim qua thành ngực", r"siêu âm",
    r"chụp cắt lớp vi tính", r"CT",
    r"MRI", r"mri",
    r"monitor holter", r"holter",
    r"phân tích nước tiểu", r"nước tiểu",
    r"bảng công thức sinh hóa máu", r"bảng công thức máu", r"bảng chức năng gan",
    r"bản phân tích nước tiểu",
    r"troponin", r"creatinine", r"canxi", r"bilirubin",
    r"hemoglobin", r"Hb", r"hematocrit", r"Hct",
    r"tiểu cầu", r"bạch cầu", r"hồng cầu",
    r"đường huyết", r"glucose", r"BUN", r"ure",
    r"sodium", r"potassium", r"chloride",
    r"protein", r"albumin",
    r"canxi toàn phần", r"canxi ion hóa",
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
    (r"(protein)\s*(\d+\.?\d*)", "protein"),
    (r"(albumin)\s*(\d+\.?\d*)", "albumin"),
]

NORMAL_RESULTS = [
    r"bình thường",
    r"không ghi nhận gì bất thường",
    r"không có gì đáng chú ý",
    r"không có gì bất thường",
    r"không có bất thường",
    r"không có quá trình cấp tính",
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


def extract_drugs(text: str) -> List[Dict]:
    raw = []
    for m in DRUG_PATTERN.finditer(text):
        start, end = m.start(), m.end()
        name = m.group("name").strip() if m.group("name") else ""
        if len(name) < 3:
            continue
        dose = m.group("dose") or ""
        unit = m.group("unit") or ""
        route = m.group("route") or ""
        freq = m.group("freq") or ""
        drug_text = f"{name} {dose}{unit}".strip()
        if route:
            drug_text += f" {route}"
        if freq:
            drug_text += f" {freq}"
        raw.append({
            "text": drug_text,
            "position": [start, end],
            "type": "THUỐC",
            "confidence": 0.9,
            "source": "rule",
        })
    return _non_overlapping(raw)


def extract_test_names(text: str) -> List[Dict]:
    mentions = []
    for pattern in TEST_NAME_PATTERNS:
        for m in re.finditer(re.escape(pattern), text, re.IGNORECASE):
            start, end = m.start(), m.end()
            mention_text = text[start:end].strip()
            if len(mention_text) < 2:
                continue
            mentions.append({
                "text": mention_text,
                "position": [start, end],
                "type": "TÊN_XÉT_NGHIỆM",
                "confidence": 0.9,
                "source": "rule",
            })
    return _non_overlapping(mentions)


def extract_lab_values(text: str) -> List[Dict]:
    mentions = []
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
                    "confidence": 0.9,
                    "source": "rule",
                })
    for pattern in NORMAL_RESULTS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            start, end = m.start(), m.end()
            mention_text = text[start:end].strip()
            mentions.append({
                "text": mention_text,
                "position": [start, end],
                "type": "KẾT_QUẢ_XÉT_NGHIỆM",
                "confidence": 0.85,
                "source": "rule",
            })
    return _non_overlapping(mentions)


DIAGNOSIS_PATTERNS = [
    r"(?:chẩn đoán|chẩn đoán chính)\s*[:\-]?\s*(.+?)(?:\n|$)",
    r"(?:bệnh chính|bệnh nền)\s*[:\-]?\s*(.+?)(?:\n|$)",
    r"(?:Lý do nhập viện)\s*[:\-]?\s*(.+?)(?:\n|$)",
]

SYMPTOM_PATTERNS = [
    r"(?:triệu chứng hiện tại|các triệu chứng)\s*[:\-]?\s*(.+?)(?:\n|$)",
    r"(?:đánh trống ngực|khó thở|đau ngực|ngất xỉu|chóng mặt|buồn nôn|đau đầu|đau bụng|sốt|ho|khàn tiếng|nuốt khó|nôn|tiêu chảy|táo bón|phù|ngứa|ban đỏ|vàng da|mệt mỏi|sụt cân|chán ăn)",
]


def extract_diagnoses(text: str) -> List[Dict]:
    mentions = []
    for pattern in DIAGNOSIS_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            if m.lastindex and m.lastindex >= 1:
                diag_text = m.group(1).strip()
                if len(diag_text) < 3:
                    continue
                diag_text = re.sub(r"\s+", " ", diag_text)
                start = m.start(1)
                end = m.end(1)
                mentions.append({
                    "text": diag_text,
                    "position": [start, end],
                    "type": "CHẨN_ĐOÁN",
                    "confidence": 0.8,
                    "source": "rule",
                })
    return _non_overlapping(mentions)


def extract_symptoms(text: str) -> List[Dict]:
    mentions = []
    for pattern in SYMPTOM_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            if m.lastindex and m.lastindex >= 1:
                sym_text = m.group(1).strip()
                if len(sym_text) < 3:
                    continue
                sym_text = re.sub(r"\s+", " ", sym_text)
                start = m.start(1)
                end = m.end(1)
                mentions.append({
                    "text": sym_text,
                    "position": [start, end],
                    "type": "TRIỆU_CHỨNG",
                    "confidence": 0.7,
                    "source": "rule",
                })
    return _non_overlapping(mentions)


def detect_assertions(text: str, start: int, end: int, window: int = 50) -> Dict:
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)
    prefix = text[ctx_start:start].lower()
    suffix = text[end:ctx_end].lower()
    is_negated = False
    is_family = False
    is_historical = False
    for trigger in NEGATION_TRIGGERS:
        if re.search(trigger, prefix):
            is_negated = True
            break
    for trigger in FAMILY_TRIGGERS:
        if re.search(trigger, prefix):
            is_family = True
            break
    for trigger in HISTORICAL_TRIGGERS:
        if re.search(trigger, prefix):
            is_historical = True
            break
    return {
        "isNegated": is_negated,
        "isFamily": is_family,
        "isHistorical": is_historical,
    }


def extract_rules(text: str) -> List[Dict]:
    drugs = extract_drugs(text)
    tests = extract_test_names(text)
    labs = extract_lab_values(text)
    diagnoses = extract_diagnoses(text)
    symptoms = extract_symptoms(text)
    all_mentions = drugs + tests + labs + diagnoses + symptoms
    all_mentions = _non_overlapping(all_mentions)
    for m in all_mentions:
        m["assertions"] = detect_assertions(text, m["position"][0], m["position"][1])
    return all_mentions
