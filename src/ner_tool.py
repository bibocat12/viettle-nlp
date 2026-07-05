import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

NER_SYSTEM_PROMPT = """Bạn là hệ thống trích xuất thực thể y khoa tiếng Việt.

CHỈ ĐƯỢC sử dụng 5 loại thực thể sau:
- THUỐC: tên thuốc + liều + đường dùng + tần suất (VD: "metoprolol 25mg po bid")
- TRIỆU_CHỨNG: triệu chứng, dấu hiệu lâm sàng (VD: "đau ngực", "khó thở", "sốt")
- CHẨN_ĐOÁN: chẩn đoán bệnh (VD: "tăng huyết áp", "viêm phổi", "đái tháo đường")
- TÊN_XÉT_NGHIỆM: tên xét nghiệm (VD: "điện tâm đồ", "troponin")
- KẾT_QUẢ_XÉT_NGHIỆM: giá trị kết quả (VD: "bình thường", "0.01")

QUAN TRỌNG:
1. THUỐC phải bao gồm tên + liều (VD: "metoprolol 25mg", KHÔNG tách riêng)
2. KHÔNG dùng loại thực thể khác ngoài 5 loại trên
3. KHÔNG trích xuất các tiêu đề phần:
   "Tiền sử bệnh", "Các bệnh lý mãn tính", "Triệu chứng hiện tại",
   "Các sự kiện trước khi nhập viện", "Đánh giá tại bệnh viện",
   "Các thủ thuật đã thực hiện", "Kết quả xét nghiệm", "Các yếu tố nguy cơ"
4. KHÔNG trích xuất các metadata:
   "Vị trí:", "Mức độ:", "Thời gian:", "Chiếu xạ:"
5. KHÔNG trích xuất câu không hoàn chỉnh (thiếu chủ ngữ/vị ngữ)
6. Mỗi thực thể phải là một khái niệm y tế HOÀN CHỈNH
7. Trích xuất TẤT CẢ thực thể y tế có trong văn bản

Trả về JSON array:
[{"entity": "...", "type": "..."}]"""


# Section headers to filter out from LLM output
SECTION_HEADERS = [
    "tiền sử bệnh", "bệnh sử hiện tại", "bệnh sử", "đánh giá tại bệnh viện",
    "kết quả xét nghiệm", "kết quả hình ảnh", "kết quả khám lâm sàng",
    "các triệu chứng hiện tại", "đặc điểm triệu chứng", "các diễn biến",
    "lý do nhập viện", "thời điểm khởi phát", "tình trạng ngay trước",
    "các yếu tố nguy cơ", "thuốc trước khi nhập viện", "các bệnh lý nội khoa",
    "các tập kinh lâm sàng", "kết quả chẩn đoán", "xử trí thuốc",
    "các bệnh đã điều trị trước đây",
    # Additional headers
    "triệu chứng hiện tại", "các bệnh lý mãn tính",
    "các sự kiện trước khi nhập viện", "các thủ thuật đã thực hiện",
    "tiền sử bệnh nội khoa", "các yếu tố nguy cơ liên quan",
    "đặc điểm triệu chứng khi khám", "kết quả chẩn đoán hình ảnh",
    "bệnh sử hiện tại", "triệu chứng",
]


def _is_section_header(text: str) -> bool:
    text_lower = text.lower().strip()
    for header in SECTION_HEADERS:
        if text_lower.startswith(header) or text_lower == header:
            return True
    return False


# Drug regex patterns for verification
DRUG_PATTERN = re.compile(
    r"(?P<name>[A-Za-zÀ-ỹ]{3,25})"
    r"\s+(?P<dose>\d+(?:\.\d+)?)\s*(?P<unit>mg|ml|mcg|IU|U)\b"
    r"(?:\s+(?P<route>po|iv|im|sc|sl|tiêm|uống|đặt|nhỏ|bôi|thoa))?"
    r"(?:\s+(?P<freq>bid|tid|qd|qid|prn|x\s*\d+))?",
    re.IGNORECASE
)

TEST_PATTERNS = [
    r"điện tâm đồ", r"ecg", r"chụp x-quang", r"x-quang",
    r"siêu âm tim", r"siêu âm", r"chụp cắt lớp vi tính", r"CT",
    r"MRI", r"holter", r"monitor holter",
    r"phân tích nước tiểu", r"bảng công thức máu",
    r"bảng công thức sinh hóa", r"bảng chức năng gan",
    r"troponin", r"creatinine", r"canxi", r"canxi toàn phần",
    # Medical procedures
    r"đặt shunt", r"dẫn lưu", r"shunt dẫn lưu",
    r"đặt stent", r"nội soi", r"mổ", r"phẫu thuật",
]

NORMAL_RESULTS = [
    r"bình thường", r"không ghi nhận gì bất thường",
    r"không có gì đáng chú ý", r"không có gì bất thường",
]


def _verify_with_rules(text: str, mentions: list) -> list:
    """LLM-first: rules verify and correct LLM output.

    - Rules catch structured patterns LLMs often miss (drugs, tests)
    - Rules correct wrong types from LLM
    - Rules add missing high-precision mentions
    """
    verified = list(mentions)
    existing_spans = set()
    for m in verified:
        s, e = m["position"]
        existing_spans.add((s, e))

    # Add drugs LLM might have missed
    for m in DRUG_PATTERN.finditer(text):
        name = m.group("name").strip()
        if len(name) < 3:
            continue
        dose = m.group("dose") or ""
        unit = m.group("unit") or ""
        drug_text = f"{name} {dose}{unit}".strip()
        s, e = m.start(), m.end()

        # Skip if already covered
        if any(s < existing_e and e > existing_s for existing_s, existing_e in existing_spans):
            continue

        verified.append({
            "text": drug_text,
            "position": [s, e],
            "type": "THUỐC",
            "type_candidates": [{"label": "THUỐC", "score": 0.98}],
            "source": "rule_verified",
        })
        existing_spans.add((s, e))

    # Add test names and medical procedures LLM might have missed
    for pattern in TEST_PATTERNS:
        for m in re.finditer(re.escape(pattern), text, re.IGNORECASE):
            s, e = m.start(), m.end()
            if any(s < existing_e and e > existing_s for existing_s, existing_e in existing_spans):
                continue
            # Expand to full procedure text (from start of line to end of line)
            line_start = text.rfind('\n', 0, s)
            if line_start < 0:
                line_start = 0
            else:
                line_start += 1  # Skip the newline
            line_end = text.find('\n', s)
            if line_end < 0:
                line_end = len(text)
            full_text = text[line_start:line_end].strip()
            # Remove colon-prefixed label (e.g., "Các thủ thuật đã thực hiện: ")
            colon_idx = full_text.find(':')
            if colon_idx >= 0 and colon_idx < 50:
                after_colon = full_text[colon_idx+1:].strip()
                if after_colon and len(after_colon) > len(full_text) * 0.3:
                    full_text = after_colon
            start_pos = line_start + (full_text.find(text[s:e]) if full_text.find(text[s:e]) >= 0 else (s - line_start))
            if full_text.find(text[s:e]) >= 0:
                start_pos = line_start + full_text.find(text[s:e])
            else:
                start_pos = s
            verified.append({
                "text": full_text,
                "position": [line_start, line_start + len(full_text)],
                "type": "TÊN_XÉT_NGHIỆM",
                "type_candidates": [{"label": "TÊN_XÉT_NGHIỆM", "score": 0.95}],
                "source": "rule_verified",
            })
            existing_spans.add((line_start, line_start + len(full_text)))

    # Filter out remaining section headers and fragments
    filtered = []
    for m in verified:
        text = m.get("text", "").strip()
        text_lower = text.lower()

        # Skip section headers
        if _is_section_header(text):
            continue

        # Skip fragments (too short, starts mid-word)
        if len(text) < 3:
            continue
        # Fragment starts with lowercase Vietnamese char (cut mid-word like "ệnh")
        vn_lower = "àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳýỹỷỵ"
        if text[0] in vn_lower and len(text) < 10:
            continue
        # Fragment is a single letter + optional chars (like "ình", "ăm")
        if len(text) < 5 and text[0].islower() and not any(c.isdigit() for c in text):
            continue
        # Numeric-only values without test context
        if re.match(r'^\d+\.?\d*$', text) and len(text) > 10:
            continue

        # Fix wrong types with rules
        mtype = m.get("type", m.get("type_candidates", [{}])[0].get("label", ""))
        text_lower = m.get("text", "").lower()

        # Drug pattern detected but LLM classified wrong
        if DRUG_PATTERN.match(m.get("text", "")):
            if mtype != "THUỐC":
                m["type"] = "THUỐC"
                m["type_candidates"] = [{"label": "THUỐC", "score": 0.95}]

        # Normal results
        for pattern in NORMAL_RESULTS:
            if re.match(pattern, text_lower):
                if mtype != "KẾT_QUẢ_XÉT_NGHIỆM":
                    m["type"] = "KẾT_QUẢ_XÉT_NGHIỆM"
                    m["type_candidates"] = [{"label": "KẾT_QUẢ_XÉT_NGHIỆM", "score": 0.90}]

        # Diagnoses classified wrong — fix
        diag_kw = ['hội chứng', 'viêm', 'ung thư', 'bệnh', 'suy ', 'tăng huyết áp',
                   'đái tháo', 'nhồi máu', 'rung nhĩ', 'xơ vữa', 'phình', 'loét',
                   'nhiễm trùng', 'viêm phổi', 'viêm gan', 'xơ gan', 'xơ',
                   'viêm thận', 'viêm khớp', 'viêm màng não', 'lao', 'hen']
        if mtype in ("THUỐC", "TRIỆU_CHỨNG"):
            for kw in diag_kw:
                if kw in text_lower:
                    m["type"] = "CHẨN_ĐOÁN"
                    m["type_candidates"] = [{"label": "CHẨN_ĐOÁN", "score": 0.80}]
                    break

        filtered.append(m)

    return filtered


def _extract_candidate_spans(text: str) -> list:
    """Fallback heuristic extraction when no LLM available."""
    spans = []
    segments = re.split(r'(?<=[.!?\n])\s+', text)
    offset = 0
    for seg in segments:
        start = text.find(seg, offset)
        if start < 0:
            offset += len(seg) + 1
            continue
        for m in re.finditer(r'[A-ZÀ-ỹ][a-zà-ỹ]+(?:\s+[a-zà-ỹ]+){0,5}', seg):
            span_text = m.group()
            if len(span_text) >= 3 and not _is_section_header(span_text):
                spans.append({
                    "text": span_text,
                    "position": [start + m.start(), start + m.end()],
                })
        offset = start + len(seg)
    return spans


def _classify_type_heuristic(text: str) -> str:
    text_lower = text.lower()
    if DRUG_PATTERN.match(text):
        return "THUỐC"
    test_kw = ['điện tâm đồ', 'ecg', 'x-quang', 'siêu âm', 'holter',
               'troponin', 'creatinine', 'canxi', 'bảng công thức', 'chụp cắt lớp']
    for kw in test_kw:
        if kw in text_lower:
            return "TÊN_XÉT_NGHIỆM"
    if re.match(r'^[\d\.]+$', text.strip()):
        return "KẾT_QUẢ_XÉT_NGHIỆM"
    result_kw = ['bình thường', 'không ghi nhận', 'không có gì', 'bất thường']
    for kw in result_kw:
        if kw in text_lower:
            return "KẾT_QUẢ_XÉT_NGHIỆM"
    diag_kw = ['viêm', 'ung thư', 'bệnh', 'hội chứng', 'suy', 'tăng huyết áp',
               'đái tháo', 'nhồi máu', 'rung nhĩ', 'xơ vữa', 'phình']
    for kw in diag_kw:
        if kw in text_lower:
            return "CHẨN_ĐOÁN"
    return "TRIỆU_CHỨNG"


def ner_and_type(text: str, llm=None) -> dict:
    """LLM-first extraction with rules verification.

    Strategy: LLM extracts everything first, then rules verify and correct.
    """
    mentions = []

    if llm and llm.is_available():
        # Split into chunks
        chunks = [text[i:i+600] for i in range(0, len(text), 600)]
        for chunk_idx, chunk in enumerate(chunks):
            chunk_offset = chunk_idx * 1200
            if len(chunk.strip()) < 20:
                continue

            user_prompt = f"Trích xuất thực thể y tế từ văn bản lâm sàng sau:\n\n{chunk}"
            response = llm.generate(NER_SYSTEM_PROMPT, user_prompt, 3072)

            try:
                parsed = llm.parse_json(response)
                entities = parsed if isinstance(parsed, list) else parsed.get("entities", []) if isinstance(parsed, dict) else []

                for ent in entities:
                    if not isinstance(ent, dict):
                        continue
                    entity_text = ent.get("entity", "").strip()
                    entity_type = ent.get("type", "").strip().upper()
                    start = ent.get("start", -1)
                    end = ent.get("end", -1)

                    if not entity_text or len(entity_text) < 2:
                        continue
                    if _is_section_header(entity_text):
                        continue

                    # Always use text search for positions (LLM positions are unreliable)
                    idx = chunk.find(entity_text)
                    if idx < 0:
                        idx = chunk.lower().find(entity_text.lower())
                    if idx < 0:
                        continue
                    pos = [chunk_offset + idx, chunk_offset + idx + len(entity_text)]

                    type_map = {
                        "THUỐC": "THUỐC", "TRIỆU_CHỨNG": "TRIỆU_CHỨNG",
                        "CHẨN_ĐOÁN": "CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM": "TÊN_XÉT_NGHIỆM",
                        "KẾT_QUẢ_XÉT_NGHIỆM": "KẾT_QUẢ_XÉT_NGHIỆM",
                        "DRUG": "THUỐC", "SYMPTOM": "TRIỆU_CHỨNG", "DISEASE": "CHẨN_ĐOÁN",
                        "DIAGNOSIS": "CHẨN_ĐOÁN", "TEST": "TÊN_XÉT_NGHIỆM",
                        "RESULT": "KẾT_QUẢ_XÉT_NGHIỆM", "SYMPTOM_AND_DISEASE": "TRIỆU_CHỨNG",
                        "MEDICAL_PROCEDURE": "TÊN_XÉT_NGHIỆM",
                    }
                    mtype = type_map.get(entity_type, _classify_type_heuristic(entity_text))

                    mentions.append({
                        "text": entity_text,
                        "position": pos,
                        "type_candidates": [{"label": mtype, "score": 0.85}],
                        "source": "ner",
                    })
            except Exception as e:
                print(f"NER parse error: {e}", file=sys.stderr)

        # LLM-first: rules verify and add missing structured mentions
        mentions = _verify_with_rules(text, mentions)

    # Fallback: heuristic only if no LLM
    if not mentions:
        spans = _extract_candidate_spans(text)
        for span in spans[:50]:
            mtype = _classify_type_heuristic(span["text"])
            mentions.append({
                "text": span["text"],
                "position": span["position"],
                "type_candidates": [{"label": mtype, "score": 0.70}],
                "source": "ner_heuristic",
            })

    return {"mentions": mentions}
