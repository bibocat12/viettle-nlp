import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import AutoTokenizer, AutoModel


# Section headers that should NOT be extracted as entities
SECTION_HEADERS = [
    "tiền sử bệnh", "bệnh sử hiện tại", "bệnh sử", "đánh giá tại bệnh viện",
    "kết quả xét nghiệm", "kết quả hình ảnh", "kết quả khám lâm sàng",
    "các triệu chứng hiện tại", "đặc điểm triệu chứng", "các diễn biến",
    "lý do nhập viện", "thời điểm khởi phát", "tình trạng ngay trước",
    "các yếu tố nguy cơ", "thuốc trước khi nhập viện", "các bệnh lý nội khoa",
    "các tập kinh lâm sàng", "kết quả chẩn đoán", "xử trí thuốc",
]


def _is_section_header(text: str) -> bool:
    text_lower = text.lower().strip()
    for header in SECTION_HEADERS:
        if text_lower.startswith(header) or text_lower == header:
            return True
    return False


def _is_fragment(text: str) -> bool:
    """Check if text is a fragment (too short, starts/ends mid-word)."""
    text = text.strip()
    if len(text) < 3:
        return True
    # Check if starts with lowercase (mid-sentence fragment)
    if text[0].islower() and not text[0].isdigit():
        return True
    # Check if it's just punctuation or numbers
    if re.match(r'^[\d\s\.\,\:\-\(\)]+$', text):
        return True
    return False


NER_SYSTEM_PROMPT = """You are a Vietnamese medical NER expert. Extract medical entities from clinical text.

ENTITY TYPES (extract EXACTLY these 5 types):

1. **THUỐC** (Medications)
   - Drug names WITH dosage when present
   - Examples: "metoprolol 25mg", "aspirin 325mg", "amoxicillin 500mg"
   - Include: drug name + dose + unit (e.g., "metoprolol 25mg po bid")
   - Do NOT include: "không tuân thủ điều trị" (that's a symptom)

2. **TRIỆU_CHỨNG** (Symptoms)
   - Patient complaints, signs, observable conditions
   - Examples: "đau ngực", "khó thở", "đánh trống ngực", "ngất xỉu", "phù mắt cá chân"
   - Include: symptom descriptions, patient complaints
   - Do NOT include: "Không đánh trống ngực" (negated symptoms should still be extracted, but mark as negated)

3. **CHẨN_ĐOÁN** (Diagnoses)
   - Confirmed or suspected diseases/conditions
   - Examples: "tăng huyết áp", "rung nhĩ", "bệnh tim mạch do xơ vữa động mạch", "phình động mạch chủ"
   - Include: disease names, confirmed conditions
   - Do NOT include: section headers like "Tiền sử bệnh"

4. **TÊN_XÉT_NGHIỆM** (Test names)
   - Medical tests, procedures, examinations
   - Examples: "điện tâm đồ", "chụp x-quang ngực", "siêu âm tim", "bảng công thức máu"
   - Include: test names, examination names
   - Do NOT include: test results (those are KẾT_QUẢ_XÉT_NGHIỆM)

5. **KẾT_QUẢ_XÉT_NGHIỆM** (Test results)
   - Results, values, findings from tests
   - Examples: "bình thường", "0.01", "tim to", "không có bất thường"
   - Include: numeric values, qualitative results
   - Do NOT include: test names (those are TÊN_XÉT_NGHIỆM)

CRITICAL RULES:
1. Extract COMPLETE mentions, not fragments (e.g., "bệnh tim mạch do xơ vữa động mạch" NOT "ệnh tim mạch")
2. Do NOT extract section headers (e.g., "Tiền sử bệnh", "Lý do nhập viện")
3. Do NOT extract metadata (e.g., "Vị trí:", "Mức độ:", "Thời gian:")
4. Each mention must be a complete medical concept
5. Return ONLY valid JSON, no other text

OUTPUT FORMAT:
Return a JSON list of objects with "entity" and "type" keys:
[{"entity": "đau ngực", "type": "TRIỆU_CHỨNG"}, {"entity": "metoprolol 25mg", "type": "THUỐC"}]"""


_ner_model = None
_ner_tokenizer = None


def _load_ner_model():
    """Load ViPubMedDeBERTa for embeddings (used for type classification fallback)."""
    global _ner_model, _ner_tokenizer
    if _ner_model is not None:
        return
    model_path = str(Path(__file__).parent.parent / "models" / "vipubmed-deberta-base")
    _ner_tokenizer = AutoTokenizer.from_pretrained(model_path)
    _ner_model = AutoModel.from_pretrained(model_path)
    _ner_model.eval()


def _classify_type_heuristic(text: str, context: str = "") -> dict:
    """Rule-based type classification as fallback."""
    text_lower = text.lower()

    # Drug patterns
    drug_patterns = [r'\d+\s*(?:mg|ml|mcg|IU|U)\b', r'(?:po|iv|im|sc|sl)\b']
    for p in drug_patterns:
        if re.search(p, text_lower):
            return {"type": "THUỐC", "score": 0.85}

    # Test names
    test_kw = ['điện tâm đồ', 'ecg', 'x-quang', 'siêu âm', 'mri', 'ct scan', 'holter',
                'troponin', 'creatinine', 'canxi', 'hemoglobin', 'tiểu cầu', 'bạch cầu',
                'bảng công thức', 'bản phân tích', 'chụp cắt lớp', 'siêu âm tim']
    for kw in test_kw:
        if kw in text_lower:
            return {"type": "TÊN_XÉT_NGHIỆM", "score": 0.85}

    # Test results
    result_kw = ['bình thường', 'không ghi nhận', 'không có gì', 'bất thường', 'tim to',
                 'không có bất thường', 'không có quá trình', 'không có gì đáng']
    for kw in result_kw:
        if kw in text_lower:
            return {"type": "KẾT_QUẢ_XÉT_NGHIỆM", "score": 0.80}

    # Check for numeric values (likely results)
    if re.match(r'^\d+\.?\d*$', text.strip()):
        return {"type": "KẾT_QUẢ_XÉT_NGHIỆM", "score": 0.75}

    # Diagnoses
    diag_kw = ['viêm', 'u ', 'ung thư', 'bệnh', 'hội chứng', 'suy ', 'tăng huyết áp',
                'đái tháo', 'nhồi máu', 'rung nhĩ', 'xơ vữa', 'phình', 'loét', 'nhiễm trùng',
                'viêm phổi', 'viêm gan', 'viêm thận', 'đau thắt ngực', 'suyễn', 'hen', 'lao']
    for kw in diag_kw:
        if kw in text_lower:
            return {"type": "CHẨN_ĐOÁN", "score": 0.70}

    # Default to symptom
    return {"type": "TRIỆU_CHỨNG", "score": 0.60}


def _extract_candidate_spans(text: str) -> list:
    """Extract candidate spans using sentence splitting and noun phrase patterns."""
    spans = []
    # Split by common delimiters
    segments = re.split(r'(?<=[.!?\n])\s+', text)
    offset = 0
    for seg in segments:
        start = text.find(seg, offset)
        if start < 0:
            offset += len(seg) + 1
            continue

        # Extract noun-like spans (Vietnamese: 2+ word sequences)
        for m in re.finditer(r'[A-ZÀ-ỹ][a-zà-ỹ]+(?:\s+[a-zà-ỹ]+){0,5}', seg):
            span_text = m.group()
            if len(span_text) >= 3:
                spans.append({
                    "text": span_text,
                    "position": [start + m.start(), start + m.end()],
                })
        offset = start + len(seg)
    return spans


def ner_and_type(text: str, llm=None) -> dict:
    """Extract mentions and classify types.

    Uses Qwen via RunPod for NER if available, falls back to heuristic extraction.
    """
    mentions = []

    if llm and llm.is_available():
        # Split into chunks if too long (RunPod has context limits)
        chunks = [text[i:i+1200] for i in range(0, len(text), 1200)]
        for chunk_idx, chunk in enumerate(chunks):
            chunk_offset = chunk_idx * 1200

            # Skip chunks that are mostly metadata
            if len(chunk.strip()) < 20:
                continue

            user_prompt = f"Extract medical entities from this Vietnamese clinical text:\n\n{chunk}"
            response = llm.generate(NER_SYSTEM_PROMPT, user_prompt, 1024)

            try:
                parsed = llm.parse_json(response)
                if isinstance(parsed, list):
                    entities = parsed
                elif isinstance(parsed, dict) and "entities" in parsed:
                    entities = parsed["entities"]
                else:
                    entities = []

                for ent in entities:
                    if not isinstance(ent, dict):
                        continue
                    entity_text = ent.get("entity", "").strip()
                    entity_type = ent.get("type", "").strip().upper()

                    if not entity_text or len(entity_text) < 2:
                        continue

                    # Filter out section headers
                    if _is_section_header(entity_text):
                        continue

                    # Filter out fragments
                    if _is_fragment(entity_text):
                        continue

                    # Find position in chunk
                    idx = chunk.find(entity_text)
                    if idx < 0:
                        # Try case-insensitive search
                        idx = chunk.lower().find(entity_text.lower())
                        if idx < 0:
                            continue

                    # Map type from prompt format to competition format
                    type_map = {
                        "THUỐC": "THUỐC",
                        "TRIỆU_CHỨNG": "TRIỆU_CHỨNG",
                        "CHẨN_ĐOÁN": "CHẨN_ĐOÁN",
                        "TÊN_XÉT_NGHIỆM": "TÊN_XÉT_NGHIỆM",
                        "KẾT_QUẢ_XÉT_NGHIỆM": "KẾT_QUẢ_XÉT_NGHIỆM",
                        "DRUG": "THUỐC",
                        "SYMPTOM": "TRIỆU_CHỨNG",
                        "DISEASE": "CHẨN_ĐOÁN",
                        "DIAGNOSIS": "CHẨN_ĐOÁN",
                        "TEST": "TÊN_XÉT_NGHIỆM",
                        "RESULT": "KẾT_QUẢ_XÉT_NGHIỆM",
                        "SYMPTOM_AND_DISEASE": "TRIỆU_CHỨNG",
                        "MEDICAL_PROCEDURE": "TÊN_XÉT_NGHIỆM",
                    }
                    mtype = type_map.get(entity_type, "TRIỆU_CHỨNG")

                    # Validate type with heuristic
                    heuristic = _classify_type_heuristic(entity_text, chunk)
                    if heuristic["score"] > 0.8 and heuristic["type"] != mtype:
                        mtype = heuristic["type"]

                    mentions.append({
                        "text": entity_text,
                        "position": [chunk_offset + idx, chunk_offset + idx + len(entity_text)],
                        "type_candidates": [{"label": mtype, "score": 0.85}],
                        "source": "ner",
                    })
            except Exception as e:
                print(f"NER parse error: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)

    # Fallback: heuristic extraction if no LLM or no results
    if not mentions:
        spans = _extract_candidate_spans(text)
        for span in spans[:50]:  # Limit to 50 spans
            # Filter out section headers and fragments
            if _is_section_header(span["text"]) or _is_fragment(span["text"]):
                continue
            type_info = _classify_type_heuristic(span["text"], text)
            mentions.append({
                "text": span["text"],
                "position": span["position"],
                "type_candidates": [{"label": type_info["type"], "score": type_info["score"]}],
                "source": "ner_heuristic",
            })

    return {"mentions": mentions}
