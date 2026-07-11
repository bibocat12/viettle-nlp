import json
import re
import sys
from pathlib import Path
from difflib import SequenceMatcher

sys.path.insert(0, str(Path(__file__).parent))

from prompt import NER_PROMPT as NER_SYSTEM_PROMPT


# Section headers to filter
SECTION_HEADERS = [
    "tiền sử bệnh", "bệnh sử hiện tại", "bệnh sử", "đánh giá tại bệnh viện",
    "kết quả xét nghiệm", "kết quả hình ảnh", "kết quả khám lâm sàng",
    "các triệu chứng hiện tại", "đặc điểm triệu chứng", "các diễn biến",
    "lý do nhập viện", "thời điểm khởi phát", "tình trạng ngay trước",
    "các yếu tố nguy cơ", "thuốc trước khi nhập viện", "các bệnh lý nội khoa",
    "các tập kinh lâm sàng", "kết quả chẩn đoán", "xử trí thuốc",
    "triệu chứng hiện tại", "các bệnh lý mãn tính",
    "các sự kiện trước khi nhập viện", "các thủ thuật đã thực hiện",
    "tiền sử bệnh nội khoa", "các yếu tố nguy cơ liên quan",
    "đặc điểm triệu chứng khi khám", "kết quả chẩn đoán hình ảnh",
    "bệnh sử hiện tại", "triệu chứng", "các thủ thuật",
]


def _is_section_header(text: str) -> bool:
    text_lower = text.lower().strip()
    for header in SECTION_HEADERS:
        if text_lower.startswith(header) or text_lower == header:
            return True
    return False


def _rule_find_position(entity_text: str, full_text: str) -> list:
    """Find entity position in original text using rules, not LLM."""
    entity_text = entity_text.strip()
    if not entity_text or len(entity_text) < 2:
        return None

    # Strategy 1: Exact match
    idx = full_text.find(entity_text)
    if idx >= 0:
        return [idx, idx + len(entity_text)]

    # Strategy 2: Case-insensitive
    idx = full_text.lower().find(entity_text.lower())
    if idx >= 0:
        return [idx, idx + len(entity_text)]

    # Strategy 3: Fuzzy — find first and last token
    tokens = entity_text.split()
    if len(tokens) >= 2:
        first = re.escape(tokens[0])
        last = re.escape(tokens[-1])
        starts = [m.start() for m in re.finditer(first, full_text, re.IGNORECASE)]
        for s in starts:
            search_end = min(s + len(entity_text) * 2, len(full_text))
            ends = [m.start() for m in re.finditer(last, full_text[s:search_end], re.IGNORECASE)]
            for e in ends:
                end_abs = s + e + len(tokens[-1])
                candidate = full_text[s:end_abs]
                score = SequenceMatcher(None, entity_text.lower(), candidate.lower()).ratio()
                if score > 0.6:
                    return [s, end_abs]

    # Strategy 4: Find first token only
    first_token = re.escape(tokens[0]) if tokens else re.escape(entity_text[:4])
    m = re.search(first_token, full_text, re.IGNORECASE)
    if m:
        return [m.start(), m.end()]

    return None


def _rule_verify_type(entity_text: str, llm_type: str) -> str:
    """Rule-based type verification/correction. Returns None for non-medical text."""
    text_lower = entity_text.lower()

    # Filter out non-medical administrative text
    skip_patterns = [
        'tiền sử bệnh nội khoa', 'tiền sử bệnh hiện tại', 'tiền sử bệnh',
        'bệnh nhân có tiền sử', 'bệnh nhân được', 'bệnh nhân nhập viện',
        'bệnh nhân không', 'bệnh nhân có',
        'được theo dõi', 'ghi nhận trên', 'tình trạng ngay',
        'chỉ định', 'đánh giá', 'nguy cơ', 'ích điều trị',
        'định kỳ', 'chuyên khoa', 'khoa nội', 'khoa ngoại',
        'đến bệnh viện', 'nhập viện', 'chuyển viện',
        'các bệnh lý mãn tính', 'các yếu tố nguy cơ',
        'các sự kiện', 'các thủ thuật', 'đã thực hiện',
        'bệnh nhân được chuyển',
        'có tiền sử', 'triệu chứng hiện tại',
        'đánh giá tại bệnh viện',
    ]
    for pat in skip_patterns:
        if pat in text_lower:
            return None

    # Skip sentence fragments (start with certain words)
    frag_starts = ['đến', 'bệnh nhân', 'để', 'có thể', 'cần', 'sau', 'trước',
                   'trong', 'với', 'vào', 'ngày', 'khoa', 'được', 'đã',
                   'các', 'những', 'một', 'khi']
    first_word = text_lower.split()[0] if text_lower.split() else ''
    if first_word in frag_starts and len(text_lower) > 20:
        return None

    # Skip overly long mentions (sentence fragments)
    if len(entity_text) > 50:
        return None

    # Drug patterns override everything
    if re.search(r'\d+\s*(?:mg|ml|mcg|IU|U)\b', text_lower):
        return "THUỐC"

    # Test names
    test_kw = ['điện tâm đồ', 'ecg', 'x-quang', 'siêu âm', 'holter', 'chụp cắt lớp',
               'troponin', 'creatinine', 'canxi', 'bảng công thức', 'mri',
               'shunt', 'dẫn lưu', 'nội soi', 'phẫu thuật']
    for kw in test_kw:
        if kw in text_lower:
            return "TÊN_XÉT_NGHIỆM"

    # Results
    result_kw = ['bình thường', 'không ghi nhận', 'không có gì', 'bất thường', 'tim to']
    for kw in result_kw:
        if kw in text_lower:
            return "KẾT_QUẢ_XÉT_NGHIỆM"
    if re.match(r'^\d+\.?\d*$', text_lower.strip()):
        return "KẾT_QUẢ_XÉT_NGHIỆM"

    # Diagnoses
    diag_kw = ['viêm', 'ung thư', 'bệnh', 'hội chứng', 'suy', 'rối loạn',
               'tăng huyết áp', 'đái tháo', 'nhồi máu', 'rung nhĩ', 'xơ vữa',
               'phình', 'xơ gan', 'loét', 'nhiễm trùng', 'xuất huyết',
               'sỏi', 'trào ngược', 'thoát vị', 'trĩ', 'u xơ', 'u nang', 'polyp']
    for kw in diag_kw:
        if kw in text_lower:
            return "CHẨN_ĐOÁN"

    return llm_type  # Keep LLM's type if no rule override


def _llm_verify_type(entity_text: str, rule_type: str, context: str, llm) -> str:
    """LLM verification of type classification."""
    if not llm or not llm.is_available():
        return rule_type

    from prompt import TYPE_VERIFY_PROMPT
    system = TYPE_VERIFY_PROMPT
    user = f'Mention: "{entity_text}"\nRule-classified as: {rule_type}\nContext: "{context[:200]}"'
    response = llm.generate(system, user, 150)
    result = llm.parse_json(response)

    if isinstance(result, dict) and not result.get("correct", True):
        corrected = result.get("corrected_type", "")
        valid = ("THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM")
        if corrected in valid:
            return corrected
    return rule_type


def _extract_chunk(chunk: str, llm) -> list:
    """Extract entities from a single chunk via LLM."""
    entities = []
    user_prompt = f"Extract all medical entities:\n\n{chunk}"
    response = llm.generate(NER_SYSTEM_PROMPT, user_prompt, 2048)
    parsed = llm.parse_json(response)
    items = parsed if isinstance(parsed, list) else []
    
    for ent in items:
        if not isinstance(ent, dict):
            continue
        entity_text = ent.get("entity", "").strip()
        entity_type = ent.get("type", "").strip()
        
        if not entity_text or len(entity_text) < 2:
            continue
        if _is_section_header(entity_text):
            continue
        
        # Map to Vietnamese types
        type_map = {
            "THUỐC": "THUỐC", "TRIỆU CHỨNG": "TRIỆU_CHỨNG", "CHẨN ĐOÁN": "CHẨN_ĐOÁN",
            "TÊN XÉT NGHIỆM": "TÊN_XÉT_NGHIỆM", "KẾT QUẢ XÉT NGHIỆM": "KẾT_QUẢ_XÉT_NGHIỆM",
            "DRUG": "THUỐC", "SYMPTOM": "TRIỆU_CHỨNG", "DISEASE": "CHẨN_ĐOÁN",
            "TEST": "TÊN_XÉT_NGHIỆM", "RESULT": "KẾT_QUẢ_XÉT_NGHIỆM",
        }
        mtype = type_map.get(entity_type.upper(), entity_type)
        
        entities.append({
            "text": entity_text,
            "type": mtype,
            "source": "ner_llm",
        })
    return entities


def ner_and_type(text: str, llm=None) -> dict:
    """LLM extracts keywords → rules find positions → rules verify → LLM verifies."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    mentions = []

    # Step 1: LLM extracts keywords (NO positions)
    if llm and llm.is_available():
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        chunks = [(idx, c) for idx, c in enumerate(chunks) if len(c.strip()) >= 20]

        if len(chunks) <= 1:
            # Single chunk: no parallelism needed
            for chunk_idx, chunk in chunks:
                entities = _extract_chunk(chunk, llm)
                mentions.extend(entities)
        else:
            # Parallel chunk processing
            with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as ex:
                futures = {ex.submit(_extract_chunk, chunk, llm): idx for idx, chunk in chunks}
                for future in as_completed(futures):
                    try:
                        mentions.extend(future.result())
                    except Exception as e:
                        print(f"Chunk NER error: {e}", file=__import__('sys').stderr)

    # Step 2: Rule-based position finding
    for m in mentions:
        pos = _rule_find_position(m["text"], text)
        if pos:
            m["position"] = pos
        else:
            m["position"] = None

    mentions = [m for m in mentions if m.get("position") is not None]

    # Step 3: Rule-based type verification (skip non-medical text)
    filtered = []
    for m in mentions:
        ctx_start = max(0, m["position"][0] - 50)
        ctx_end = min(len(text), m["position"][1] + 50)
        context = text[ctx_start:ctx_end]
        rule_type = _rule_verify_type(m["text"], m["type"])
        if rule_type is None:
            continue  # Non-medical text, skip
        m["type"] = rule_type
        m["type_candidates"] = [{"label": rule_type, "score": 0.85}]
        m["source"] = "ner_llm_relocated"
        filtered.append(m)
    mentions = filtered

    # Step 4: LLM verification of types (only for uncertain/ambiguous cases)
    for m in mentions:
        mtype = m["type"]
        # Only verify types that might be ambiguous
        if mtype in ("TRIỆU_CHỨNG", "CHẨN_ĐOÁN"):
            ctx_start = max(0, m["position"][0] - 100)
            ctx_end = min(len(text), m["position"][1] + 100)
            context = text[ctx_start:ctx_end]
            verified = _llm_verify_type(m["text"], mtype, context, llm)
            if verified != mtype:
                m["type"] = verified
                m["type_candidates"] = [{"label": verified, "score": 0.90}]
                m["llm_corrected"] = True

    # Step 5: Deduplicate
    seen_positions = set()
    unique = []
    for m in mentions:
        pos = tuple(m.get("position", [0, 0]))
        if pos in seen_positions:
            continue
        seen_positions.add(pos)
        unique.append(m)
    mentions = unique

    return {"mentions": mentions}
