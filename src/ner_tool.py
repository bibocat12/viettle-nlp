import json
import os
import re
import sys
from pathlib import Path

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


VALID_TYPES = {
    "THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN",
    "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM",
}

TYPE_MAP = {
    "THUỐC": "THUỐC", "TRIỆU CHỨNG": "TRIỆU_CHỨNG", "CHẨN ĐOÁN": "CHẨN_ĐOÁN",
    "TÊN XÉT NGHIỆM": "TÊN_XÉT_NGHIỆM", "KẾT QUẢ XÉT NGHIỆM": "KẾT_QUẢ_XÉT_NGHIỆM",
    "DRUG": "THUỐC", "SYMPTOM": "TRIỆU_CHỨNG", "DISEASE": "CHẨN_ĐOÁN",
    "TEST": "TÊN_XÉT_NGHIỆM", "RESULT": "KẾT_QUẢ_XÉT_NGHIỆM",
}


def _clean_entity_text(entity_text: str, mtype: str) -> str:
    if mtype != "THUỐC":
        return entity_text.strip()
    cleaned = re.sub(
        r"^(?:được\s+)?(?:điều\s+trị\s+)?(?:bằng|uống|dùng|tiêm)\s+",
        "",
        entity_text.strip(),
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _rule_find_positions(entity_text: str, full_text: str, base_offset: int = 0) -> list:
    """Return every exact occurrence; never reduce a hallucinated span to one token."""
    entity_text = entity_text.strip()
    if not entity_text or len(entity_text) < 2:
        return []

    escaped = re.escape(entity_text)
    boundary_pattern = escaped
    if entity_text[0].isalnum():
        boundary_pattern = rf"(?<!\w){boundary_pattern}"
    if entity_text[-1].isalnum():
        boundary_pattern = rf"{boundary_pattern}(?!\w)"

    # A model keyword is expanded to every occurrence in the chunk. Without
    # lexical boundaries, short entities such as "ho" and "ct" also match
    # inside "khoa", "cho", "bactrim", "lactate", etc. Prefer only clean
    # occurrences. Long mentions may fall back because the translated source
    # occasionally glues a real entity to the following word ("atenololtrong").
    matches = list(re.finditer(boundary_pattern, full_text, re.IGNORECASE))
    if matches:
        return [[base_offset + m.start(), base_offset + m.end()] for m in matches]

    compact_length = sum(char.isalnum() for char in entity_text)
    if compact_length <= 3:
        return []

    matches = list(re.finditer(escaped, full_text, re.IGNORECASE))
    if matches:
        return [[base_offset + m.start(), base_offset + m.end()] for m in matches]

    # Tolerate whitespace differences while retaining exact raw offsets.
    tokens = entity_text.split()
    if len(tokens) >= 2:
        pattern = r"\s+".join(re.escape(token) for token in tokens)
        matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
        return [[base_offset + m.start(), base_offset + m.end()] for m in matches]
    return []


def _rule_find_position(entity_text: str, full_text: str) -> list:
    """Backward-compatible first exact occurrence helper."""
    positions = _rule_find_positions(entity_text, full_text)
    return positions[0] if positions else None


DRUG_ORDER_SUFFIX = re.compile(
    r"^(?:(?:\s+|:)(?:po|iv|im|sc|sl|daily|bid|tid|qid|qd|qam|qhs|"
    r"q\d+h|prn|x\s*\d+))+",
    re.IGNORECASE,
)


def _extend_drug_order(position: list, chunk: str, chunk_start: int) -> list:
    """Extend a model medication span through route/frequency, never indication."""
    local_end = position[1] - chunk_start
    suffix = DRUG_ORDER_SUFFIX.match(chunk[local_end:])
    if suffix:
        return [position[0], position[1] + suffix.end()]
    return position


def _rule_verify_type(entity_text: str, llm_type: str) -> str:
    """Rule-based type verification/correction. Returns None for non-medical text."""
    text_lower = entity_text.lower()

    if text_lower.strip(" .,:;()") in {
        "đó", "thêm", "tiếp", "bằng thuốc", "bán trú",
        "bảo tồn", "nội khoa", "ngoại khoa", "theo đơn", "phù hợp",
    }:
        return None

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
        'mất việc làm', 'luyện tập thể dục', 'tập thể dục',
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
        if re.match(r"^(?:from|còn|từ|xuống|lên|khoảng)\b", text_lower):
            return None
        return "THUỐC"

    normalized = text_lower.strip(" .,:;()")
    if re.fullmatch(
        r"(?:hr|rr|spo2|bp|t)\s*[:=]?\s*\d+(?:[.,/]\d+)?(?:\s*(?:ra|%|°?c))?",
        normalized,
        flags=re.IGNORECASE,
    ):
        return "KẾT_QUẢ_XÉT_NGHIỆM"
    if normalized in {"hạ huyết áp", "tăng men gan"}:
        return "KẾT_QUẢ_XÉT_NGHIỆM"
    if normalized in {
        "bnp", "bilirubin", "bạch cầu", "đường huyết", "glucose",
        "kali", "protein", "tiểu cầu",
    }:
        return "TÊN_XÉT_NGHIỆM"
    if normalized in {
        "giả gout", "gãy xương", "tăng nhãn áp", "thiếu máu",
        "u tuyến", "viêm túi mật",
    }:
        return "CHẨN_ĐOÁN"

    # Test names
    test_kw = ['điện tâm đồ', 'ecg', 'x-quang', 'siêu âm', 'holter', 'chụp cắt lớp',
               'troponin', 'creatinine', 'canxi', 'bảng công thức', 'mri',
               'shunt', 'dẫn lưu', 'nội soi', 'phẫu thuật', 'cea',
               'kháng nguyên ung thư phôi']
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


def _extract_chunk(chunk: str, chunk_start: int, llm) -> list:
    """Extract verbatim entities and deterministically align them inside a chunk."""
    entities = []
    user_prompt = f"Extract all medical entities:\n\n{chunk}"
    response = llm.generate(NER_SYSTEM_PROMPT, user_prompt, 2048)
    parsed = llm.parse_json(response)
    items = parsed if isinstance(parsed, list) else []
    
    for ent in items:
        if not isinstance(ent, dict):
            continue
        entity_text = (ent.get("entity") or ent.get("text") or "").strip()
        entity_type = ent.get("type", "").strip()
        
        if not entity_text or len(entity_text) < 2:
            continue
        if _is_section_header(entity_text):
            continue
        
        mtype = TYPE_MAP.get(entity_type.upper(), entity_type.upper())
        if mtype not in VALID_TYPES:
            continue
        entity_text = _clean_entity_text(entity_text, mtype)
        if len(entity_text) < 2:
            continue

        for position in _rule_find_positions(entity_text, chunk, chunk_start):
            if mtype == "THUỐC":
                position = _extend_drug_order(position, chunk, chunk_start)
            local_start = position[0] - chunk_start
            local_end = position[1] - chunk_start
            entities.append({
                "text": chunk[local_start:local_end],
                "position": position,
                "type": mtype,
                "source": "qwen",
            })
    return entities


def _make_chunks(text: str, max_chars: int = 1200, overlap: int = 120) -> list:
    """Split on line/sentence boundaries while retaining absolute offsets."""
    if len(text) <= max_chars:
        return [(0, text)] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        hard_end = min(len(text), start + max_chars)
        end = hard_end
        if hard_end < len(text):
            search_from = start + max_chars // 2
            candidates = [
                text.rfind("\n", search_from, hard_end),
                text.rfind(". ", search_from, hard_end),
                text.rfind("; ", search_from, hard_end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + (1 if text[boundary] == "\n" else 2)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append((start, chunk))
        if end >= len(text):
            break
        next_start = max(start + 1, end - overlap)
        newline = text.find("\n", next_start, min(end, next_start + overlap))
        start = newline + 1 if newline >= 0 else next_start
    return chunks


def _append_nonoverlapping_mentions(primary: list, supplemental: list) -> list:
    """Add recall proposals without allowing them to replace existing boundaries."""
    merged = list(primary)
    occupied = [tuple(item.get("position", [0, 0])) for item in primary]
    seen = {
        (*item.get("position", [0, 0]), item.get("type"))
        for item in primary
    }
    for mention in supplemental:
        start, end = mention.get("position", [0, 0])
        key = (start, end, mention.get("type"))
        if key in seen:
            continue
        if any(min(end, old_end) > max(start, old_start) for old_start, old_end in occupied):
            continue
        merged.append(mention)
        occupied.append((start, end))
        seen.add(key)
    return merged


def _extract_chunks(text: str, llm, max_chars: int, overlap: int) -> list:
    """Extract one chunk scale in parallel while preserving absolute positions."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    chunks = _make_chunks(text, max_chars=max_chars, overlap=overlap)
    if len(chunks) <= 1:
        return [
            entity
            for chunk_start, chunk in chunks
            for entity in _extract_chunk(chunk, chunk_start, llm)
        ]

    mentions = []
    errors = []
    with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as executor:
        futures = {
            executor.submit(_extract_chunk, chunk, chunk_start, llm): chunk_start
            for chunk_start, chunk in chunks
        }
        for future in as_completed(futures):
            try:
                mentions.extend(future.result())
            except Exception as exc:
                errors.append(f"chunk@{futures[future]}: {exc}")
    if errors:
        raise RuntimeError("; ".join(errors))
    return mentions


def ner_and_type(text: str, llm=None) -> dict:
    """LLM proposes verbatim spans; deterministic rules align and validate them."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    mentions = []

    # Step 1: LLM extracts keywords; model-provided positions are ignored.
    if llm and llm.is_available():
        mentions = _extract_chunks(text, llm, max_chars=1200, overlap=120)
        if os.getenv("NER_MULTISCALE", "").casefold() in {"1", "true", "yes"}:
            supplemental = _extract_chunks(text, llm, max_chars=500, overlap=60)
            mentions = _append_nonoverlapping_mentions(mentions, supplemental)

    # Step 2: Rule-based type verification (skip administrative fragments).
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
        m["source"] = "qwen"
        filtered.append(m)
    mentions = filtered

    # Step 3: remove duplicates introduced by chunk overlap, not real repetitions.
    seen = set()
    unique = []
    for m in mentions:
        key = (*m.get("position", [0, 0]), m.get("type"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(m)
    unique.sort(key=lambda item: (item["position"][0], item["position"][1], item["type"]))
    return {"mentions": unique}
