import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))


NEGATION_PATTERNS = [
    (r"không\b", 5), (r"chưa\b", 5), (r"không có\b", 8), (r"chưa có\b", 8),
    (r"không ghi nhận\b", 15), (r"không có gì\b", 12), (r"bình thường\b", 12),
    (r"không có bất thường\b", 20), (r"không có gì đáng\b", 18),
    (r"không buồn nôn\b", 15), (r"không đánh trống ngực\b", 20),
    (r"không quầng sáng\b", 15), (r"không chóng mặt\b", 15),
    (r"không có biểu hiện\b", 18), (r"không có gì đáng chú ý\b", 22),
    (r"không có triệu chứng\b", 20), (r"không có dấu hiệu\b", 18),
    (r"không mất kiểm soát\b", 18), (r"không đổ mồ hôi\b", 15),
    (r"không còn\b", 8), (r"không cải thiện\b", 15),
]

FAMILY_PATTERNS = [
    (r"bố\b", 3), (r"mẹ\b", 3), (r"cha\b", 3), (r"người nhà\b", 8),
    (r"gia đình\b", 8), (r"tiền sử gia đình\b", 18), (r"theo lời\b", 8),
    ("thân nhân", 8), ("anh", 3), ("chị", 3), ("em", 2), ("ông", 3), ("bà", 3),
]

HISTORICAL_PATTERNS = [
    (r"tiền sử\b", 8), (r"trước đây\b", 10), (r"trước khi\b", 10),
    (r"trước nhập viện\b", 16), (r"trước đó\b", 10), (r"đã điều trị\b", 12),
    (r"đã dùng\b", 8), (r"đã chỉ định\b", 13), (r"đã sử dụng\b", 12),
    (r"được chỉ định\b", 15), (r"các bệnh đã điều trị\b", 22),
    (r"các bệnh lý mãn tính\b", 22), (r"tái phát\b", 8),
    (r"có tiền sử\b", 12), (r"hút thuốc\b", 15), (r"sử dụng rượu\b", 15),
    (r"nghi ngờ\b", 12), (r"mạn tính\b", 12), (r"mãn tính\b", 12),
    (r"đã từng\b", 10),
]

LLM_ASSERTION_SYSTEM = """Bạn là chuyên gia xác định assertion y khoa tiếng Việt.
Phân tích mention trong context và trả về 3 loại assertion:

- isNegated: mention bị phủ định? (có "không", "chưa" đứng trước)
- isFamily: mention liên quan đến tiền sử gia đình? (có "bố", "mẹ", "người nhà")
- isHistorical: mention là quá khứ/tiền sử? (có "tiền sử", "trước đây", "đã")

Ví dụ:
Mention: "ho", Context: "bệnh nhân không ho"
→ {"isNegated": true, "isFamily": false, "isHistorical": false}

Mention: "đái tháo đường", Context: "bố bị đái tháo đường"
→ {"isNegated": false, "isFamily": true, "isHistorical": false}

Mention: "viêm gan B", Context: "tiền sử bệnh nhân bị viêm gan B"
→ {"isNegated": false, "isFamily": false, "isHistorical": true}

Return JSON only."""


def _check_pattern(text_lower: str, start: int, end: int, patterns: list, window: int = 50) -> bool:
    ctx_start = max(0, start - window)
    prefix = text_lower[ctx_start:start]
    # Only check prefix (before entity), not suffix (after entity)
    # Negation must precede the entity: "không X" not "X không"
    for pattern, _ in patterns:
        if re.search(pattern, prefix):
            return True
    return False


def _llm_classify(text: str, mention: dict, llm) -> Optional[Dict]:
    """Use LLM to classify assertions."""
    if not llm or not llm.is_available():
        return None

    start, end = mention["position"][0], mention["position"][1]
    ctx_start = max(0, start - 150)
    ctx_end = min(len(text), end + 150)
    context = text[ctx_start:ctx_end]

    mtype = mention.get("type", "")
    # Only use LLM for meaningful types
    if mtype not in ("CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"):
        return None

    user = f"Mention: \"{mention['text']}\"\nType: {mtype}\nContext: \"{context}\""
    response = llm.generate(LLM_ASSERTION_SYSTEM, user, 100)
    result = llm.parse_json(response)

    if isinstance(result, dict) and "isNegated" in result:
        return {
            "isNegated": bool(result.get("isNegated", False)),
            "isFamily": bool(result.get("isFamily", False)),
            "isHistorical": bool(result.get("isHistorical", False)),
            "confidence": 0.90,
            "evidence": context,
            "source": "llm",
        }
    return None


def _relative_is_reporter(text_lower: str, start: int) -> bool:
    """Return true when a relative is only the source of patient information."""
    prefix = text_lower[max(0, start - 120):start]
    boundary = max(prefix.rfind(token) for token in ("\n", ".", ";"))
    if boundary >= 0:
        prefix = prefix[boundary + 1:]
    return bool(re.search(
        r"(?:người\s+nhà|gia\s+đình).{0,40}"
        r"(?:nhận\s+thấy|kể\s+lại|cho\s+biết|báo|thấy)",
        prefix,
        flags=re.DOTALL,
    ))


def classify_assertions(text: str, mention: dict, llm=None) -> dict:
    """Classify assertions for a single mention.

    Strategy: LLM for CHẨN_ĐOÁN/THUỐC/TRIỆU_CHỨNG, rules as fallback.
    """
    start, end = mention["position"][0], mention["position"][1]
    text_lower = text.lower()
    mtype = mention.get("type", "")

    # Phase 1: LLM assertion (for meaningful types)
    if mtype in ("CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"):
        llm_result = _llm_classify(text, mention, llm)
        if llm_result:
            if _relative_is_reporter(text_lower, start):
                llm_result["isFamily"] = False
            return llm_result

    # Phase 2: Rule-based fallback
    is_negated = _check_pattern(text_lower, start, end, NEGATION_PATTERNS)
    is_family = (
        not _relative_is_reporter(text_lower, start)
        and _check_pattern(text_lower, start, end, FAMILY_PATTERNS)
    )
    is_historical = _check_pattern(text_lower, start, end, HISTORICAL_PATTERNS)

    return {
        "isNegated": is_negated,
        "isFamily": is_family,
        "isHistorical": is_historical,
        "confidence": 0.85,
        "evidence": text[max(0, start - 30):end + 30],
        "source": "rule",
    }
