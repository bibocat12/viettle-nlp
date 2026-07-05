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
]

FAMILY_PATTERNS = [
    (r"bố\b", 3), (r"mẹ\b", 3), (r"cha\b", 3), (r"người nhà\b", 8),
    (r"gia đình\b", 8), (r"tiền sử gia đình\b", 18), (r"theo lời\b", 8),
]

HISTORICAL_PATTERNS = [
    (r"tiền sử\b", 8), (r"trước đây\b", 10), (r"trước khi\b", 10),
    (r"trước nhập viện\b", 16), (r"trước đó\b", 10), (r"đã điều trị\b", 12),
    (r"đã dùng\b", 8), (r"đã chỉ định\b", 13), (r"đã sử dụng\b", 12),
    (r"được chỉ định\b", 15), (r"các bệnh đã điều trị\b", 22),
    (r"các bệnh lý mãn tính\b", 22), (r"tái phát\b", 8),
]


def _check_pattern(text_lower: str, start: int, end: int, patterns: list, window: int = 40) -> bool:
    ctx_start = max(0, start - window)
    ctx_end = min(len(text_lower), end + window)
    prefix = text_lower[ctx_start:start]
    for pattern, _ in patterns:
        if re.search(pattern, prefix):
            return True
    return False


def classify_assertions(text: str, mention: dict, llm=None) -> dict:
    """Classify assertions for a single mention.

    Uses rule-based first, then LLM for uncertain cases.
    """
    start, end = mention["position"][0], mention["position"][1]
    text_lower = text.lower()

    # Rule-based
    is_negated = _check_pattern(text_lower, start, end, NEGATION_PATTERNS)
    is_family = _check_pattern(text_lower, start, end, FAMILY_PATTERNS)
    is_historical = _check_pattern(text_lower, start, end, HISTORICAL_PATTERNS)

    # Check if uncertain (conflicting signals)
    uncertain = (is_family and is_historical) or (is_negated and (is_family or is_historical))

    # Use LLM for uncertain cases
    if uncertain and llm and llm.is_available():
        ctx_start = max(0, start - 150)
        ctx_end = min(len(text), end + 150)
        context = text[ctx_start:ctx_end]

        system = """Determine assertions for this Vietnamese medical mention.
- isNegated: negated (không, chưa)
- isFamily: family history (bố, mẹ, người nhà)
- isHistorical: past/historical (tiền sử, trước đây)

Return JSON: {"isNegated": bool, "isFamily": bool, "isHistorical": bool}"""

        user = f"Mention: \"{mention['text']}\"\nType: {mention.get('type', '')}\nContext: \"{context}\""
        response = llm.generate(system, user, 100)
        result = llm.parse_json(response)
        if isinstance(result, dict) and "isNegated" in result:
            return {
                "isNegated": bool(result.get("isNegated", False)),
                "isFamily": bool(result.get("isFamily", False)),
                "isHistorical": bool(result.get("isHistorical", False)),
                "confidence": 0.90,
                "evidence": context[max(0, start - ctx_start - 20):end - ctx_start + 20],
            }

    return {
        "isNegated": is_negated,
        "isFamily": is_family,
        "isHistorical": is_historical,
        "confidence": 0.85 if not uncertain else 0.60,
        "evidence": text[max(0, start - 20):end + 20],
    }
