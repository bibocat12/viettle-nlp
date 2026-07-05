import re
from typing import List, Dict

NEGATION_PATTERNS = [
    (r"không\b", 5),
    (r"chưa\b", 5),
    (r"không có\b", 8),
    (r"chưa có\b", 8),
    (r"không ghi nhận\b", 15),
    (r"không có gì\b", 12),
    (r"không có triệu chứng\b", 20),
    (r"bình thường\b", 12),
    (r"không có bất thường\b", 20),
    (r"không có gì đáng\b", 18),
    (r"không có cải thiện\b", 18),
    (r"không liên quan\b", 15),
    (r"không cảm giác\b", 15),
    (r"không buồn nôn\b", 15),
    (r"không đổ mồ hôi\b", 15),
    (r"không mất kiểm soát\b", 18),
    (r"không có khó chịu\b", 18),
    (r"không đánh trống ngực\b", 20),
    (r"không quầng sáng\b", 15),
    (r"không chóng mặt\b", 15),
    (r"không có biểu hiện\b", 18),
    (r"không có quá trình\b", 18),
    (r"không phàn nàn\b", 15),
    (r"không có gì bất thường\b", 22),
    (r"không có gì đáng chú ý\b", 22),
    (r"không có gì đáng\b", 18),
    (r"không sốt\b", 8),
    (r"không ho\b", 8),
]

FAMILY_PATTERNS = [
    (r"bố\b", 3),
    (r"mẹ\b", 3),
    (r"cha\b", 3),
    (r"anh\b", 3),
    (r"chị\b", 3),
    (r"em\b", 2),
    (r"ông\b", 3),
    (r"bà\b", 3),
    (r"người nhà\b", 8),
    (r"gia đình\b", 8),
    (r"gia đình có\b", 11),
    (r"tiền sử gia đình\b", 18),
    (r"theo lời\b", 8),
    (r"theo lời bệnh nhân\b", 18),
]

HISTORICAL_PATTERNS = [
    (r"tiền sử\b", 8),
    (r"trước đây\b", 10),
    (r"trước khi\b", 10),
    (r"trước nhập viện\b", 16),
    (r"trước đó\b", 10),
    (r"đã điều trị\b", 12),
    (r"đã dùng\b", 8),
    (r"đã chỉ định\b", 13),
    (r"đã sử dụng\b", 12),
    (r"được chỉ định\b", 15),
    (r"các bệnh đã điều trị\b", 22),
    (r"các bệnh lý mãn tính\b", 22),
    (r"bệnh lý nội khoa mạn\b", 22),
    (r"các tập kinh\b", 12),
    (r"tái phát\b", 8),
    (r"tiền sử bệnh\b", 14),
    (r"bệnh sử\b", 8),
]


def _check_pattern(text_lower: str, start: int, end: int, patterns: list, window: int = 60) -> bool:
    ctx_start = max(0, start - window)
    ctx_end = min(len(text_lower), end + window)
    prefix = text_lower[ctx_start:start]
    suffix = text_lower[end:ctx_end]
    for pattern, _ in patterns:
        if re.search(pattern, prefix):
            return True
        if re.search(pattern, suffix):
            return False
    return False


def classify_assertions(text: str, mentions: List[Dict], window: int = 40) -> List[Dict]:
    text_lower = text.lower()
    for m in mentions:
        start, end = m["position"][0], m["position"][1]
        ctx_start = max(0, start - window)
        ctx_end = min(len(text_lower), end + window)
        prefix = text_lower[ctx_start:start]
        suffix = text_lower[end:ctx_end]

        is_negated = False
        is_family = False
        is_historical = False

        for pattern, _ in NEGATION_PATTERNS:
            if re.search(pattern, prefix):
                is_negated = True
                break

        for pattern, _ in FAMILY_PATTERNS:
            if re.search(pattern, prefix):
                is_family = True
                break

        for pattern, _ in HISTORICAL_PATTERNS:
            if re.search(pattern, prefix):
                is_historical = True
                break

        if "assertions" not in m:
            m["assertions"] = {}
        existing = m.get("assertions", {})
        m["assertions"] = {
            "isNegated": existing.get("isNegated", False) or is_negated,
            "isFamily": existing.get("isFamily", False) or is_family,
            "isHistorical": existing.get("isHistorical", False) or is_historical,
        }
    return mentions
