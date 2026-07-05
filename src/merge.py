import re
from typing import List, Dict


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    if inter_start >= inter_end:
        return 0.0
    inter = inter_end - inter_start
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def _normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def _dedup(mentions: List[Dict]) -> List[Dict]:
    if not mentions:
        return []
    mentions.sort(key=lambda m: (m["position"][0], -(m["position"][1] - m["position"][0])))
    result = []
    for m in mentions:
        merged = False
        for r in result:
            ov = _overlap(m["position"][0], m["position"][1], r["position"][0], r["position"][1])
            if ov > 0.6:
                if m.get("confidence", 0) > r.get("confidence", 0):
                    r.update(m)
                merged = True
                break
        if not merged:
            result.append(m)
    return result


def _enforce_type_constraints(mentions: List[Dict]) -> List[Dict]:
    for m in mentions:
        t = m.get("type", "")
        if t not in ("THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"):
            m["type"] = "TRIỆU_CHỨNG"
    return mentions


def _resolve_conflicts(mentions: List[Dict]) -> List[Dict]:
    if not mentions:
        return []
    mentions.sort(key=lambda m: m["position"][0])
    result = []
    for m in mentions:
        conflict = False
        for r in result:
            ov = _overlap(m["position"][0], m["position"][1], r["position"][0], r["position"][1])
            if ov > 0.3 and m.get("type") != r.get("type"):
                if m.get("confidence", 0) > r.get("confidence", 0):
                    result.remove(r)
                    result.append(m)
                conflict = True
                break
        if not conflict:
            result.append(m)
    return result


def merge_mentions(rules: List[Dict], ner: List[Dict]) -> List[Dict]:
    all_mentions = []
    for m in rules:
        m["source"] = m.get("source", "rule")
        all_mentions.append(m)
    for m in ner:
        m["source"] = m.get("source", "qwen_ner")
        all_mentions.append(m)
    all_mentions = _dedup(all_mentions)
    all_mentions = _resolve_conflicts(all_mentions)
    all_mentions = _enforce_type_constraints(all_mentions)
    all_mentions.sort(key=lambda m: m["position"][0])
    return all_mentions
