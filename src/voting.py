"""
Model Voting: Combines Qwen + ViHealthBERT + Rules with confidence scoring.

Each model extracts entities independently. The voting mechanism:
1. Groups entities by position overlap
2. Scores confidence based on how many models agree
3. LLM resolves conflicts when models disagree
"""
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional
from collections import defaultdict


MODEL_WEIGHTS = {
    "qwen": 0.40,      # Best recall, may hallucinate
    "rules": 0.35,     # High precision for structured patterns
    "healthbert": 0.25, # Fast keyword matching, lower precision
}

VOTE_CONFIDENCE = {
    3: 0.95,  # All 3 agree
    2: 0.80,  # 2 out of 3 agree
    1: 0.50,  # Only 1 model found it
}


def group_mentions(all_mentions: List[Dict]) -> List[List[Dict]]:
    """Group mentions from different models by position overlap."""
    if not all_mentions:
        return []

    # Sort by position
    sorted_mentions = sorted(all_mentions, key=lambda m: (m["position"][0], m["position"][1]))

    groups = []
    used = set()

    for i, m in enumerate(sorted_mentions):
        if i in used:
            continue

        group = [m]
        used.add(i)
        s_i, e_i = m["position"]

        for j, n in enumerate(sorted_mentions):
            if j in used:
                continue
            s_j, e_j = n["position"]

            # Check overlap: positions within 3 chars or overlapping
            overlap = min(e_i, e_j) - max(s_i, s_j)
            if overlap > 0 or abs(s_i - s_j) < 3:
                group.append(n)
                used.add(j)
                # Expand bounds
                s_i = min(s_i, s_j)
                e_i = max(e_i, e_j)

        groups.append(group)

    return groups


def _text_similarity(t1: str, t2: str) -> float:
    """Compute text similarity for fuzzy matching."""
    return SequenceMatcher(None, t1.lower(), t2.lower()).ratio()


def _merge_group(group: List[Dict]) -> Dict:
    """Merge a group of mentions into one with voting."""
    if not group:
        return None

    if len(group) == 1:
        m = group[0]
        m["vote_count"] = 1
        m["voters"] = [m.get("source", m.get("extraction_source", "unknown"))]
        m["confidence"] = VOTE_CONFIDENCE[1]
        return m

    # Vote on position: use the widest span
    positions = [m["position"] for m in group]
    best_pos = min(p[0] for p in positions), max(p[1] for p in positions)

    # Vote on text: use the most common text variant (original behavior)
    texts = [m.get("text", "") for m in group]
    text_votes = defaultdict(int)
    for t in texts:
        key = t.lower().strip()
        text_votes[key] += 1
    best_text_key = max(text_votes, key=text_votes.get)
    best_text = [t for t in texts if t.lower().strip() == best_text_key][0]

    # Vote on type
    type_votes = defaultdict(int)
    source_votes = defaultdict(int)
    for m in group:
        mtype = m.get("type", m.get("type_candidates", [{}])[0].get("label", "TRIỆU_CHỨNG"))
        type_votes[mtype] += 1
        source = m.get("source", m.get("extraction_source", "unknown"))
        source_votes[source] += 1

    # Winner type = most votes
    best_type = max(type_votes, key=type_votes.get)
    total_votes = len(group)
    max_type_votes = type_votes[best_type]

    # Confidence based on agreement
    if max_type_votes >= 3:
        confidence = VOTE_CONFIDENCE[3]
    elif max_type_votes >= 2:
        confidence = VOTE_CONFIDENCE[2]
    else:
        confidence = VOTE_CONFIDENCE[1]

    return {
        "text": best_text,
        "position": list(best_pos),
        "type": best_type,
        "type_candidates": [{"label": best_type, "score": confidence}],
        "confidence": confidence,
        "vote_count": total_votes,
        "max_type_votes": max_type_votes,
        "voters": dict(source_votes),
        "type_votes": dict(type_votes),
        "needs_review": max_type_votes < 2,
    }


def merge_with_voting(all_mentions: List[Dict]) -> List[Dict]:
    """Merge mentions from all models with voting-based confidence."""
    groups = group_mentions(all_mentions)
    merged = []

    for g in groups:
        mention = _merge_group(g)
        if mention:
            merged.append(mention)

    # Sort by position
    merged.sort(key=lambda m: m["position"][0])
    return merged


VOTE_REASON_SYSTEM = """Bạn là chuyên gia phân tích kết quả trích xuất thực thể y khoa.
Ba hệ thống khác nhau đã trích xuất cùng một mention với các kết quả khác nhau.
Hãy phân tích và chọn kết quả đúng nhất.

Các loại: THUỐC, TRIỆU_CHỨNG, CHẨN_ĐOÁN, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM

Trả về JSON: {"type": "...", "reason": "..."}"""


def llm_resolve_conflicts(mentions: List[Dict], full_text: str, llm) -> List[Dict]:
    """Use LLM to resolve low-confidence mentions (vote disagreement)."""
    if not llm or not llm.is_available():
        return mentions

    for m in mentions:
        if not m.get("needs_review"):
            continue

        text = m["text"]
        s, e = m["position"]
        ctx = full_text[max(0, s - 80):min(len(full_text), e + 80)]
        type_votes = m.get("type_votes", {})

        user = f"""Mention: "{text}"
Context: "{ctx}"
Type votes: {type_votes}

Chọn loại đúng nhất cho mention này."""

        response = llm.generate(VOTE_REASON_SYSTEM, user, 150)
        result = llm.parse_json(response)

        if isinstance(result, dict) and result.get("type"):
            new_type = result["type"]
            valid = ("THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM")
            if new_type in valid:
                m["type"] = new_type
                m["type_candidates"] = [{"label": new_type, "score": 0.85}]
                m["confidence"] = 0.85
                m["llm_resolved"] = True
                m["reason"] = result.get("reason", "")

    return mentions
