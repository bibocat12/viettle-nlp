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
    "qwen": 0.45,      # Recall proposer
    "rule": 0.55,      # High precision for structured patterns
    "rules": 0.55,     # Backward-compatible source name
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

            # Only overlapping spans describe the same occurrence. Nearby
            # symptoms/tests must remain separate entities.
            overlap = min(e_i, e_j) - max(s_i, s_j)
            if overlap > 0:
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
        m = dict(group[0])
        m["vote_count"] = 1
        src = m.get("source", m.get("extraction_source", "unknown"))
        m["voters"] = {src: 1}
        m["confidence"] = max(m.get("confidence", 0.0), 0.90 if src in ("rule", "rules") else VOTE_CONFIDENCE[1])
        m["needs_review"] = False
        return m

    # Pick one real source span. Combining min(start)/max(end) with another
    # source's text creates an invalid text-position pair.
    def boundary_score(candidate):
        cs, ce = candidate["position"]
        support = 0.0
        for other in group:
            os, oe = other["position"]
            overlap = max(0, min(ce, oe) - max(cs, os))
            denominator = max(1, min(ce - cs, oe - os))
            if overlap / denominator >= 0.7:
                source = other.get("source", other.get("extraction_source", "unknown"))
                support += MODEL_WEIGHTS.get(source, 0.25)
        return support, ce - cs, candidate.get("confidence", 0.0)

    # Vote on type
    type_votes = defaultdict(float)
    type_sources = defaultdict(set)
    source_votes = defaultdict(int)
    for m in group:
        mtype = m.get("type", m.get("type_candidates", [{}])[0].get("label", "TRIỆU_CHỨNG"))
        source = m.get("source", m.get("extraction_source", "unknown"))
        if source not in type_sources[mtype]:
            type_votes[mtype] += MODEL_WEIGHTS.get(source, 0.25)
            type_sources[mtype].add(source)
        source_votes[source] += 1

    # Winner type = most votes
    best_type = max(type_votes, key=type_votes.get)
    total_votes = len(source_votes)
    max_type_votes = len(type_sources[best_type])

    same_type = [
        mention for mention in group
        if mention.get("type", mention.get("type_candidates", [{}])[0].get("label")) == best_type
    ]
    # Official medication spans include the full order (ingredient, dose,
    # route, frequency). For other types the longest supported real span also
    # avoids creating a synthetic boundary from two partially-overlapping spans.
    boundary = max(same_type or group, key=lambda item: (
        item["position"][1] - item["position"][0], boundary_score(item)
    ))
    best_pos = list(boundary["position"])
    best_text = boundary.get("text", "")

    # Confidence based on agreement
    if max_type_votes >= 3:
        confidence = VOTE_CONFIDENCE[3]
    elif max_type_votes >= 2:
        confidence = VOTE_CONFIDENCE[2]
    else:
        confidence = VOTE_CONFIDENCE[1]

    return {
        "text": best_text,
        "position": best_pos,
        "type": best_type,
        "type_candidates": [{"label": best_type, "score": confidence}],
        "confidence": confidence,
        "vote_count": total_votes,
        "max_type_votes": max_type_votes,
        "voters": dict(source_votes),
        "type_votes": {key: round(value, 3) for key, value in type_votes.items()},
        "needs_review": len(type_votes) > 1,
    }


def merge_with_voting(all_mentions: List[Dict]) -> List[Dict]:
    """Merge mentions from all models with voting-based confidence."""
    # When a generative span contains two high-precision, non-overlapping rule
    # spans of the same type, keep the two atomic mentions. Otherwise the
    # transitive overlap group collapses e.g. "lo âu" + "mất ngủ" back into
    # the model's single "lo âu mất ngủ" span.
    filtered_mentions = []
    for mention in all_mentions:
        source = mention.get("source", mention.get("extraction_source", "unknown"))
        if source == "qwen":
            start, end = mention.get("position", [0, 0])
            mtype = mention.get("type")
            contained_rules = sorted(
                (
                    candidate for candidate in all_mentions
                    if candidate is not mention
                    and candidate.get("source", candidate.get("extraction_source")) in ("rule", "rules")
                    and candidate.get("type") == mtype
                    and start <= candidate.get("position", [0, 0])[0]
                    and candidate.get("position", [0, 0])[1] <= end
                ),
                key=lambda item: item["position"],
            )
            non_overlapping = all(
                left["position"][1] <= right["position"][0]
                for left, right in zip(contained_rules, contained_rules[1:])
            )
            covered = sum(item["position"][1] - item["position"][0] for item in contained_rules)
            if len(contained_rules) >= 2 and non_overlapping and covered / max(1, end - start) >= 0.7:
                continue
        filtered_mentions.append(mention)

    groups = group_mentions(filtered_mentions)
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
