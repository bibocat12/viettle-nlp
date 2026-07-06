"""
Position Fix: Re-locate entity positions in original text after LLM extraction.

LLMs hallucinate character offsets. This module finds the correct position
by searching entity text in the original document.
"""
import re
from difflib import SequenceMatcher
from typing import Dict, List


def fix_positions(mentions: List[Dict], original_text: str) -> List[Dict]:
    """Fix entity positions by re-locating entity text in original input."""
    fixed = []
    for m in mentions:
        entity = m.get("text", "").strip()
        if not entity:
            continue

        original_pos = m.get("position", [0, 0])
        original_len = original_pos[1] - original_pos[0]

        result = _relocate(entity, original_text, original_pos)
        if result:
            m["position"] = result["position"]
            m["text"] = result["text"]
            m["position_fixed"] = result["changed"]
        fixed.append(m)
    return fixed


def _relocate(entity: str, text: str, original_pos: List[int]) -> Dict:
    """Find best position for entity in text."""
    entity = entity.strip()
    if not entity or len(entity) < 2:
        return {"position": original_pos, "text": entity, "changed": False}

    # Strategy 1: Exact match
    idx = text.find(entity)
    if idx >= 0:
        return {"position": [idx, idx + len(entity)], "text": entity, "changed": idx != original_pos[0]}

    # Strategy 2: Exact match (case-insensitive)
    idx = text.lower().find(entity.lower())
    if idx >= 0:
        actual = text[idx:idx + len(entity)]
        return {"position": [idx, idx + len(actual)], "text": actual, "changed": True}

    # Strategy 3: Fuzzy — split tokens, find best span
    tokens = entity.split()
    if len(tokens) >= 2:
        first_word = re.escape(tokens[0])
        last_word = re.escape(tokens[-1])

        starts = [m.start() for m in re.finditer(first_word, text, re.IGNORECASE)]
        for s in starts:
            search_end = min(s + len(entity) * 2, len(text))
            ends = [m.start() for m in re.finditer(last_word, text[s:search_end], re.IGNORECASE)]
            for e in ends:
                end_abs = s + e + len(tokens[-1])
                candidate = text[s:end_abs]
                score = SequenceMatcher(None, entity.lower(), candidate.lower()).ratio()
                if score > 0.6:
                    return {"position": [s, end_abs], "text": candidate, "changed": True}

    # Strategy 4: Just find first token
    first = re.escape(tokens[0]) if tokens else re.escape(entity[:3])
    m = re.search(first, text, re.IGNORECASE)
    if m:
        return {"position": [m.start(), m.end()], "text": text[m.start():m.end()], "changed": True}

    return {"position": original_pos, "text": entity, "changed": False}
