"""
Position Fix: Re-locate entity positions in original text after LLM extraction.

LLMs hallucinate character offsets. This module finds the correct position
by searching entity text in the original document.
"""
import re
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

    if (
        isinstance(original_pos, list)
        and len(original_pos) == 2
        and 0 <= original_pos[0] < original_pos[1] <= len(text)
    ):
        current = text[original_pos[0]:original_pos[1]]
        if current == entity or current.lower() == entity.lower():
            return {
                "position": original_pos,
                "text": current,
                "changed": current != entity,
            }

    # Choose the occurrence nearest to the model/source position. This keeps
    # repeated mentions distinct instead of moving all of them to the first.
    matches = list(re.finditer(re.escape(entity), text, re.IGNORECASE))
    if matches:
        anchor = original_pos[0] if isinstance(original_pos, list) and len(original_pos) == 2 else 0
        match = min(matches, key=lambda item: abs(item.start() - anchor))
        return {
            "position": [match.start(), match.end()],
            "text": text[match.start():match.end()],
            "changed": True,
        }

    # Tolerate whitespace changes, but do not fall back to a single token.
    tokens = entity.split()
    if len(tokens) >= 2:
        pattern = r"\s+".join(re.escape(token) for token in tokens)
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            anchor = original_pos[0] if isinstance(original_pos, list) and len(original_pos) == 2 else 0
            match = min(matches, key=lambda item: abs(item.start() - anchor))
            return {
                "position": [match.start(), match.end()],
                "text": text[match.start():match.end()],
                "changed": True,
            }

    return None
