"""Section parser: splits clinical text into sections by `N. Title` patterns."""

import re
from typing import List, Dict


def parse_sections(text: str) -> List[Dict]:
    """Split text into sections by `N. Title` patterns.
    
    Returns list of:
      {"id": "1", "title": "Tiền sử bệnh", "text": "...", "start": 0, "end": 100}
    """
    pattern = re.compile(r'(?:^|\n)\s*(\d+)\.\s+(.+?)(?=\n\s*\d+\.\s|\Z)', re.DOTALL)
    sections = []
    for m in pattern.finditer(text):
        sec_id = m.group(1).strip()
        title = m.group(2).strip().split('\n')[0]
        # Extract the full section text
        sec_text = m.group(0).strip()
        # Skip metadata-only sections (numbered lists that are part of a section)
        if len(sec_text) < 20 and not any(c.isalpha() for c in sec_text):
            continue
        sections.append({
            "id": sec_id,
            "title": title,
            "text": sec_text,
            "start": m.start(),
            "end": m.end(),
        })
    
    # Fallback: if no sections found, treat whole text as one section
    if not sections and text.strip():
        sections.append({
            "id": "1",
            "title": "Toàn bộ",
            "text": text.strip(),
            "start": 0,
            "end": len(text),
        })
    
    return sections


def get_section_summary(text: str) -> str:
    """Create a brief summary/context for a section."""
    # Extract key: first 2 lines or first 150 chars
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    summary = ' | '.join(lines[:3]) if lines else text[:150]
    if len(summary) > 200:
        summary = summary[:200] + '...'
    return summary
