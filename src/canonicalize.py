import re
from typing import List, Dict, Optional

from llm import get_llm


CANONICALIZE_SYSTEM = """You are a Vietnamese medical text expert. Your task is to rewrite noisy, abbreviated, or informal Vietnamese medical mentions into their canonical, standard form.

Rules:
1. Expand abbreviations (e.g., "đt" -> "điều trị", "kh" -> "khám")
2. Fix spelling errors
3. Expand shorthand (e.g., "trào ngc d dày-tq" -> "trào ngược dạ dày thực quản")
4. Keep the same meaning, do not add or remove medical content
5. Return ONLY the canonical form, nothing else

Examples:
- "trào ngc d dày-tq" -> "trào ngược dạ dày thực quản"
- "đái tháo đường t2" -> "đái tháo đường típ 2"
- "tăng huyết áp" -> "tăng huyết áp"
- "viêm phổi" -> "viêm phổi"
- "suy tim 3" -> "suy tim độ 3"
"""

CANONICALIZE_USER = """Rewrite this Vietnamese medical mention into its canonical form:

"{mention}"

Context: {context}

Return ONLY the canonical form:"""


def canonicalize_mention(mention: str, context: str = "") -> str:
    llm = get_llm()
    user_prompt = CANONICALIZE_USER.format(mention=mention, context=context[:200])
    response = llm.generate(CANONICALIZE_SYSTEM, user_prompt, max_new_tokens=100)
    response = response.strip().strip('"').strip("'")
    if len(response) < 2 or len(response) > 200:
        return mention
    return response


def canonicalize_mentions(mentions: List[Dict], full_text: str) -> List[Dict]:
    for m in mentions:
        text = m.get("text", "")
        if len(text) < 5:
            continue
        ctx_start = max(0, m["position"][0] - 100)
        ctx_end = min(len(full_text), m["position"][1] + 100)
        context = full_text[ctx_start:ctx_end]
        canonical = canonicalize_mention(text, context)
        if canonical != text:
            m["canonical_text"] = canonical
        else:
            m["canonical_text"] = text
    return mentions
