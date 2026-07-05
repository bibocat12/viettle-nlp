import re
from typing import List, Dict, Optional

from llm import get_llm


ASSERTION_SYSTEM = """You are a Vietnamese clinical NLP expert. Your task is to determine the correct assertions for a medical mention in context.

You must classify these 3 boolean assertions:
1. isNegated: The mention is negated (e.g., "không đau ngực" -> pain is negated)
2. isFamily: The mention refers to family history (e.g., "bố bị tiểu đường" -> family history)
3. isHistorical: The mention is historical/past (e.g., "tiền sử tăng huyết áp" -> historical)

Rules:
- Look at the FULL sentence context, not just adjacent words
- "không" before a symptom usually negates it
- "tiền sử" or "trước đây" usually indicates historical
- "bố", "mẹ", "người nhà", "gia đình" usually indicates family history
- If unsure, set the assertion to false (do not guess)

Return a JSON object with exactly these 3 keys:
{"isNegated": true/false, "isFamily": true/false, "isHistorical": true/false}"""

ASSERTION_USER = """Determine assertions for this medical mention:

Mention: "{mention}"
Type: {type}
Context sentence: "{context}"

Return JSON:"""


def check_assertions_llm(mention: str, mention_type: str, context: str) -> Optional[Dict]:
    llm = get_llm()
    user_prompt = ASSERTION_USER.format(
        mention=mention,
        type=mention_type,
        context=context[:500]
    )
    result = llm.generate_json(ASSERTION_SYSTEM, user_prompt, max_new_tokens=100)
    if isinstance(result, dict):
        return {
            "isNegated": bool(result.get("isNegated", False)),
            "isFamily": bool(result.get("isFamily", False)),
            "isHistorical": bool(result.get("isHistorical", False)),
        }
    return None


def is_uncertain_assertion(m: Dict) -> bool:
    assertions = m.get("assertions", {})
    neg = assertions.get("isNegated", False)
    fam = assertions.get("isFamily", False)
    hist = assertions.get("isHistorical", False)
    if fam and hist:
        return True
    if neg and (fam or hist):
        return True
    return False


def refine_assertions_llm(mentions: List[Dict], full_text: str) -> List[Dict]:
    for m in mentions:
        if not is_uncertain_assertion(m):
            continue
        ctx_start = max(0, m["position"][0] - 150)
        ctx_end = min(len(full_text), m["position"][1] + 150)
        context = full_text[ctx_start:ctx_end]
        result = check_assertions_llm(
            m.get("text", ""),
            m.get("type", ""),
            context
        )
        if result is not None:
            m["assertions"] = result
            m["assertion_refined_by_llm"] = True
    return mentions
