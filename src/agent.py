import re
from typing import List, Dict, Optional

from llm import get_llm


AGENT_SYSTEM = """You are a Vietnamese medical NLP agent reviewer. You review uncertain extraction cases and make final decisions.

You receive:
- Original sentence with the mention highlighted
- The extracted mention and its type
- Rule-based output
- Model-based output (if available)
- Candidate ICD-10 or RxNorm codes
- Confidence scores

You must return a JSON object with your final decision:
{
  "final_type": "one of: THUỐC, TRIỆU_CHỨNG, CHẨN_ĐOÁN, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM",
  "assertions": {"isNegated": bool, "isFamily": bool, "isHistorical": bool},
  "selected_candidates": ["code1", "code2"] or [],
  "reasoning": "brief explanation"
}

Rules:
1. You may ONLY select candidates from the provided pool
2. You may ONLY use the 5 allowed types
3. You may ONLY use the 3 assertion flags
4. If uncertain, prefer the safer option (fewer false positives)"""

AGENT_USER = """Review this uncertain extraction:

Sentence: "{sentence}"
Mention: "{mention}"
Type: {type}
Confidence: {confidence}

Rule output: {rule_output}
Model output: {model_output}

Candidate codes:
{candidates}

Make your final decision as JSON:"""


def needs_review(m: Dict) -> bool:
    confidence = m.get("confidence", 1.0)
    if confidence < 0.7:
        return True
    assertions = m.get("assertions", {})
    neg = assertions.get("isNegated", False)
    fam = assertions.get("isFamily", False)
    hist = assertions.get("isHistorical", False)
    if fam and hist:
        return True
    if neg and (fam or hist):
        return True
    candidates = m.get("candidates", [])
    if len(candidates) >= 2:
        return True
    return False


def _format_candidates(candidates: list, kb, code_type: str) -> str:
    lines = []
    for c in candidates[:6]:
        if code_type == "icd10":
            info = kb.get_icd10_code(c)
            name = info.get("vietnamese_name", "") or info.get("english_name", "") if info else ""
            lines.append(f"- {c}: {name}")
        elif code_type == "rxnorm":
            info = kb.get_rxnorm_concept(c)
            name = info.get("prefLabel", "") if info else ""
            lines.append(f"- {c}: {name}")
    return "\n".join(lines) if lines else "No candidates"


def review_mention(m: Dict, full_text: str, kb) -> Optional[Dict]:
    ctx_start = max(0, m["position"][0] - 200)
    ctx_end = min(len(full_text), m["position"][1] + 200)
    sentence = full_text[ctx_start:ctx_end]

    mtype = m.get("type", "")
    code_type = "icd10" if mtype == "CHẨN_ĐOÁN" else "rxnorm" if mtype == "THUỐC" else ""
    candidates = m.get("candidates", [])
    candidates_str = _format_candidates(candidates, kb, code_type) if candidates else "No candidates"

    llm = get_llm()
    user_prompt = AGENT_USER.format(
        sentence=sentence[:400],
        mention=m.get("text", ""),
        type=mtype,
        confidence=m.get("confidence", 0),
        rule_output=m.get("source", ""),
        model_output="N/A",
        candidates=candidates_str,
    )
    result = llm.generate_json(AGENT_SYSTEM, user_prompt, max_new_tokens=300)

    if not isinstance(result, dict):
        return None

    valid_types = {"THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}
    final_type = result.get("final_type", "")
    if final_type not in valid_types:
        return None

    assertions = result.get("assertions", {})
    selected = result.get("selected_candidates", [])
    if isinstance(selected, list):
        selected = [c for c in selected if c in candidates]
    else:
        selected = candidates[:3]

    return {
        "type": final_type,
        "assertions": {
            "isNegated": bool(assertions.get("isNegated", False)),
            "isFamily": bool(assertions.get("isFamily", False)),
            "isHistorical": bool(assertions.get("isHistorical", False)),
        },
        "candidates": selected[:5],
        "reviewed_by_agent": True,
        "agent_reasoning": result.get("reasoning", ""),
    }


def review_mentions(mentions: List[Dict], full_text: str, kb) -> List[Dict]:
    reviewed = []
    for m in mentions:
        if needs_review(m):
            result = review_mention(m, full_text, kb)
            if result:
                m.update(result)
        reviewed.append(m)
    return reviewed
