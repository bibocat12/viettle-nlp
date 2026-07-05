import re
from typing import List, Dict, Optional

from llm import get_llm


RERANK_ICD10_SYSTEM = """You are a Vietnamese medical coding expert. Your task is to select the best ICD-10 diagnosis code(s) from a candidate list.

Rules:
1. You may select ZERO, ONE, or MORE codes from the candidate pool
2. You may ONLY select codes that appear in the candidate_pool
3. NEVER invent new codes
4. Consider the mention text, context, and code descriptions
5. If none of the candidates match well, return an empty list

Return a JSON object: {"selected_codes": ["code1", "code2", ...], "reasoning": "brief explanation"}"""

RERANK_ICD10_USER = """Select the best ICD-10 code(s) for this diagnosis:

Mention: "{mention}"
Context: "{context}"
Type: {type}

Candidate codes:
{candidates}

Return JSON:"""


RERANK_RXNORM_SYSTEM = """You are a Vietnamese pharmaceutical coding expert. Your task is to select the best RxNorm medication code(s) from a candidate list.

Rules:
1. You may select ZERO, ONE, or MORE codes from the candidate pool
2. You may ONLY select codes that appear in the candidate_pool
3. NEVER invent new codes
4. Consider the drug name, dosage, route, and context
5. If none of the candidates match well, return an empty list

Return a JSON object: {"selected_codes": ["rxcui1", "rxcui2", ...], "reasoning": "brief explanation"}"""

RERANK_RXNORM_USER = """Select the best RxNorm code(s) for this medication:

Mention: "{mention}"
Context: "{context}"

Candidate codes:
{candidates}

Return JSON:"""


def _format_candidates(candidates: list, kb, code_type: str) -> str:
    lines = []
    for c in candidates[:8]:
        if code_type == "icd10":
            info = kb.get_icd10_code(c)
            name = info.get("vietnamese_name", "") or info.get("english_name", "") if info else ""
            lines.append(f"- {c}: {name}")
        elif code_type == "rxnorm":
            info = kb.get_rxnorm_concept(c)
            name = info.get("prefLabel", "") if info else ""
            lines.append(f"- {c}: {name}")
    return "\n".join(lines)


def rerank_icd10(mention: str, context: str, candidates: list, kb) -> list:
    if not candidates:
        return []
    if len(candidates) == 1:
        return candidates

    llm = get_llm()
    candidates_str = _format_candidates(candidates, kb, "icd10")
    user_prompt = RERANK_ICD10_USER.format(
        mention=mention,
        context=context[:300],
        type="CHẨN_ĐOÁN",
        candidates=candidates_str,
    )
    result = llm.generate_json(RERANK_ICD10_SYSTEM, user_prompt, max_new_tokens=200)

    if isinstance(result, dict) and "selected_codes" in result:
        selected = result["selected_codes"]
        if isinstance(selected, list):
            valid = [c for c in selected if c in candidates]
            if valid:
                return valid[:5]

    return candidates[:3]


def rerank_rxnorm(mention: str, context: str, candidates: list, kb) -> list:
    if not candidates:
        return []
    if len(candidates) == 1:
        return candidates

    llm = get_llm()
    candidates_str = _format_candidates(candidates, kb, "rxnorm")
    user_prompt = RERANK_RXNORM_USER.format(
        mention=mention,
        context=context[:300],
        candidates=candidates_str,
    )
    result = llm.generate_json(RERANK_RXNORM_SYSTEM, user_prompt, max_new_tokens=200)

    if isinstance(result, dict) and "selected_codes" in result:
        selected = result["selected_codes"]
        if isinstance(selected, list):
            valid = [c for c in selected if c in candidates]
            if valid:
                return valid[:5]

    return candidates[:3]


def rerank_mentions(mentions: List[Dict], full_text: str, kb) -> List[Dict]:
    for m in mentions:
        mtype = m.get("type", "")
        candidates = m.get("candidates", [])
        if not candidates:
            continue

        ctx_start = max(0, m["position"][0] - 150)
        ctx_end = min(len(full_text), m["position"][1] + 150)
        context = full_text[ctx_start:ctx_end]
        text = m.get("canonical_text", m.get("text", ""))

        if mtype == "CHẨN_ĐOÁN":
            m["candidates"] = rerank_icd10(text, context, candidates, kb)
        elif mtype == "THUỐC":
            m["candidates"] = rerank_rxnorm(text, context, candidates, kb)

    return mentions
