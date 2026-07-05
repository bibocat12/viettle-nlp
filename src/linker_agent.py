import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))


def retrieve_candidates(mention_text: str, mention_type: str, kb, context: str = "") -> list:
    """Retrieve top-k candidates from ICD-10 or RxNorm."""
    candidates = []

    if mention_type == "CHẨN_ĐOÁN":
        results = kb.search_icd10(mention_text, top_k=8)
        for r in results:
            candidates.append({
                "code": r["code"],
                "label": r.get("name", ""),
                "score": r.get("score", 0),
            })
    elif mention_type == "THUỐC":
        # Extract ingredient
        ingredient = re.sub(r"\d+(?:\.\d+)?\s*(?:mg|ml|mcg|IU|U)\b", "", mention_text, flags=re.IGNORECASE).strip()
        query = ingredient if ingredient else mention_text
        results = kb.search_rxnorm(query, top_k=8)
        for r in results:
            candidates.append({
                "code": r["rxcui"],
                "label": r.get("name", ""),
                "score": r.get("score", 0),
            })

    return candidates


def link_candidates(mention: dict, context: str, kb, llm=None) -> dict:
    """Link mention to ICD-10 or RxNorm candidates.

    Uses LLM for reranking when candidates are close.
    """
    mtype = mention.get("type", "")
    text = mention.get("text", "")

    if mtype not in ("CHẨN_ĐOÁN", "THUỐC"):
        return {"candidates": [], "confidence": 1.0}

    # Retrieve candidates
    candidates = retrieve_candidates(text, mtype, kb, context)

    if not candidates:
        return {"candidates": [], "confidence": 0.0}

    # If only one candidate, return it
    if len(candidates) == 1:
        return {"candidates": [candidates[0]["code"]], "confidence": candidates[0]["score"]}

    # Use LLM for reranking if available and candidates are close
    if llm and llm.is_available() and len(candidates) >= 2:
        # Check if top candidates are close
        top_score = candidates[0]["score"]
        second_score = candidates[1]["score"]
        if top_score - second_score < 0.1:
            return _llm_rerank(mention, context, candidates, mtype, kb, llm)

    # Default: return top 3
    return {"candidates": [c["code"] for c in candidates[:3]], "confidence": candidates[0]["score"]}


def _llm_rerank(mention: dict, context: str, candidates: list, mtype: str, kb, llm) -> dict:
    """Use LLM to rerank candidates."""
    # Build candidate descriptions
    cand_desc = []
    for c in candidates[:6]:
        if mtype == "CHẨN_ĐOÁN":
            info = kb.get_icd10_code(c["code"])
            name = info.get("vietnamese_name", "") or info.get("english_name", "") or c.get("label", "")
        else:
            info = kb.get_rxnorm_concept(c["code"])
            name = info.get("prefLabel", "") or c.get("label", "")
        cand_desc.append(f"- {c['code']}: {name[:50]} (score: {c['score']:.2f})")

    system = f"""Select best code(s) from candidates for this {mtype} mention.
Return JSON: {{"selected": ["code1", "code2"], "reason": "brief"}}"""

    user = f"""Mention: "{mention['text']}"
Context: "{context[:300]}"
Type: {mtype}

Candidates:
{chr(10).join(cand_desc)}"""

    response = llm.generate(system, user, 200)
    result = llm.parse_json(response)

    if isinstance(result, dict) and "selected" in result:
        selected = result["selected"]
        if isinstance(selected, list):
            valid = [c for c in selected if any(cand["code"] == c for cand in candidates)]
            if valid:
                return {"candidates": valid[:5], "confidence": 0.85, "rationale": result.get("reason", "")}

    return {"candidates": [c["code"] for c in candidates[:3]], "confidence": candidates[0]["score"]}
