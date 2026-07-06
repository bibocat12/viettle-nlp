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
        results = kb.search_icd10(mention_text, top_k=15)
        for r in results:
            candidates.append({
                "code": r["code"],
                "label": r.get("name", ""),
                "score": r.get("score", 0),
            })
    elif mention_type == "THUỐC":
        ingredient = re.sub(r"\d+(?:\.\d+)?\s*(?:mg|ml|mcg|IU|U)\b", "", mention_text, flags=re.IGNORECASE).strip()
        query = ingredient if ingredient else mention_text
        results = kb.search_rxnorm(query, top_k=15)
        for r in results:
            candidates.append({
                "code": r["rxcui"],
                "label": r.get("name", ""),
                "score": r.get("score", 0),
            })

    return candidates


def _embedding_rerank(candidates: list, mention_text: str, mtype: str, kb) -> list:
    """Rerank using ViPubMedDeBERTa semantic embeddings."""
    try:
        from embedding_linking import rerank_icd10, rerank_rxnorm
    except ImportError:
        return candidates[:5]

    if not candidates:
        return []

    if mtype == "CHẨN_ĐOÁN":
        results = rerank_icd10(mention_text, [{"code": c["code"]} for c in candidates], kb, top_k=5)
        return [r["code"] for r in results]
    elif mtype == "THUỐC":
        results = rerank_rxnorm(mention_text, [{"rxcui": c["code"]} for c in candidates], kb, top_k=5)
        return [r["rxcui"] for r in results]

    return [c["code"] for c in candidates[:5]]


def link_candidates(mention: dict, context: str, kb, llm=None) -> dict:
    """Link mention to ICD-10 or RxNorm candidates.

    Three-stage reranking:
    1. Fuzzy/lexical retrieval (fast)
    2. ViPubMedDeBERTa embedding reranking (semantic)
    3. LLM reranking (for close cases)
    """
    mtype = mention.get("type", "")
    text = mention.get("text", "")

    if mtype not in ("CHẨN_ĐOÁN", "THUỐC"):
        return {"candidates": [], "confidence": 1.0}

    # Stage 1: Retrieve candidates via fuzzy/lexical search
    candidates = retrieve_candidates(text, mtype, kb, context)

    if not candidates:
        return {"candidates": [], "confidence": 0.0}

    # Stage 2: Embedding reranking (ViPubMedDeBERTa)
    reranked = _embedding_rerank(candidates, text, mtype, kb)

    if not reranked:
        return {"candidates": [c["code"] for c in candidates[:3]], "confidence": 0.5}

    # Stage 3: LLM reranking for close cases
    if llm and llm.is_available() and len(reranked) >= 2:
        candidate_objects = [{"code": c} for c in reranked]
        return _llm_rerank(mention, context, candidate_objects, mtype, kb, llm)

    return {"candidates": reranked[:5], "confidence": 0.85}


def _llm_rerank(mention: dict, context: str, candidates: list, mtype: str, kb, llm) -> dict:
    """Use LLM to rerank candidates."""
    cand_desc = []
    for c in candidates[:6]:
        if mtype == "CHẨN_ĐOÁN":
            info = kb.get_icd10_code(c["code"])
            name = info.get("vietnamese_name", "") or info.get("english_name", "") or ""
        else:
            info = kb.get_rxnorm_concept(c["code"])
            name = info.get("prefLabel", "") if info else ""
        cand_desc.append(f"- {c['code']}: {name[:50]}")

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

    return {"candidates": [c["code"] for c in candidates[:3]], "confidence": 0.80}
