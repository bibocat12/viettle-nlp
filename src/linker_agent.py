import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from prompt import LINKER_RERANK_SYSTEM, LINKER_AGENT_SYSTEM


def _search_icd10(mention_text: str, kb, top_k: int = 15) -> list:
    # Fuzzy string search (fast, default)
    results = kb.search_icd10(mention_text, top_k=top_k)
    return [{"code": r["code"], "name": r.get("name", ""), "score": r.get("score", 0)} for r in results]


def _search_rxnorm(mention_text: str, kb, top_k: int = 15) -> list:
    ingredient = re.sub(r"\d+(?:\.\d+)?\s*(?:mg|ml|mcg|IU|U)\b", "", mention_text, flags=re.IGNORECASE).strip()
    query = ingredient if ingredient else mention_text
    results = kb.search_rxnorm(query, top_k=top_k)
    return [{"code": r["rxcui"], "name": r.get("name", ""), "score": r.get("score", 0)} for r in results]


def _embedding_rerank(mention_text: str, mtype: str, candidates: list, kb) -> list:
    try:
        from embedding_linking import rerank_icd10, rerank_rxnorm
    except ImportError:
        return candidates[:5]
    if not candidates:
        return []
    if mtype == "CHẨN_ĐOÁN":
        results = rerank_icd10(mention_text, [{"code": c["code"]} for c in candidates], kb, top_k=5)
        # Merge original names
        lookup = {c["code"]: c["name"] for c in candidates}
        return [{"code": r["code"], "name": lookup.get(r["code"], r.get("label", "")), "score": r["score"]} for r in results]
    elif mtype == "THUỐC":
        results = rerank_rxnorm(mention_text, [{"rxcui": c["code"]} for c in candidates], kb, top_k=5)
        lookup = {c["code"]: c["name"] for c in candidates}
        return [{"code": r["rxcui"], "name": lookup.get(r["rxcui"], r.get("label", "")), "score": r["score"]} for r in results]
    return candidates[:5]


def link_candidates(mention: dict, context: str, kb, llm=None, max_steps: int = 5) -> dict:
    """Agentic linker: fuzzy search → embedding rerank → LLM picks → loop if needed."""
    mtype = mention.get("type", "")
    text = mention.get("text", "")

    if mtype not in ("CHẨN_ĐOÁN", "THUỐC"):
        return {"candidates": [], "confidence": 1.0}

    # Step 1: Fuzzy search
    candidates = _search_icd10(text, kb) if mtype == "CHẨN_ĐOÁN" else _search_rxnorm(text, kb)
    if not candidates:
        return {"candidates": [], "confidence": 0.0}

    # Step 2: Embedding rerank (ViPubMedDeBERTa)
    reranked = _embedding_rerank(text, mtype, candidates, kb)
    top = reranked[:5] if reranked else candidates[:3]

    # Step 3: LLM picks (single call, agentic loop if unsure)
    if llm and llm.is_available():
        result = _llm_pick_or_loop(text, context, top, mtype, kb, llm, max_steps)
        if result:
            return result

    return {"candidates": [c["code"] for c in top], "confidence": 0.80}


def _llm_pick_or_loop(mention_text: str, context: str, candidates: list, mtype: str, kb, llm, max_steps: int) -> Optional[dict]:
    """LLM picks from candidates. If uncertain, enters agentic loop."""
    # First try: single LLM call to pick
    cand_desc = "\n".join([f"  {c['code']:15s} | {c.get('name', '')[:50]:50s}" for c in candidates[:8]])

    system = LINKER_RERANK_SYSTEM.format(mtype=mtype)
    user = f"""Mention: "{mention_text}"
Context: "{context[:200]}"
Type: {mtype}

Candidates:
{cand_desc}

Pick the best code(s). Return JSON: {{"selected": ["code1", "code2"], "reason": "brief"}}"""

    response = llm.generate(system, user, 200)
    result = llm.parse_json(response)

    if isinstance(result, dict) and "selected" in result:
        selected = result["selected"]
        if isinstance(selected, list) and len(selected) > 0:
            valid = [c for c in selected if any(cand["code"] == c for cand in candidates)]
            if valid:
                return {"candidates": valid[:5], "confidence": 0.85, "rationale": result.get("reason", "")}

    # If LLM failed to pick, try agentic loop for self-correction
    return _agentic_loop(mention_text, context, candidates, mtype, kb, llm, max_steps)


def _agentic_loop(mention_text: str, context: str, candidates: list, mtype: str, kb, llm, max_steps: int) -> Optional[dict]:
    """Agentic tool-calling loop for complex cases."""
    conversation = []
    system = LINKER_AGENT_SYSTEM.format(
        mtype=mtype, mention=mention_text, context=context[:300], tools=TOOL_DESCRIPTIONS,
    )

    seed = "\n".join([f"  {c['code']:15s} | {c.get('name', '')[:50]:50s} | score={c.get('score', 0):.3f}" for c in candidates[:8]])
    conversation.append(f"Initial search results:\n{seed}")

    for step in range(max_steps):
        user_prompt = "\n".join(conversation)
        response = llm.generate(system, user_prompt, 400)
        parsed = llm.parse_json(response)

        if not isinstance(parsed, dict):
            continue

        if parsed.get("final"):
            selected = parsed.get("candidates", [])
            if isinstance(selected, list) and len(selected) > 0:
                return {"candidates": selected[:5], "confidence": 0.85, "rationale": parsed.get("reason", ""), "steps": step + 1}

        tool_name = parsed.get("tool", "")
        args = parsed.get("args", {})

        if tool_name == "search_icd10":
            results = _search_icd10(args.get("mention_text", mention_text), kb, top_k=10)
        elif tool_name == "search_rxnorm":
            results = _search_rxnorm(args.get("mention_text", mention_text), kb, top_k=10)
        elif tool_name == "embedding_search":
            results = _embedding_rerank(args.get("mention_text", mention_text), mtype, candidates, kb)
        elif tool_name == "get_code_info":
            code = args.get("code", "")
            if mtype == "CHẨN_ĐOÁN":
                info = kb.get_icd10_code(code)
                results = [{"code": code, "name": info.get("vietnamese_name", "") or info.get("english_name", "") if info else ""}]
            else:
                info = kb.get_rxnorm_concept(code)
                results = [{"code": code, "name": info.get("prefLabel", "") if info else ""}]
        else:
            results = None

        if results:
            output = "\n".join([f"  {r['code']:15s} | {r.get('name', '')[:50]:50s}" for r in results[:6]])
            conversation.append(f"> Tool {tool_name} result:\n{output}")
        else:
            conversation.append(f"> Tool '{tool_name}' not found. Available: search_icd10, search_rxnorm, embedding_search, get_code_info")

    return None  # Agentic loop failed, caller falls back to defaults


TOOL_DESCRIPTIONS = """
- search_icd10(mention_text): Tìm mã ICD-10
- search_rxnorm(mention_text): Tìm mã RxNorm
- embedding_search(mention_text, candidates): Rerank bằng semantic embedding
- get_code_info(code): Xem thông tin mã
"""
