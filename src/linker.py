import re
from typing import List, Dict, Optional

from rapidfuzz import fuzz


def _normalize_query(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"[^\w\sàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳýỹỷỵ]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _fuzzy_match(query: str, candidates: list, threshold: int = 70) -> list:
    results = []
    q = _normalize_query(query)
    for cand in candidates:
        name = _normalize_query(cand["name"])
        ratio = fuzz.ratio(q, name)
        partial = fuzz.partial_ratio(q, name)
        score = max(ratio, partial) / 100.0
        if score >= threshold / 100.0:
            results.append({**cand, "score": score})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def link_icd10(mention_text: str, kb, context: str = "", top_k: int = 5) -> List[Dict]:
    direct = kb.search_icd10(mention_text, top_k=top_k)
    if direct and direct[0]["score"] >= 0.95:
        return direct[:top_k]

    candidates = []
    for code, info in kb.icd10.items():
        name = info.get("vietnamese_name", "") or info.get("english_name", "")
        if name:
            candidates.append({"code": code, "name": name})

    fuzzy = _fuzzy_match(mention_text, candidates, threshold=60)
    seen_codes = set()
    merged = []
    for r in direct:
        if r["code"] not in seen_codes:
            seen_codes.add(r["code"])
            merged.append(r)
    for r in fuzzy:
        if r["code"] not in seen_codes:
            seen_codes.add(r["code"])
            merged.append(r)
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:top_k]


def link_rxnorm(mention_text: str, kb, context: str = "", top_k: int = 5) -> List[Dict]:
    ingredient = _extract_drug_ingredient(mention_text)
    queries = [mention_text]
    if ingredient and ingredient != mention_text:
        queries.insert(0, ingredient)

    all_results = []
    seen_cuis = set()
    for q in queries:
        direct = kb.search_rxnorm(q, top_k=top_k)
        for r in direct:
            if r["rxcui"] not in seen_cuis:
                seen_cuis.add(r["rxcui"])
                all_results.append(r)

    candidates = []
    for cui, info in kb.rxnorm_concepts.items():
        name = info["prefLabel"]
        if name:
            candidates.append({"rxcui": cui, "name": name})

    for q in queries:
        fuzzy = _fuzzy_match(q, candidates, threshold=65)
        for r in fuzzy:
            if r["rxcui"] not in seen_cuis:
                seen_cuis.add(r["rxcui"])
                all_results.append(r)

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]


def _extract_drug_ingredient(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\d+(?:\.\d+)?\s*(?:mg|ml|g|mcg|IU|U)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:po|iv|im|sc|sl|tiêm|uống|đặt|nhỏ|bôi|thoa)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:bid|tid|qd|qid|prn|x\s*\d+|mỗi ngày|ngày)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def link_mentions(mentions: List[Dict], kb) -> List[Dict]:
    for m in mentions:
        mtype = m.get("type", "")
        text = m.get("text", "")
        ctx_start = max(0, m["position"][0] - 100)
        ctx_end = m["position"][1] + 100
        context = ""

        if mtype == "CHẨN_ĐOÁN":
            results = link_icd10(text, kb, context, top_k=5)
            m["candidates"] = [r["code"] for r in results[:5]]
        elif mtype == "THUỐC":
            results = link_rxnorm(text, kb, context, top_k=5)
            m["candidates"] = [r["rxcui"] for r in results[:5]]
        else:
            m["candidates"] = []
    return mentions
