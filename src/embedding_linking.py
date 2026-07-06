import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return
    model_path = str(Path(__file__).parent.parent / "models" / "vipubmed-deberta-base")
    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    _model = AutoModel.from_pretrained(model_path)
    _model.eval()
    if torch.cuda.is_available():
        _model = _model.cuda()


@torch.no_grad()
def _encode(texts: List[str], max_length: int = 128) -> np.ndarray:
    _load_model()
    if isinstance(texts, str):
        texts = [texts]
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = _tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = _model(**inputs)
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)


def get_embedding(text: str) -> List[float]:
    """Get DeBERTa embedding for a single text."""
    emb = _encode(text)
    return emb[0].tolist()


def rerank_icd10(query: str, candidates: List[Dict], kb, top_k: int = 5) -> List[Dict]:
    """Rerank ICD-10 candidates using semantic embeddings.
    
    Args:
        query: The mention text
        candidates: List of candidate dicts with 'code' key from kb.search_icd10
        kb: KnowledgeBase instance
        
    Returns:
        Reranked list of candidates with updated scores
    """
    if not candidates:
        return []
    
    # Build texts for each candidate
    cand_texts = []
    for c in candidates:
        code = c["code"]
        info = kb.get_icd10_code(code)
        name = info.get("vietnamese_name", "") or info.get("english_name", "") or ""
        # Combine Vietnamese + English for better matching
        en = info.get("english_name", "")
        text = name
        if en and en != name:
            text += " " + en
        cand_texts.append(text if text else code)
    
    # Encode query and candidates
    query_emb = _encode(query)
    cand_embs = _encode(cand_texts)
    
    # Compute cosine similarity
    scores = np.dot(cand_embs, query_emb.T).squeeze()
    if len(cand_texts) == 1:
        scores = np.array([float(scores)])
    
    # Rerank
    results = []
    for i, c in enumerate(candidates):
        results.append({
            "code": c["code"],
            "label": cand_texts[i],
            "score": float(scores[i]),
            "old_score": c.get("score", 0),
        })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def rerank_rxnorm(query: str, candidates: List[Dict], kb, top_k: int = 5) -> List[Dict]:
    """Rerank RxNorm candidates using semantic embeddings."""
    if not candidates:
        return []
    
    cand_texts = []
    for c in candidates:
        cui = c["rxcui"]
        info = kb.get_rxnorm_concept(cui)
        name = info.get("prefLabel", "") if info else ""
        cand_texts.append(name if name else cui)
    
    query_emb = _encode(query)
    cand_embs = _encode(cand_texts)
    scores = np.dot(cand_embs, query_emb.T).squeeze()
    if len(cand_texts) == 1:
        scores = np.array([float(scores)])
    
    results = []
    for i, c in enumerate(candidates):
        results.append({
            "rxcui": c["rxcui"],
            "label": cand_texts[i],
            "score": float(scores[i]),
            "old_score": c.get("score", 0),
        })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
