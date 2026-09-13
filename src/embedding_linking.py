import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

# transformers 5.x validates repo IDs strictly.
# Monkey-patch to allow local absolute paths like /kaggle/...
import huggingface_hub.utils._validators as _hub_val
_hub_val.validate_repo_id = lambda repo_id: None

from transformers import AutoTokenizer, AutoModel


_model = None
_tokenizer = None
_device = "cpu"


def _load_model(device: str = "cpu"):
    global _model, _tokenizer, _device
    if _model is not None:
        return
    _device = device
    base = os.environ.get("VIETTLE_MODELS_DIR", str(Path(__file__).parent.parent / "models"))
    model_path = os.path.join(base, "vipubmed-deberta-base")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    _model = AutoModel.from_pretrained(model_path, local_files_only=True)
    _model.eval()
    if device == "cuda" and torch.cuda.is_available():
        _model = _model.cuda()
        _device = "cuda"


@torch.no_grad()
def _encode(texts, max_length: int = 128) -> np.ndarray:
    _load_model()
    if isinstance(texts, str):
        texts = [texts]
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = _tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        if _device == "cuda":
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


# ─── Reranking (existing) ───────────────────────────────────────────


def rerank_icd10(query: str, candidates: List[Dict], kb, top_k: int = 5) -> List[Dict]:
    if not candidates:
        return []
    cand_texts = []
    for c in candidates:
        code = c["code"]
        info = kb.get_icd10_code(code)
        name = info.get("vietnamese_name", "") or info.get("english_name", "") or ""
        en = info.get("english_name", "")
        text = name
        if en and en != name:
            text += " " + en
        cand_texts.append(text if text else code)
    query_emb = _encode(query)
    cand_embs = _encode(cand_texts)
    scores = np.dot(cand_embs, query_emb.T).squeeze()
    if len(cand_texts) == 1:
        scores = np.array([float(scores)])
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


# ─── Full KB embedding search (new) ─────────────────────────────────

_icd10_embeddings = None
_icd10_codes: List[str] = []
_icd10_names: List[str] = []
_rxnorm_embeddings = None
_rxnorm_codes: List[str] = []
_rxnorm_names: List[str] = []


def _build_icd10_index(kb):
    global _icd10_embeddings, _icd10_codes, _icd10_names
    if _icd10_embeddings is not None:
        return
    codes, names = [], []
    for code, info in kb.icd10.items():
        name = info.get("vietnamese_name", "") or info.get("english_name", "")
        if name:
            codes.append(code)
            names.append(name)
    _icd10_codes = codes
    _icd10_names = names
    _icd10_embeddings = _encode(names)


def _build_rxnorm_index(kb):
    global _rxnorm_embeddings, _rxnorm_codes, _rxnorm_names
    if _rxnorm_embeddings is not None:
        return
    codes, names = [], []
    for cui, info in kb.rxnorm_concepts.items():
        name = info.get("prefLabel", "")
        alt_labels = [a for a in info.get("altLabels", []) if a and a != name]
        if name:
            codes.append(cui)
            names.append(name)
        for alt in alt_labels[:3]:
            codes.append(cui)
            names.append(alt)
    _rxnorm_codes = codes
    _rxnorm_names = names
    _rxnorm_embeddings = _encode(names)


def search_icd10_embedding(query: str, kb, top_k: int = 15) -> List[Dict]:
    """Full KB embedding search for ICD-10. Replaces fuzzy search."""
    _load_model()
    _build_icd10_index(kb)
    q_emb = _encode(query)
    scores = np.dot(_icd10_embeddings, q_emb.T).squeeze()
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return [
        {"code": _icd10_codes[i], "score": float(scores[i]), "name": _icd10_names[i]}
        for i in top_idx if scores[i] > 0.1
    ]


def search_rxnorm_embedding(query: str, kb, top_k: int = 15) -> List[Dict]:
    """Full KB embedding search for RxNorm. Replaces fuzzy search."""
    _load_model()
    _build_rxnorm_index(kb)
    q_emb = _encode(query)
    scores = np.dot(_rxnorm_embeddings, q_emb.T).squeeze()
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return [
        {"rxcui": _rxnorm_codes[i], "score": float(scores[i]), "name": _rxnorm_names[i]}
        for i in top_idx if scores[i] > 0.1
    ]
