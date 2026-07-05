import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))

from preprocess import preprocess
from rules import extract_rules
from merge import merge_mentions
from assertion import classify_assertions
from linker import link_mentions


def _load_qwen_ner(models_dir: str = None):
    from ner import QwenNER
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    model_path = os.path.join(models_dir, "Qwen2.5-7B-Vietnamese-Medical-NER-GRPO")
    return QwenNER(model_path=model_path)


def _load_kb():
    from kb import KnowledgeBase
    base_dir = os.path.join(os.path.dirname(__file__), "..", "database-stuff")
    kb = KnowledgeBase(base_dir=base_dir)
    kb.load()
    return kb


def _load_llm(models_dir: str = None):
    from llm import LLMEngine
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    model_path = os.path.join(models_dir, "Qwen2.5-7B-Vietnamese-Medical-NER-GRPO")
    return LLMEngine(model_path=model_path)


def _validate_output(mentions: List[Dict]) -> List[Dict]:
    valid_types = {"THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}
    result = []
    for m in mentions:
        if m.get("type") not in valid_types:
            continue
        pos = m.get("position", [0, 0])
        if not isinstance(pos, list) or len(pos) != 2:
            continue
        if pos[1] <= pos[0]:
            continue
        out = {
            "text": m.get("text", ""),
            "position": pos,
            "type": m["type"],
            "assertions": {
                "isNegated": bool(m.get("assertions", {}).get("isNegated", False)),
                "isFamily": bool(m.get("assertions", {}).get("isFamily", False)),
                "isHistorical": bool(m.get("assertions", {}).get("isHistorical", False)),
            },
        }
        if m["type"] in ("CHẨN_ĐOÁN", "THUỐC"):
            out["candidates"] = m.get("candidates", [])
        result.append(out)
    return result


def predict(text: str, ner_model=None, kb=None, use_llm: bool = False) -> List[Dict]:
    pp = preprocess(text)
    normalized = pp["normalized"]

    rule_mentions = extract_rules(normalized)

    ner_mentions = []
    if ner_model is not None:
        try:
            ner_mentions = ner_model.extract(normalized)
        except Exception as e:
            print(f"NER extraction failed: {e}", file=sys.stderr)

    merged = merge_mentions(rule_mentions, ner_mentions)

    for m in merged:
        m["text"] = normalized[m["position"][0]:m["position"][1]]

    merged = classify_assertions(normalized, merged)

    if kb is not None:
        merged = link_mentions(merged, kb)

    if use_llm:
        try:
            from canonicalize import canonicalize_mentions
            merged = canonicalize_mentions(merged, normalized)
        except Exception as e:
            print(f"Canonicalization failed: {e}", file=sys.stderr)

        try:
            from assertion_llm import refine_assertions_llm
            merged = refine_assertions_llm(merged, normalized)
        except Exception as e:
            print(f"Assertion refinement failed: {e}", file=sys.stderr)

        if kb is not None:
            try:
                from reranker import rerank_mentions
                merged = rerank_mentions(merged, normalized, kb)
            except Exception as e:
                print(f"Reranking failed: {e}", file=sys.stderr)

            try:
                from agent import review_mentions
                merged = review_mentions(merged, normalized, kb)
            except Exception as e:
                print(f"Agent review failed: {e}", file=sys.stderr)

    result = _validate_output(merged)
    return result


def predict_file(input_path: str, output_path: str, ner_model=None, kb=None, use_llm: bool = False) -> List[Dict]:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    result = predict(text, ner_model=ner_model, kb=kb, use_llm=use_llm)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Vietnamese Medical NER Pipeline")
    parser.add_argument("--input", default=None, help="Input file or directory")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--models", default=None, help="Models directory (default: ./models)")
    parser.add_argument("--use-ner", action="store_true", help="Use Qwen NER model (slow on CPU)")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for canonicalization, assertions, reranking, and agent review")
    parser.add_argument("--single", default=None, help="Process single file and print to stdout")
    args = parser.parse_args()

    input_dir = args.input or os.path.join(os.path.dirname(__file__), "..", "input")
    output_dir = args.output or os.path.join(os.path.dirname(__file__), "..", "outputs")
    models_dir = args.models or os.path.join(os.path.dirname(__file__), "..", "models")

    print("Loading knowledge base...")
    kb = _load_kb()
    print(f"  ICD-10 codes: {len(kb.icd10)}")
    print(f"  RxNorm concepts: {len(kb.rxnorm_concepts)}")

    ner_model = None
    if args.use_ner:
        print("Loading Qwen NER model...")
        ner_model = _load_qwen_ner(models_dir)
        ner_model.load()
        print("  Model loaded.")

    if args.use_llm:
        print("Loading LLM engine...")
        llm = _load_llm(models_dir)
        llm.load()
        print("  LLM loaded.")

    if args.single:
        with open(args.single, "r", encoding="utf-8") as f:
            text = f.read()
        result = predict(text, ner_model=ner_model, kb=kb, use_llm=args.use_llm)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    os.makedirs(output_dir, exist_ok=True)
    input_files = sorted(Path(input_dir).glob("*.txt"), key=lambda p: int(p.stem))
    print(f"Processing {len(input_files)} files...")

    total_time = 0
    for i, fpath in enumerate(input_files):
        t0 = time.time()
        out_path = os.path.join(output_dir, f"{fpath.stem}.json")
        result = predict_file(str(fpath), out_path, ner_model=ner_model, kb=kb, use_llm=args.use_llm)
        elapsed = time.time() - t0
        total_time += elapsed
        print(f"  [{i+1}/{len(input_files)}] {fpath.name} -> {len(result)} mentions ({elapsed:.1f}s)")

    print(f"Done. Total time: {total_time:.1f}s, Avg: {total_time/len(input_files):.1f}s/file")


if __name__ == "__main__":
    main()
