#!/usr/bin/env python3
"""Create GRPO training dataset from VietBioNER + ViMedNer + ViMQ.

Formats:
  VietBioNER: BIO tagged (word IOB) — train.txt
  ViMedNer:   BIO tagged (word IOB) — train.txt  
  ViMQ:       JSON with seq_label [start_idx, end_idx, type]

Output: GRPO-compatible JSONL with messages format + reward labels
"""

import json, sys, re, os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "src"))

# ─── Label mapping ──────────────────────────────────────────
LABEL_MAP = {
    "drug": "THUỐC",
    "Drug": "THUỐC",
    "DRUG": "THUỐC",
    "ten_thuoc": "THUỐC",
    "thuoc": "THUỐC",
    "symptom": "TRIỆU_CHỨNG",
    "Symptom": "TRIỆU_CHỨNG",
    "SYMPTOM": "TRIỆU_CHỨNG",
    "disease": "CHẨN_ĐOÁN",
    "Disease": "CHẨN_ĐOÁN",
    "DISEASE": "CHẨN_ĐOÁN",
    "ten_benh": "CHẨN_ĐOÁN",
    "diagnosis": "CHẨN_ĐOÁN",
    "medical_procedure": "THỦ_THUẬT",
    "Medical_Procedure": "THỦ_THUẬT",
    "test": "TÊN_XÉT_NGHIỆM",
    "Test": "TÊN_XÉT_NGHIỆM",
    "result": "KẾT_QUẢ_XÉT_NGHIỆM",
    "Result": "KẾT_QUẢ_XÉT_NGHIỆM",
    "bien_phap_dieu_tri": "THỦ_THUẬT",
    "bien_phap_chan_doan": "TÊN_XÉT_NGHIỆM",
    "nguyen_nhan_benh": "TRIỆU_CHỨNG",
    "trieu_chung_benh": "TRIỆU_CHỨNG",
    "Symptom_and_Disease": "SYMPTOM_AND_DISEASE",  # needs LLM split
    "SYMPTOM_AND_DISEASE": "SYMPTOM_AND_DISEASE",
}

SPLIT_NEEDED = {"SYMPTOM_AND_DISEASE", "Symptom_and_Disease"}

# ─── BIO to spans ───────────────────────────────────────────

def bio_to_spans(tokens, labels):
    """Convert BIO-tagged tokens to span entities. Handles missing B- prefix."""
    entities = []
    current_type = None
    current_tokens = []
    current_start = 0
    
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith("B-") or label == "O":
            if current_type and current_tokens:
                entities.append({
                    "text": " ".join(current_tokens),
                    "tokens": current_tokens[:],
                    "start": current_start, "end": i,
                    "type": current_type,
                })
            if label.startswith("B-"):
                current_type = label[2:]
                current_tokens = [token]
                current_start = i
            else:
                current_type = None
                current_tokens = []
        elif label.startswith("I-"):
            if current_type and label[2:] == current_type:
                current_tokens.append(token)
            elif current_type and current_tokens:
                # Different I- type or no previous B- — start new entity
                entities.append({
                    "text": " ".join(current_tokens),
                    "tokens": current_tokens[:],
                    "start": current_start, "end": i,
                    "type": current_type,
                })
                current_type = label[2:]
                current_tokens = [token]
                current_start = i
            else:
                # I- without B- — start new entity
                current_type = label[2:]
                current_tokens = [token]
                current_start = i
        else:
            if current_type and current_tokens:
                entities.append({
                    "text": " ".join(current_tokens),
                    "tokens": current_tokens[:],
                    "start": current_start, "end": i,
                    "type": current_type,
                })
            current_type = None
            current_tokens = []
    
    if current_type and current_tokens:
        entities.append({
            "text": " ".join(current_tokens),
            "tokens": current_tokens[:],
            "start": current_start, "end": len(tokens),
            "type": current_type,
        })
    return entities


# ─── ViMQ parsing ───────────────────────────────────────────

def parse_vimq(filepath: str) -> list:
    """Parse ViMQ JSON training data into NER samples."""
    samples = []
    with open(filepath) as f:
        data = json.load(f)
    
    for item in data:
        sentence_raw = item.get("sentence", "")
        seq_labels = item.get("seq_label", [])
        
        # Split the underscore-separated sentence into tokens
        tokens = sentence_raw.split()
        # Clean up underscores
        text = " ".join(t.replace("_", " ") for t in tokens)
        
        entities = []
        for start_idx, end_idx, etype in seq_labels:
            if start_idx >= len(tokens) or end_idx >= len(tokens):
                continue
            entity_tokens = tokens[start_idx:end_idx+1]
            entity_text = " ".join(t.replace("_", " ") for t in entity_tokens)
            
            mapped = LABEL_MAP.get(etype, etype)
            entities.append({
                "text": entity_text,
                "start": start_idx,
                "end": end_idx + 1,
                "type": mapped,
            })
        
        if entities:
            samples.append({
                "text": text,
                "tokens": [t.replace("_", " ") for t in tokens],
                "entities": entities,
            })
    return samples


# ─── VietBioNER parsing ─────────────────────────────────────

def parse_vietbioner(filepath: str) -> list:
    """Parse VietBioNER BIO-tagged format."""
    samples = []
    current_tokens = []
    current_labels = []
    
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    spans = bio_to_spans(current_tokens, current_labels)
                    mapped_spans = []
                    for s in spans:
                        s["type"] = LABEL_MAP.get(s["type"], s["type"])
                        mapped_spans.append(s)
                    
                    if mapped_spans:
                        samples.append({
                            "text": " ".join(current_tokens),
                            "tokens": current_tokens[:],
                            "entities": mapped_spans,
                        })
                    current_tokens = []
                    current_labels = []
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            token, label = parts[0], parts[-1]
            current_tokens.append(token)
            current_labels.append(label)
    
    if current_tokens:
        spans = bio_to_spans(current_tokens, current_labels)
        mapped_spans = []
        for s in spans:
            s["type"] = LABEL_MAP.get(s["type"], s["type"])
            mapped_spans.append(s)
        if mapped_spans:
            samples.append({
                "text": " ".join(current_tokens),
                "tokens": current_tokens[:],
                "entities": mapped_spans,
            })
    return samples


# ─── ViMedNer parsing ───────────────────────────────────────

def parse_vimedner(filepath: str) -> list:
    """Parse ViMedNer BIO-tagged format (same as VietBioNER)."""
    return parse_vietbioner(filepath)


# ─── LLM split SYMPTOM_AND_DISEASE ────────────────────────────

SPLIT_SYSTEM = """You are a medical NER expert. Given a Vietnamese medical entity, determine if it is a SYMPTOM or a DISEASE (CHẨN_ĐOÁN).

Rules:
- SYMPTOM = patient complaint, abnormal sensation (đau, khó thở, sốt, mệt mỏi, ho...)
- DISEASE = named disease, syndrome (viêm phổi, tăng huyết áp, tiểu đường, ung thư...)
- If it could be both or unclear, choose the most appropriate one.
- Return JSON: {"type": "TRIỆU_CHỨNG"} or {"type": "CHẨN_ĐOÁN"}"""


def split_with_llm(entity_text: str, provider: str = "featherless") -> str:
    """Use LLM to split SYMPTOM_AND_DISEASE. Supports 'featherless' or 'qwen'."""
    import os, json, re
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv(PROJECT / ".env")

    if provider == "qwen":
        client = OpenAI(
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            api_key=os.getenv("QWEN_API", ""),
        )
        model = "qwen-plus"
    else:
        client = OpenAI(
            base_url=os.getenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1"),
            api_key=os.getenv("FEATHERLESS_API_KEY", ""),
        )
        model = os.getenv("FEATHERLESS_MODEL", "Qwen/Qwen3.5-397B-A17B")

    try:
        resp = client.chat.completions.create(
            model=model, max_tokens=50,
            messages=[
                {"role": "system", "content": SPLIT_SYSTEM},
                {"role": "user", "content": f"Entity: \"{entity_text}\""},
            ],
        )
        content = resp.choices[0].message.content.strip()

        # Try JSON parse
        for pattern in [content, *re.findall(r'\{[^}]+\}', content)]:
            try:
                result = json.loads(pattern)
                t = result.get("type", "")
                if t in ("TRIỆU_CHỨNG", "CHẨN_ĐOÁN"):
                    return t
            except:
                pass

        # Check raw text
        if "CHẨN_ĐOÁN" in content.upper():
            return "CHẨN_ĐOÁN"
        if "TRIỆU_CHỨNG" in content.upper():
            return "TRIỆU_CHỨNG"
    except Exception as e:
        print(f"  LLM error for \"{entity_text}\": {e}")

    # Rule-based fallback
    lower = entity_text.lower()
    if any(kw in lower for kw in ["viêm", "ung thư", "bệnh", "hội chứng", "suy", "tăng", "tiểu đường"]):
        return "CHẨN_ĐOÁN"
    return "TRIỆU_CHỨNG"


def split_entities(samples: list, use_llm: bool = False, provider: str = "featherless") -> list:
    """Split SYMPTOM_AND_DISEASE entities using LLM or rules."""
    for i, sample in enumerate(samples):
        for entity in sample["entities"]:
            if entity["type"] in SPLIT_NEEDED:
                if use_llm:
                    entity["type"] = split_with_llm(entity["text"], provider)
                else:
                    lower = entity["text"].lower()
                    if any(kw in lower for kw in ["viêm", "ung thư", "bệnh", "hội chứng", "suy", "xơ", "tăng huyết áp", "đái tháo"]):
                        entity["type"] = "CHẨN_ĐOÁN"
                    else:
                        entity["type"] = "TRIỆU_CHỨNG"
    return samples


# ─── GRPO format ────────────────────────────────────────────

SYSTEM_NER = "Extract all medical entities from Vietnamese clinical text. Return JSON array with entity and type. Types: THUỐC, TRIỆU_CHỨNG, CHẨN_ĐOÁN, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM, THỦ_THUẬT."


MEDICAL_TYPES = {"THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM", "THỦ_THUẬT"}


def to_grpo_format(samples: list) -> list:
    """Convert NER samples to GRPO training format with positions."""
    grpo_data = []
    for sample in samples:
        entities = []
        for e in sample["entities"]:
            etype = e["type"]
            if etype in SPLIT_NEEDED:
                etype = "TRIỆU_CHỨNG"
            if etype not in MEDICAL_TYPES:
                continue
            # Find position in full text
            text = sample["text"]
            entity_text = e["text"]
            pos = text.find(entity_text)
            entities.append({
                "entity": entity_text,
                "type": etype,
                "position": [pos, pos + len(entity_text)] if pos >= 0 else [0, 0],
            })
        
        if not entities:
            continue
        
        # Build JSON response with positions
        response_json = json.dumps(entities, ensure_ascii=False)
        
        grpo_data.append({
            "messages": [
                {"role": "system", "content": SYSTEM_NER},
                {"role": "user", "content": sample["text"]},
                {"role": "assistant", "content": response_json},
            ],
            "entities": entities,
        })
    return grpo_data


# ─── Main ────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-llm", action="store_true", help="Use LLM to split SYMPTOM_AND_DISEASE")
    parser.add_argument("--llm", choices=["featherless", "qwen"], default="featherless",
                        help="LLM provider for splitting")
    parser.add_argument("--output", default="data_train/grpo_dataset.jsonl")
    args = parser.parse_args()
    
    all_samples = []
    
    # 1. VietBioNER
    bioner_path = PROJECT / "data_train/VietBioNER/data_supervised_learning/train.txt"
    if bioner_path.exists():
        bioner = parse_vietbioner(str(bioner_path))
        print(f"VietBioNER: {len(bioner)} samples")
        all_samples.extend(bioner)
    
    # 2. ViMedNer
    medner_path = PROJECT / "data_train/ViMedNer/data/train.txt"
    if medner_path.exists():
        medner = parse_vimedner(str(medner_path))
        print(f"ViMedNer: {len(medner)} samples")
        all_samples.extend(medner)
    
    # 3. ViMQ
    vimq_path = PROJECT / "data_train/ViMQ/data/train.json"
    if vimq_path.exists():
        vimq = parse_vimq(str(vimq_path))
        print(f"ViMQ: {len(vimq)} samples")
        all_samples.extend(vimq)
    
    print(f"Total entities before split: {sum(len(s['entities']) for s in all_samples)}")
    
    # Split SYMPTOM_AND_DISEASE
    split_count = sum(1 for s in all_samples for e in s["entities"] if e["type"] in SPLIT_NEEDED)
    print(f"SYMPTOM_AND_DISEASE to split: {split_count}")
    
    all_samples = split_entities(all_samples, use_llm=args.use_llm, provider=args.llm)
    
    # Convert to GRPO format
    grpo_data = to_grpo_format(all_samples)
    
    # Save
    out_path = PROJECT / args.output
    with open(out_path, "w") as f:
        for item in grpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Stats
    type_counts = defaultdict(int)
    for item in grpo_data:
        for e in item["entities"]:
            type_counts[e["type"]] += 1
    
    print(f"\nSaved {len(grpo_data)} GRPO samples to {out_path}")
    print("Entity types:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
