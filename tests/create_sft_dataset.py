#!/usr/bin/env python3
"""Create SFT training dataset from pipeline extractions + existing medical data."""

import json, sys, csv, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

GT_PATH = Path(__file__).parent.parent  # Project root

def create_sft_from_pipeline():
    """Use pipeline outputs as training data for Qwen NER."""
    output_dir = Path(__file__).parent.parent / "outputs_new"
    sft_data = []
    
    for json_file in sorted(output_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text())
            mentions = data if isinstance(data, list) else data.get("mentions", [])
            if not mentions:
                continue
            
            # Read original text
            text_file = Path(__file__).parent.parent / "input" / f"{json_file.stem}.txt"
            if not text_file.exists():
                continue
            text = text_file.read_text()
            
            # Build entity list for SFT
            entities = []
            for m in mentions:
                entities.append({"entity": m.get("text", ""), "type": m.get("type", "")})
            
            if not entities:
                continue
                
            sft_data.append({
                "messages": [
                    {"role": "system", "content": "Extract all medical entities from Vietnamese clinical text. Return JSON array with entity and type."},
                    {"role": "user", "content": text[:3000]},
                    {"role": "assistant", "content": json.dumps(entities, ensure_ascii=False)},
                ]
            })
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
    
    print(f"Pipeline extractions: {len(sft_data)} samples")
    return sft_data


def create_sft_from_csv():
    """Create SFT from ViMedical_Disease.csv — disease classification."""
    csv_path = GT_PATH / "data_train/ViMedical_Disease/ViMedical_Disease.csv"
    sft_data = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            disease = row.get("Disease", "").strip()
            question = row.get("Question", "").strip()
            if not disease or not question:
                continue
            sft_data.append({
                "messages": [
                    {"role": "system", "content": "You are a medical diagnosis assistant. Based on symptoms, identify the disease."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": json.dumps({"disease": disease}, ensure_ascii=False)},
                ]
            })
    print(f"ViMedical_Disease: {len(sft_data)} samples")
    return sft_data


def create_sft_from_jsonl():
    """Create SFT from vietnamese-medical-dataset.jsonl — medical QA."""
    jsonl_path = GT_PATH / "data_train/vietnamese-medical-dataset.json"
    sft_data = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                item = json.loads(line)
                anchor = item.get("anchor", "")
                positive = item.get("positive", "")[:500]
                if not anchor or not positive:
                    continue
                sft_data.append({
                    "messages": [
                        {"role": "system", "content": "You are a Vietnamese medical QA assistant. Answer medical questions accurately."},
                        {"role": "user", "content": anchor},
                        {"role": "assistant", "content": positive},
                    ]
                })
            except:
                pass
    print(f"Vietnamese-medical-dataset: {len(sft_data)} samples")
    return sft_data


def main():
    all_data = []
    
    # 1. Vietnamese medical QA (JSONL)
    all_data.extend(create_sft_from_jsonl())
    
    # 2. ViMedical_Disease (disease classification from symptoms)
    all_data.extend(create_sft_from_csv())
    
    # 3. Vietnamese medical QA
    # all_data.extend(create_sft_from_jsonl())
    
    # Save
    out_path = GT_PATH / "data_train/sft_dataset.jsonl"
    with open(out_path, "w") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\nTotal SFT samples saved to: {out_path}")
    print(f"Total: {len(all_data)} samples")


if __name__ == "__main__":
    main()
