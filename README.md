# Vietnamese Medical NER Pipeline

Clinical information extraction system for Vietnamese free-text medical records. Extracts medical concepts, assigns types, infers assertions, and maps to ICD-10 / RxNorm ontologies.

## Architecture

```
Input .txt
  ├── Preprocess (Unicode normalization, sentence splitting)
  ├── Rule-based extraction
  │   ├── THUỐC (drugs): ingredient + dose + route + frequency
  │   ├── TÊN_XÉT_NGHIỆM (test names): ECG, X-ray, MRI, labs
  │   ├── KẾT_QUẢ_XÉT_NGHIỆM (lab values): numeric results, "bình thường"
  │   ├── CHẨN_ĐOÁN (diagnoses): "chẩn đoán:", "bệnh chính:"
  │   └── TRIỆU_CHỨNG (symptoms): clinical symptom keywords
  ├── Qwen2.5-7B NER (optional, --use-ner flag)
  │   └── Prompt-based entity extraction via chat template
  ├── Merge engine (dedup, conflict resolution, type enforcement)
  ├── Assertion classifier (negation, family, historical)
  ├── ICD-10 / RxNorm linker (exact + fuzzy match → candidates)
  └── Output JSON
```

## Setup

```bash
# Create venv and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers scikit-learn rdflib rapidfuzz pymupdf pandas numpy
```

## Usage

```bash
source .venv/bin/activate

# Rules-only mode (fast, ~1.3s per file)
python3 src/inference.py

# With Qwen2.5-7B NER model (slow on CPU, better recall)
python3 src/inference.py --use-ner

# With LLM modules (canonicalization, assertions, reranking, agent review)
python3 src/inference.py --use-llm

# Process single file, print to stdout
python3 src/inference.py --single input/1.txt

# Custom paths
python3 src/inference.py --input /path/to/txt/files --output /path/to/json/output --models /path/to/models
```

## Output Format

Each input file produces a JSON array of extracted mentions:

```json
[
  {
    "text": "metoprolol 25mg po bid",
    "position": [53, 75],
    "type": "THUỐC",
    "assertions": {
      "isNegated": false,
      "isFamily": false,
      "isHistorical": true
    },
    "candidates": ["6918", "214621", "1007340"]
  },
  {
    "text": "đánh trống ngực",
    "position": [498, 512],
    "type": "TRIỆU_CHỨNG",
    "assertions": {
      "isNegated": false,
      "isFamily": false,
      "isHistorical": false
    }
  }
]
```

### Fields

| Field | Type | Description |
|---|---|---|
| `text` | string | Extracted mention text |
| `position` | [int, int] | Character offset `[start, end)` in original text |
| `type` | string | One of 5 concept types (see below) |
| `assertions` | object | Boolean flags for contextual assertions |
| `candidates` | string[] | ICD-10 codes or RxNorm CUIs (only for `CHẨN_ĐOÁN` and `THUỐC`) |

### Concept Types

| Type | Vietnamese | Description | Has candidates |
|---|---|---|---|
| `THUỐC` | Thuốc | Medications with dose/route/freq | Yes → RxNorm CUI |
| `TRIỆU_CHỨNG` | Triệu chứng | Symptoms, clinical signs | No |
| `CHẨN_ĐOÁN` | Chẩn đoán | Diagnoses, conditions | Yes → ICD-10 code |
| `TÊN_XÉT_NGHIỆM` | Tên xét nghiệm | Test/exam names | No |
| `KẾT_QUẢ_XÉT_NGHIỆM` | Kết quả xét nghiệm | Lab/test results | No |

### Assertions

| Flag | Description |
|---|---|
| `isNegated` | Mention is negated (e.g., "không đánh trống ngực") |
| `isFamily` | Mention refers to family history (e.g., "bố bị tiểu đường") |
| `isHistorical` | Mention is historical (e.g., "tiền sử tăng huyết áp") |

## Modules

| File | Purpose |
|---|---|
| `src/preprocess.py` | Unicode normalization, sentence splitting |
| `src/kb.py` | Knowledge base: RxNorm (124K concepts), ICD-10 (6K codes), Vietnamese ICD-10 from PDF |
| `src/rules.py` | Regex + dictionary extraction for all 5 concept types |
| `src/ner.py` | Qwen2.5-7B Vietnamese Medical NER (prompt-based, optional) |
| `src/merge.py` | Combines rule + NER outputs, resolves conflicts, deduplicates |
| `src/assertion.py` | Per-mention assertion classification using trigger patterns |
| `src/linker.py` | ICD-10 and RxNorm candidate generation (exact + fuzzy matching) |
| `src/llm.py` | Core LLM wrapper for Qwen2.5-7B (shared by all LLM modules) |
| `src/canonicalize.py` | LLM-based mention canonicalization (fix abbreviations, spelling) |
| `src/assertion_llm.py` | LLM assertion checker for hard scope cases |
| `src/reranker.py` | LLM-assisted ICD-10/RxNorm candidate reranking |
| `src/agent.py` | Agent reviewer for uncertain cases (type, assertions, candidates) |
| `src/inference.py` | Pipeline orchestrator, CLI, JSON output |
| `src/extract_icd10_pdf.py` | Extract ICD-10 codes from Vietnamese MoH PDF to CSV |

## Data Sources

| File | Content |
|---|---|
| `database-stuff/RXNORM.ttl` | RxNorm ontology (2.8M lines RDF), English drug names + relationships |
| `database-stuff/ICDO.csv` | ICD-10 codes with English names (6K codes, 900+ with ICD-10 mapping) |
| `database-stuff/06-byt-kem.pdf` | Vietnamese MoH ICD-10 code list (1271 pages), Vietnamese disease names |
| `models/Qwen2.5-7B-Vietnamese-Medical-NER-GRPO/` | Vietnamese Medical NER LLM (7.6B params, bf16) |
| `models/vipubmed-deberta-base/` | Vietnamese biomedical DeBERTa (768M params) |

## Current Performance

| Metric | Value |
|---|---|
| Files processed | 100/100 |
| Total mentions | 589 |
| With ICD-10/RxNorm candidates | 222 (38%) |
| Avg processing time | 1.3s/file (rules only) |

### Type Distribution

| Type | Count |
|---|---|
| TÊN_XÉT_NGHIỆM | 225 |
| CHẨN_ĐOÁN | 186 |
| TRIỆU_CHỨNG | 83 |
| KẾT_QUẢ_XÉT_NGHIỆM | 58 |
| THUỐC | 37 |

## Improving Results

1. **Add Qwen NER**: `--use-ner` flag adds LLM-based extraction for symptoms and diagnoses the rules miss
2. **Tune assertion window**: Edit `window` parameter in `src/assertion.py` (default: 40 chars)
3. **Add drug dictionary**: Extend Vietnamese drug name mappings in `src/linker.py`
4. **Improve ICD-10 matching**: The PDF parser extracts Vietnamese names; refine in `src/kb.py`

## Project Structure

```
viettle-nlp/
├── input/                  # 100 Vietnamese clinical text files
├── outputs/                # Generated JSON outputs
├── models/
│   ├── Qwen2.5-7B-Vietnamese-Medical-NER-GRPO/
│   └── vipubmed-deberta-base/
├── database-stuff/
│   ├── RXNORM.ttl
│   ├── ICDO.csv
│   └── 06-byt-kem.pdf
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── kb.py
│   ├── rules.py
│   ├── ner.py
│   ├── merge.py
│   ├── assertion.py
│   ├── linker.py
│   └── inference.py
└── .venv/
```
