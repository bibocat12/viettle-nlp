# Vietnamese Medical NER Pipeline

**Competition score: 34.8828** (ViMedNer shared task)

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
  ├── bibocat/qwen3-ner-grpo-v2-merged through local vLLM
  ├── Exact span alignment + rules/Qwen voting
  ├── Scoped assertion rules (negation, family, historical)
  ├── ICD-10 / RxNorm retrieve → rerank → top-1
  └── Strict schema/offset validation → output.zip
```

## Setup

```bash
# Create venv and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Models

The NER proposal model is hosted on Hugging Face:

| Model | HF Repo | Size | Role |
|---|---|---|---|
| **Qwen3-GRPO-v2** | [`bibocat/qwen3-ner-grpo-v2-merged`](https://huggingface.co/bibocat/qwen3-ner-grpo-v2-merged) | 8.2B params | NER extraction (high recall) |
| **ViPubMedDeBERTa** | local `models/vipubmed-deberta-base/` | 768M params | Embedding reranking for ICD-10/RxNorm |

### Download via HF CLI

```bash
# Install hf CLI (replaces deprecated huggingface-cli)
curl -LsSf https://hf.co/cli/install.sh | bash -s

# Authenticate
export HF_TOKEN=hf_...
hf auth login

# Download the NER model (~16 GB, 4 shards)
hf download bibocat/qwen3-ner-grpo-v2-merged --local-dir ./models/qwen3-ner-grpo-v2-merged

# Download ViPubMedDeBERTa embeddings model
hf download models/vipubmed-deberta-base --local-dir ./models/vipubmed-deberta-base
```

## Usage

```bash
source .venv/bin/activate

# Recommended competition pipeline with the self-hosted Qwen3 GRPO model
python3 cli.py input/ --llm vllm \
  --base-url http://localhost:8000/v1 \
  --served-model bibocat/qwen3-ner-grpo-v2-merged \
  --output output

# Safe score-first upgrade: preserve the fixed lb3 anchor, run three schema-only
# passes, and package only output/1.json … output/100.json.
python3 cli.py input_turn2_vong1/input --llm vllm \
  --base-url http://localhost:8000/v1 \
  --served-model bibocat/qwen3-ner-grpo-v2-merged \
  --strategy grpo-v2-fusion \
  --anchor-zip input_turn2_vong1/lb3_base_qwen/output.zip \
  --passes 3 --parallel 2 --resume \
  --output input_turn2_vong1/lb4_grpo_v2_fusion/output

# Fail on schema/offset/package errors, then create output.zip
python3 scripts/build_submission.py --check-only
python3 scripts/build_submission.py

# Legacy rules-only pipeline
python3 src/inference.py

# Legacy local Qwen2.5 pipeline
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
    "type": "THUỐC",
    "assertions": ["isHistorical"],
    "position": [53, 75],
    "candidates": ["6918"]
  },
  {
    "text": "đánh trống ngực",
    "type": "TRIỆU_CHỨNG",
    "assertions": [],
    "position": [498, 512]
  }
]
```

### Fields

| Field | Type | Description |
|---|---|---|
| `text` | string | Extracted mention text |
| `position` | [int, int] | Character offset `[start, end)` in original text |
| `type` | string | One of 5 concept types (see below) |
| `assertions` | string[] | True contextual assertions only |
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
| `src/vllm_llm.py` | OpenAI-compatible client pinned to the Qwen3 GRPO checkpoint |
| `src/grpo_v2_fusion.py` | Anchored three-pass Qwen3-v2 fusion, cache/resume, audit and final ZIP validation |
| `src/ner_tool.py` | Chunked NER and deterministic source-text span alignment |
| `src/voting.py` | Qwen/rule merge, atomic span and type resolution |
| `src/agents/worker_agents.py` | Extraction, assertion, linking, and verification workers |
| `src/linker_agent.py` | Structured RxNorm and hybrid ICD-10 retrieval/reranking |
| `src/position_fix.py` | Exact offset repair without first-token fallbacks |
| `src/verifier_agent.py` | Final mention validation and exact deduplication |
| `scripts/build_submission.py` | Strict 100-file validator and ZIP builder |

## Data Sources

| File | Content |
|---|---|
| `database-stuff/RXNORM.ttl` | RxNorm ontology (2.8M lines RDF), English drug names + relationships |
| `database-stuff/ICDO.csv` | ICD-10 codes with English names (6K codes, 900+ with ICD-10 mapping) |
| `database-stuff/06-byt-kem.pdf` | Vietnamese MoH ICD-10 code list (1271 pages), Vietnamese disease names |
| [`bibocat/qwen3-ner-grpo-v2-merged`](https://huggingface.co/bibocat/qwen3-ner-grpo-v2-merged) | Self-hosted NER proposal checkpoint served by vLLM |
| `models/vipubmed-deberta-base/` | Vietnamese biomedical DeBERTa (768M params) |

## Validation Snapshot

| Check | Result |
|---|---|
| Organizer public example | Exact 19/19 entities |
| Organizer RxNorm example | Exact 11/11 top-1 codes |
| Public-example offsets | 0 mismatches |
| Regression suite | `pytest -q tests/test_pipeline_regressions.py` |

## Improving Results

1. Annotate a stratified 30–40-file development set without inspecting leaderboard labels.
2. Tune span/type thresholds on the local metric, one change per ablation.
3. Use leaderboard submissions to compare extraction, assertion, and linking variants separately.
4. Keep only changes that improve the weighted score across at least two submissions.

## Project Structure

```
viettle-nlp/
├── input/                  # 100 Vietnamese clinical text files
├── outputs/                # Generated JSON outputs
├── models/
│   ├── qwen3-ner-grpo-v2-merged/   # downloaded from HF bibocat/qwen3-ner-grpo-v2-merged
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
