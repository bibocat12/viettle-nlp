# Viettle NLP — Medical NER Pipeline

## Architecture

```
Input (Vietnamese clinical text)
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│ 1. PARALLEL EXTRACTION (ThreadPoolExecutor, max_workers=3)  │
├─────────────────┬────────────────────┬──────────────────────┤
│   Qwen2.5-7B   │   ViHealthBERT     │   Rules (regex)       │
│   (via RunPod) │   (keyword class.) │   (structured patts)  │
│   recall ↑     │   speed ↑          │   precision ↑         │
└────────┬───────┴────────┬───────────┴──────────┬────────────┘
         ▼                ▼                      ▼
    "entities"       "entities"            "mentions"
    + type            + type                + type
                                      (drugs, tests, results)
         └──────────────┼──────────────────────┘
                        ▼
┌────────────────────────────────────────────────────────────┐
│ 2. VOTING (src/voting.py)                                   │
│    Group by position → count type agreement                  │
│    3/3 agree → conf=0.95  │  2/3 → conf=0.80  │  1/3 → LLM │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────────┐
│ 3. POSITION RE-LOCATE (src/position_fix.py)                │
│    Fix LLM-hallucinated offsets via text search            │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────────┐
│ 4. ASSERTION (src/assertion_agent.py)                      │
│    LLM for every CHẨN_ĐOÁN/THUỐC/TRIỆU_CHỨNG               │
│    isNegated / isFamily / isHistorical                     │
│    Fallback: rules if LLM unavailable                      │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────────┐
│ 5. LINKING (src/linker_agent.py)                           │
│    Stage 1: fuzzy search → top-15 candidates               │
│    Stage 2: ViPubMedDeBERTa embedding reranking             │
│    Stage 3: LLM picks (single call)                        │
│    Stage 4: agentic loop if LLM unsure (multi-step)         │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────────┐
│ 6. VERIFICATION (src/verifier_agent.py)                    │
│    Validate schema, fix positions, drop invalid             │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
                    Final JSON output
```

## Models

| Model | Size | Role | Source |
|---|---|---|---|
| **Qwen2.5-7B-GRPO** | 7.6B | NER extraction (recall), LLM reasoning, assertion | RunPod API / `models/` |
| **ViHealthBERT** | 135M | Fast keyword-based type classification | `models/vihealthbert-base-word` |
| **ViPubMedDeBERTa** | 768M | Semantic embedding reranking for ICD-10/RxNorm | `models/vipubmed-deberta-base` |

## Module Files

| File | Purpose |
|---|---|
| `src/prompt/__init__.py` | All LLM prompts (NER, assertion, type verify, linker) |
| `src/ner_tool.py` | LLM extracts keywords → rules find positions → rules verify → LLM verifies |
| `src/voting.py` | 3-model voting + LLM conflict resolution |
| `src/position_fix.py` | Re-locate entity positions via text search |
| `src/assertion_agent.py` | LLM assertion for every mention (isNegated/family/historical) |
| `src/linker_agent.py` | Agentic ICD-10/RxNorm linking with tool-calling loop |
| `src/embedding_linking.py` | ViPubMedDeBERTa embedding reranker |
| `src/healthbert.py` | ViHealthBERT keyword classifier |
| `src/orchestrator.py` | Main pipeline orchestrator |
| `src/tools.py` | Preprocessing, rule extraction, output validation |
| `src/verifier_agent.py` | Schema validation and position/type fixing |
| `src/kb.py` | Knowledge base (ICD-10, RxNorm lookup) |

## Data Sources

| File | Content |
|---|---|
| `database-stuff/icd10_vietnamese.csv` | ICD-10 codes + Vietnamese names (13K codes) |
| `database-stuff/rxnorm_concepts.csv` | Pre-processed RxNorm concepts (124K) |
| `database-stuff/RXNORM.ttl` | Raw RxNorm RDF (source, 209MB) |
| `database-stuff/06-byt-kem.pdf` | Vietnamese MoH ICD-10 code list |

## Agent Logs

Each extraction saves full logs to `agent_logs/{file_id}.json`:

```json
{
  "file_id": "1",
  "timestamp": 1783316879.46,
  "elapsed": 2.15,
  "model": "runpod",
  "logs": [
    {"agent": "orchestrator", "action": "rule_extracted", "output": {"mentions": 18}},
    {"agent": "llm_verify", "action": "corrected", "input": {...}, "output": {...}}
  ],
  "stats": {"rule_mentions": 18, "ner_qwen_mentions": 25, "vote_conflicts": 3}
}
```

## Prompt Management

All LLM prompts live in `src/prompt/__init__.py`:

| Prompt | Used by |
|---|---|
| `NER_PROMPT` | `ner_tool.py` — keyword-only extraction (no positions) |
| `ASSERTION_PROMPT` | `assertion_agent.py` — Vietnamese few-shot assertion |
| `TYPE_VERIFY_PROMPT` | `ner_tool.py` — LLM verifies rule-based classification |
| `LINKER_RERANK_SYSTEM` | `linker_agent.py` — LLM picks from candidates |
| `LINKER_AGENT_SYSTEM` | `linker_agent.py` — Agentic tool-calling loop |

## CLI Usage

```bash
# Rules only
python3 cli.py input/1.txt

# With RunPod LLM
python3 cli.py input/1.txt --llm runpod

# With local Qwen model
python3 cli.py input/1.txt --llm local --model qwen --model-path ./models

# With local model on GPU (all models from same base dir)
python3 cli.py input/1.txt --llm local --model qwen --gpu \
  --model-path /kaggle/input/datasets/bibocat/viettel-nlp-models

# Batch all 100 files
python3 cli.py input/ --llm runpod --output outputs_new

# Batch with 4 parallel workers
python3 cli.py input/ --llm runpod --parallel 4 --output outputs_new
```

**`--model-path`** controls the base directory for **all 3 models**:
- `{model_path}/Qwen2.5-7B-Vietnamese-Medical-NER-GRPO`
- `{model_path}/vihealthbert-base-word`
- `{model_path}/vipubmed-deberta-base`

## Web UI

```bash
# Terminal 1: Flask backend
source .venv/bin/activate && python3 src/web/app.py

# Terminal 2: Vite frontend (hot reload)
cd frontend && npm run dev
```

Open `http://localhost:3000`

- **Model selector**: No LLM / RunPod / Local
- **File browser**: 100 input files with status
- **Extract**: Process single file with streaming logs
- **Run All**: Batch process all files with progress bar
- **Agent Logs**: Detail (input/output) + List (extraction summaries)

## Voting System

3 models run in parallel:
1. **Qwen2.5-7B** — generative NER (high recall)
2. **ViHealthBERT** — keyword-based type classification (fast)
3. **Rules** — regex patterns (high precision for drugs/tests)

Mentions grouped by position. Confidence based on agreement:
- 3/3 agree → conf=0.95
- 2/3 agree → conf=0.80
- 1/3 (only 1 model) → conf=0.50, sent to LLM for resolution

## Tests

```bash
# Quick test (rules only)
python3 -c "import sys; sys.path.insert(0,'src');
from kb import KnowledgeBase; from orchestrator import run_pipeline;
kb=KnowledgeBase(); kb.load();
print(run_pipeline('test','bệnh nhân đau ngực',kb=kb))"
```
