# Pipeline Documentation

## Overview

```
text → preprocessing → PARALLEL EXTRACTION → flatten → filter → voting merge → position fix → LLM verify → assertion → linking → verification → dedup → output
                        ├─ Rules (regex)
                        ├─ HealthBERT (ViHealthBERT keyword+embedding)
                        └─ Qwen (LLM NER, only if --llm set)
```

---

## 1. Preprocessing (`tools.py:10-53`)

**Input:** raw text string
**Output:** `{note_id, original_text, normalized_text, sentences, sections}`

What it does:
- NFC normalization (Unicode)
- Non-breaking space → regular space
- Extract sentences (non-empty lines)
- Extract section headers by patterns (numbered sections `1. ...`, known headers like `Tiền sử...`)

**Why it matters:** The normalized text is what ALL extraction models run on. Positions are relative to this normalized string.

---

## 2. Parallel Extraction (`agents/worker_agents.py:78-103`)

3 models run simultaneously via `ThreadPoolExecutor(max_workers=3)`.

### 2a. Rules (`tools.py:145-222`)
**File:** `src/tools.py` → `rule_extract(text)`

**How it works:** Pure regex matching. No ML, no LLM. Runs on normalized text.

**What it finds:**
| Pattern | Type | Confidence | Regex example |
|---|---|---|---|
| Drug + dose + unit | `THUỐC` | 0.98 | `metoprolol 25mg` matches `DRUG_PATTERN` |
| Test names | `TÊN_XÉT_NGHIỆM` | 0.95 | `ecg`, `điện tâm đồ`, `siêu âm tim` from `TEST_PATTERNS` |
| Lab values | `KẾT_QUẢ_XÉT_NGHIỆM` | 0.95 | `troponin 0.01` from `LAB_VALUE_PATTERNS` |
| Normal results | `KẾT_QUẢ_XÉT_NGHIỆM` | 0.90 | `bình thường`, `không ghi nhận gì bất thường` from `NORMAL_RESULTS` |

**Return:** `{"mentions": [...], "assertion_cues": [...]}`

Each mention: `{text, position: [start, end], type, confidence, source: "rule"}`

**Strengths:** High precision, predictable output, exact positions.
**Weaknesses:** Only finds what the regexes match. Misses symptoms, diagnoses, lowercase entities.

### 2b. HealthBERT (`agents/worker_agents.py:46-76`)
**File:** `src/agents/worker_agents.py` → `_run_healthbert(text)`

**How it works:**
1. Regex finds candidate phrases: 3-6 word sequences starting with uppercase or lowercase letter after a separator
   ```
   Pattern: (?:^|[\s,;:\(\)"])([A-ZÀ-ỹa-zà-ỹ][a-zà-ỹ]+(?:\s+[a-zà-ỹ]+){2,5})(?=[\s,;:\-\.\)]|$)
   ```
2. Each candidate is classified by ViHealthBERT embedding similarity to type prototypes
3. If embedding says `KHÔNG_PHẢI` (matches NON_MEDICAL prototype), the entity is skipped
4. Keyword fallback runs next for drug patterns, test results, test names, diagnosis keywords
5. Diagnosis keywords (`CHẨN_ĐOÁN`) run LAST, after embedding filter

**Classification order** (`healthbert.py:235-282`):
```
1. Drug keywords (THUỐC) — highest precision, never filtered
2. Test result keywords (KẾT_QUẢ_XÉT_NGHIỆM) — "bình thường", "không ghi nhận"
3. Test name keywords (TÊN_XÉT_NGHIỆM) — "ecg", "x-quang"
4. ViHealthBERT embedding (NON_MEDICAL filter + type classification)
5. Diagnosis keywords (CHẨN_ĐOÁN) — "viêm", "bệnh" (after non-medical filter)
6. Embedding fallback (if score > 0.3)
7. Default: TRIỆU_CHỨNG
```

**Prototype examples** (`healthbert.py:62-110`):
| Type | Examples |
|---|---|
| THUỐC | metoprolol 25mg, aspirin 325mg, amoxicillin 500mg |
| TRIỆU_CHỨNG | đau ngực, khó thở, căng thẳng, đánh trống ngực, thắt chặt ngực |
| CHẨN_ĐOÁN | tăng huyết áp, viêm phổi, suy tim, nhồi máu cơ tim |
| TÊN_XÉT_NGHIỆM | chụp X-quang, điện tâm đồ, siêu âm tim |
| KẾT_QUẢ_XÉT_NGHIỆM | bình thường, không ghi nhận bất thường, troponin 0.01 |
| NON_MEDICAL | Lý do nhập viện, Mức độ nghiêm trọng, Mất việc làm |

**Return:** List of mentions: `{text, position: [start, end], type, type_candidates: [{label, score}]}`

**Known limitations:**
- Regex captures 3-6 word sequences that can CROSS entity boundaries (e.g., "ecg bình thường" is one capture)
- These boundary-crossing captures are filtered out by comparing with Rules positions

### 2c. Qwen (LLM NER) (`ner_tool.py`)
**File:** `src/ner_tool.py` → `ner_and_type(text, llm)`

**How it works:**
1. Chunks text into 500-char windows with 50-char overlap
2. Each chunk is sent to the LLM with `NER_PROMPT`
3. LLM returns JSON array of entities: `[{text, type}]`
4. Entity positions found by text search in original text (not LLM)
5. Type verified by LLM (second call for ambiguous cases)

**Only runs when:** `--llm runpod` or `--llm local` is set.

**Return:** `{"mentions": [...], "errors": [...]}`

**Strengths:** High recall, understands context, finds lowercase entities.
**Weaknesses:** Slow (API calls), can hallucinate, needs GPU for local model.

---

## 3. Flatten + Filter (`agents/worker_agents.py:146-170`)

**What happens:**
1. All mentions from all 3 models are collected into one list
2. Each mention gets `extraction_source` tag ("rule", "qwen", or "healthbert")
3. **Boundary-crossing filter:** If a healthbert mention overlaps with a rules mention AND starts at or before the rules mention's start position, it's REMOVED. This prevents healthbert from spanning across entity boundaries.

```
Example: healthbert "ecg bình thường" [2331,2346] overlaps rules "ecg" [2331,2334]
         → healthbert starts AT rules start (2331 <= 2331) → REMOVED
         → "ecg" and "bình thường" survive as separate entities

Example: healthbert "ngực không ghi nhận gì bất" [13,39] overlaps rules "không ghi nhận..." [18,47]
         → healthbert starts BEFORE rules (13 <= 18) → REMOVED
         → rules entity survives intact
```

---

## 4. Voting Merge (`voting.py:28-144`)

**`group_mentions(all_mentions)`** — Groups mentions by position overlap (>0 chars or <3 chars apart).

**`_merge_group(group)`** — Merges a group into one entity:
- **Position:** Widest span from all mentions in group
- **Text:** Most common text variant (by votes)
- **Type:** Most common type (by votes)
- **Confidence:** Based on agreement:
  | Models agreeing | Confidence |
  |---|---|
  | 3/3 | 0.95 |
  | 2/3 | 0.80 |
  | 1/3 (or 1/1) | 0.50 |
- **needs_review:** True if max_type_votes < 2 (disagreement)
- **voters:** Dict of which models contributed, e.g., `{"rule": 1, "healthbert": 2}`

**`llm_resolve_conflicts(mentions, text, llm)`** — If any mention has `needs_review=True`, sends to LLM for resolution. LLM votes on type. Updates confidence to 0.85 if resolved.

**Output format per merged mention:**
```json
{
  "text": "đánh trống ngực",
  "position": [497, 512],
  "type": "TRIỆU_CHỨNG",
  "confidence": 0.80,
  "vote_count": 2,
  "max_type_votes": 2,
  "voters": {"qwen": 1, "rule": 1},
  "type_votes": {"TRIỆU_CHỨNG": 2},
  "needs_review": false
}
```

---

## 5. Position Fix (`position_fix.py`)

**Input:** merged mentions with approx positions
**Output:** corrected positions by searching entity text in original text

**Search strategies (in order):**
1. Exact string match
2. Case-insensitive match
3. Fuzzy match (find first+last token, return span)

**Sets:** `position_fixed: true/false` on each mention.

---

## 6. LLM Verify (`agents/worker_agents.py:173-181`)

For each mention with `needs_review=True` AND NOT already `llm_resolved`:
- Sends to LLM with context for type verification
- LLM can correct the type
- Sets `llm_corrected: true` if changed

**Only runs when LLM is available.**

---

## 7. Assertion (`agents/worker_agents.py:229-259`)

For each mention, determines:
- `isNegated` — "không", "chưa", "không có" in context
- `isFamily` — "bố", "mẹ", "người nhà" in context
- `isHistorical` — "tiền sử", "trước đây", "đã điều trị" in context

**Method:** Rule-based for all mentions. LLM-based for CHẨN_ĐOÁN/THUỐC if LLM available.

**Context window:** 60 chars before entity for rules, 150 chars for LLM.

---

## 8. Linking (`agents/worker_agents.py:291-305`)

For CHẨN_ĐOÁN and THUỐC entities:
- CHẨN_ĐOÁN → search ICD-10 knowledge base
- THUỐC → search RxNorm knowledge base
- Candidates added to mention

**Linking pipeline** (from `linker_agent.py`):
1. Fuzzy keyword search → top candidates
2. ViPubMedDeBERTa embedding reranking
3. LLM picks best match
4. Agentic loop if LLM unsure

---

## 9. Verification (`agents/worker_agents.py:303-325`)

**`verify_all()`** from `verifier_agent.py`:
- Validates schema (type, position, assertions)
- Retries assertion/linking for borderline cases
- Drops invalid mentions

**Rules-based dedup:** Keeps highest-confidence mention per exact `[start, end]` position.

---

## 10. Output Format

```json
{
  "note_id": "1",
  "mentions": [
    {
      "text": "đánh trống ngực",
      "position": [497, 512],
      "type": "TRIỆU_CHỨNG",
      "confidence": 0.80,
      "voters": {"qwen": 1, "rule": 1},
      "assertions": {
        "isNegated": false,
        "isFamily": false,
        "isHistorical": false
      }
    }
  ],
  "validation": {
    "is_valid": true,
    "errors": [],
    "warnings": []
  }
}
```

---

## Current Known Issues (Rules-only mode)

| Problem | Root cause |
|---|---|
| 2-word entities missed (Căng thẳng, hay nôn) | HealthBERT regex requires 3+ words `{2,5}` |
| Drug names split from doses | Rules capture `metoprolol 25mg`, HealthBERT captures wider context → overlap merge loses dose |
| Overlap between test-name and result | Rules find "ecg" and "bình thường" separately; healthbert merges them (filtered but position overlap causes issues) |
| No symptom/diagnosis extraction without LLM | Rules only find drugs, tests, results. Symptoms need Qwen or HealthBERT |

---

## Files Reference

| File | What it does |
|---|---|
| `src/tools.py` | Preprocessing, rule extraction regexes (drugs, tests, results), output validation |
| `src/ner_tool.py` | Qwen LLM NER extraction (chunked, position-by-search) |
| `src/healthbert.py` | ViHealthBERT prototype-based classifier + keyword fallback |
| `src/voting.py` | 3-model voting merge + LLM conflict resolution |
| `src/position_fix.py` | Re-locate entity positions via text search |
| `src/assertion_agent.py` | Rule + LLM assertion (negation, family, historical) |
| `src/linker_agent.py` | ICD-10/RxNorm linking with agentic loop |
| `src/verifier_agent.py` | Schema validation and fix |
| `src/embedding_linking.py` | ViPubMedDeBERTa embedding reranking |
| `src/orchestrator.py` | Pipeline entry point (delegates to agentic orchestrator) |
| `src/agents/__init__.py` | Tool, Agent, AgentContext base classes |
| `src/agents/orchestrator.py` | Agentic orchestrator (runs extraction → assertion → linking → verification) |
| `src/agents/worker_agents.py` | ExtractionAgent, AssertionAgent, LinkingAgent, VerifierAgent |
| `src/prompt/__init__.py` | All LLM prompts |
