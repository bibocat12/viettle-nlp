#!/usr/bin/env python3
"""Synthetic Vietnamese clinical text generator — agentic ICD-10/RxNorm linking.

The LLM uses **JSON-mode tool simulation** (Featherless recommended approach
for non-native tool-calling models).  Tools are described in the system prompt;
the model outputs structured JSON to request tool execution, and our code
loops until a final answer is produced.

Multi-code candidates are supported — an entity may have several valid codes
in `candidates`, with `preferred_candidate` set only when context disambiguates.

Usage:
    python3 src/train/generate_synthetic_data.py --num 100 --output data_train/synthetic.jsonl
    python3 src/train/generate_synthetic_data.py --verify-only data_train/synthetic.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from featherless_llm import get_featherless_llm
from position_fix import fix_positions

# ---------------------------------------------------------------------------
# Pools (symptoms & labs — no KB lookup needed)
# ---------------------------------------------------------------------------

SYMPTOM_POOL = [
    "ho khan", "ho có đờm", "ho ra máu", "khó thở", "thở khò khè",
    "đau ngực", "đau đầu", "đau bụng", "đau thượng vị", "đau hạ sườn phải",
    "đau khớp", "đau lưng", "đau cơ", "sốt cao", "sốt nhẹ", "rét run",
    "vã mồ hôi", "mệt mỏi", "chán ăn", "sụt cân", "buồn nôn", "nôn",
    "tiêu chảy", "táo bón", "đầy bụng", "ợ hơi", "ợ chua", "khó nuốt",
    "chóng mặt", "hoa mắt", "ù tai", "mất ngủ", "lo âu", "hồi hộp",
    "đánh trống ngực", "phù chân", "tê bì tay chân", "cứng khớp buổi sáng",
    "yếu cơ", "co giật", "tiểu nhiều", "tiểu đau", "tiểu rắt",
    "tăng huyết áp", "huyết áp thấp", "nhịp tim nhanh", "nhịp tim chậm",
]

LAB_POOL = [
    {"name": "WBC", "unit": "K/μL", "values": ["14.43", "12.1", "9.8", "6.5", "3.2", "11.2"]},
    {"name": "RBC", "unit": "M/μL", "values": ["5.2", "4.8", "3.9", "4.1", "5.5"]},
    {"name": "Hb", "unit": "g/dL", "values": ["14.5", "13.2", "10.1", "8.9", "15.0"]},
    {"name": "PLT", "unit": "K/μL", "values": ["256", "189", "450", "98", "320"]},
    {"name": "NEUT%", "unit": "%", "values": ["76.4", "71.2", "68.5", "55.0", "82.1"]},
    {"name": "LYM%", "unit": "%", "values": ["20.1", "25.3", "18.5", "30.0", "12.4"]},
    {"name": "CRP", "unit": "mg/dL", "values": ["8.4", "12.2", "3.1", "0.8", "25.0"]},
    {"name": "ALT", "unit": "U/L", "values": ["42", "78", "35", "120", "25"]},
    {"name": "AST", "unit": "U/L", "values": ["38", "65", "28", "98", "22"]},
    {"name": "Creatinin", "unit": "mg/dL", "values": ["1.2", "0.9", "1.8", "0.8", "2.1"]},
    {"name": "Ure", "unit": "mg/dL", "values": ["38", "45", "22", "55", "18"]},
    {"name": "Glico", "unit": "%", "values": ["7.2", "6.5", "8.1", "5.8", "9.0"]},
    {"name": "Na+", "unit": "mmol/L", "values": ["138", "142", "135", "140", "146"]},
    {"name": "K+", "unit": "mmol/L", "values": ["3.8", "4.2", "3.5", "5.0", "4.5"]},
]

# ---------------------------------------------------------------------------
# System prompt (describes tools available via JSON output)
# ---------------------------------------------------------------------------

AGENT_SYSTEM = """Bạn là hệ thống sinh dữ liệu huấn luyện tiếng Việt cho bài toán trích xuất thực thể y khoa.

Bạn có 2 công cụ để tra cứu mã chuẩn. Khi cần tra mã, hãy xuất JSON với định dạng:
{"tool": "search_icd10", "args": {"query": "tên bệnh"}}
{"tool": "search_rxnorm", "args": {"query": "tên thuốc"}}
{"tool": "get_icd10_code", "args": {"code": "M10.0"}}
{"tool": "get_rxnorm_concept", "args": {"rxcui": "8601"}}

Khi đã có đủ thông tin, xuất kết quả cuối cùng dưới dạng JSON với "text" và "entities".

QUY TRÌNH BẮT BUỘC:
1. Chọn 1-2 bệnh và 1-2 thuốc.
2. Tra mã ICD-10 / RxNorm bằng công cụ.
3. Viết đoạn text y khoa tiếng Việt tự nhiên (kiểu ghi chú bác sĩ / giấy ra viện).
4. Xuất entities với assertions, candidates, positions.

CÁC LOẠI THỰC THỂ:
- CHẨN_ĐOÁN: bệnh, chẩn đoán → candidates là list ICD-10
- THUỐC: tên thuốc, có thể kèm liều → candidates là list RxNorm
- TRIỆU_CHỨNG: triệu chứng → candidates rỗng
- TÊN_XÉT_NGHIỆM: xét nghiệm → candidates rỗng
- KẾT_QUẢ_XÉT_NGHIỆM: kết quả xét nghiệm → candidates rỗng

QUY TẮC CANDIDATES (đọc kỹ):
1. Nếu mention chỉ khớp 1 mã → candidates=[mã đó], preferred_candidate=mã đó, ambiguous=false
2. Nếu mention có NHIỀU mã hợp lệ (VD: trào ngược dạ dày → cả K21.0 và K21.9):
   candidates=[tất cả mã đúng], preferred_candidate=null, ambiguous=true
3. Nếu ngữ cảnh đủ mạnh để chọn 1 trong nhiều mã:
   candidates=[tất cả mã đúng], preferred_candidate=mã được chọn, ambiguous=false
4. KHÔNG BAO GIỜ tự bịa mã ngoài kết quả tra cứu.
5. KHÔNG loại bỏ mã hợp lệ để ép còn 1 candidate.

QUY TẮC ASSERTION:
- Chỉ dùng: ["isNegated", "isFamily", "isHistorical"]
- Chỉ áp dụng cho TRIỆU_CHỨNG, CHẨN_ĐOÁN, THUỐC.

YÊU CẦU TỐI THIỂU MỖI MẪU: 1 CHẨN_ĐOÁN + 1 THUỐC + 2 TRIỆU_CHỨNG
Có thể thêm: xét nghiệm, kết quả, assertion, thủ thuật.

ĐỊNH DẠNG KẾT QUẢ CUỐI CÙNG (CHỈ JSON, KHÔNG GIẢI THÍCH):
{
  "text": "đoạn văn bản y khoa...",
  "entities": [
    {
      "text": "tên thực thể",
      "position": [start, end],
      "type": "CHẨN_ĐOÁN | THUỐC | TRIỆU_CHỨNG | TÊN_XÉT_NGHIỆM | KẾT_QUẢ_XÉT_NGHIỆM",
      "assertions": [],
      "candidates": [],
      "gold_candidates": [],
      "preferred_candidate": null,
      "ambiguous": false,
      "candidate_notes": ""
    }
  ],
  "relations": []
}

Nếu không thể đảm bảo linking chính xác, trả về: {"error": "FAIL_LINKING"}
"""


def build_agent_user_prompt() -> str:
    """Build user message with pools to inspire the generation."""
    n_symp = min(len(SYMPTOM_POOL), 12)
    n_lab = min(len(LAB_POOL), 5)
    symp_sample = random.sample(SYMPTOM_POOL, n_symp)
    lab_sample = random.sample(LAB_POOL, n_lab)

    symp_str = "\n".join(f"  - {s}" for s in symp_sample)
    lab_str = "\n".join(f'  - {lb["name"]} ({lb["unit"]}): [{", ".join(lb["values"])}]' for lb in lab_sample)

    return f"""Sinh 1 mẫu dữ liệu y khoa tiếng Việt.

Triệu chứng tham khảo:
{symp_str}

Xét nghiệm tham khảo:
{lab_str}

Dùng công cụ tra cứu mã ICD-10 và RxNorm trước khi viết câu.
Chỉ dùng mã từ kết quả tra cứu. KHÔNG tự bịa mã.
Khi có kết quả, xuất JSON cuối cùng với "text" và "entities"."""


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def make_tool_registry(kb):
    def search_icd10(query: str):
        return [{"code": r["code"], "name": r["name"], "score": r["score"]}
                for r in kb.search_icd10(query, top_k=10)]

    def search_rxnorm(query: str):
        return [{"rxcui": r["rxcui"], "name": r["name"], "score": r["score"]}
                for r in kb.search_rxnorm(query, top_k=10)]

    def get_icd10_code(code: str):
        info = kb.get_icd10_code(code)
        return info if info else {"error": f"ICD-10 {code} not found"}

    def get_rxnorm_concept(rxcui: str):
        info = kb.get_rxnorm_concept(rxcui)
        return info if info else {"error": f"RxNorm {rxcui} not found"}

    return {
        "search_icd10": search_icd10,
        "search_rxnorm": search_rxnorm,
        "get_icd10_code": get_icd10_code,
        "get_rxnorm_concept": get_rxnorm_concept,
    }


# ---------------------------------------------------------------------------
# Parsing & validation
# ---------------------------------------------------------------------------


def parse_json_output(raw: str) -> dict | None:
    """Parse JSON from response (handles code fences, thinking tags)."""
    if not raw:
        return None
    text = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    text = re.sub(r'```(?:json)?\s*', '', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def validate_linking(sample: dict, kb, fix_pos: bool = True) -> list[str]:
    """Check entity codes exist in KB, types are valid."""
    errors = []
    if not sample:
        return ["empty sample"]
    text = sample.get("text", "")
    entities = sample.get("entities", [])

    if fix_pos and entities and text:
        sample["entities"] = fix_positions(entities, text)

    for ent in sample.get("entities", []):
        ent_text = ent.get("text", "")
        etype = ent.get("type", "")
        candidates = ent.get("candidates", [])
        preferred = ent.get("preferred_candidate")

        if etype == "CHẨN_ĐOÁN":
            if not candidates:
                errors.append(f"CHẨN_ĐOÁN '{ent_text}' has empty candidates")
            for code in candidates:
                if not kb.get_icd10_code(code):
                    errors.append(f"CHẨN_ĐOÁN '{ent_text}': invalid ICD-10 '{code}'")
            if preferred and preferred not in candidates:
                errors.append(f"CHẨN_ĐOÁN '{ent_text}': preferred '{preferred}' not in candidates")
        elif etype == "THUỐC":
            if not candidates:
                errors.append(f"THUỐC '{ent_text}' has empty candidates")
            for code in candidates:
                if not kb.get_rxnorm_concept(code):
                    errors.append(f"THUỐC '{ent_text}': invalid RxNorm '{code}'")
            if preferred and preferred not in candidates:
                errors.append(f"THUỐC '{ent_text}': preferred '{preferred}' not in candidates")
        elif etype in ("TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"):
            if candidates:
                errors.append(f"{etype} '{ent_text}' should have empty candidates, got {candidates}")

        for a in ent.get("assertions", []):
            if a not in ("isNegated", "isFamily", "isHistorical"):
                errors.append(f"'{ent_text}' has invalid assertion '{a}'")

        gold = ent.get("gold_candidates", [])
        for code in gold:
            if code not in candidates:
                errors.append(f"gold '{code}' not in candidates for '{ent_text}'")

    return errors


def count_entity_types(sample: dict) -> dict:
    counts = {}
    for ent in sample.get("entities", []):
        t = ent.get("type", "UNKNOWN")
        counts[t] = counts.get(t, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Generator — JSON-mode tool simulation loop
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pool-based fast generation (no tool calls, ~38s/sample)
# ---------------------------------------------------------------------------

POOL_SYSTEM = """Bạn là bác sĩ lâm sàng, hãy viết 1 đoạn ghi chú y khoa tiếng Việt ngắn (3-5 câu).

Yêu cầu:
- Viết đoạn văn tự nhiên kiểu bệnh án / giấy ra viện / hồ sơ bệnh án.
- Trong đoạn văn PHẢI chứa các thực thể từ pools dưới đây.
- Xuất JSON với text và entities (entity, position, type, assertions, candidates).

CÁC LOẠI THỰC THỂ:
- CHẨN_ĐOÁN: bệnh, chẩn đoán → candidates là list ICD-10 tương ứng
- THUỐC: tên thuốc, có thể kèm liều → candidates là list RxNorm tương ứng
- TRIỆU_CHỨNG: triệu chứng, dấu hiệu → candidates rỗng
- TÊN_XÉT_NGHIỆM: xét nghiệm → candidates rỗng
- KẾT_QUẢ_XÉT_NGHIỆM: kết quả xét nghiệm → candidates rỗng

QUY TẮC:
1. CHỈ dùng CHẨN_ĐOÁN từ diagnosis_pool.
2. CHỈ dùng THUỐC từ drug_pool.
3. candidates PHẢI chứa đúng code tương ứng từ pool.
4. Nếu một bệnh có nhiều code hợp lệ: candidates = [tất cả code], ambiguous = true, preferred_candidate = null
5. assertions chỉ dùng: ["isNegated", "isFamily", "isHistorical"]
6. Tối thiểu: 1 CHẨN_ĐOÁN + 1 THUỐC + 2 TRIỆU_CHỨNG
7. position là [start, end] chính xác trong text.
8. KHÔNG tự bịa code ngoài pool.
9. entity.text KHÔNG được chứa mã số, ID, code trong ngoặc.
   Đúng: "tăng sản niêm mạc miệng"  Sai: "tăng sản niêm mạc miệng (K13.6)"
   Đúng: "WBC"  Sai: "WBC (K/μL)"  — đơn vị là phần riêng.
10. Phải có xét nghiệm + kết quả nếu pool có lab:
    - TÊN_XÉT_NGHIỆM = chỉ tên, VD: "WBC", "CRP", "ALT"
    - KẾT_QUẢ_XÉT_NGHIỆM = giá trị + đơn vị, VD: "14.43 K/μL", "12.2 mg/dL"

ĐỊNH DẠNG OUTPUT:
{
  "text": "...",
  "entities": [
    {
      "text": "...",
      "position": [start, end],
      "type": "...",
      "assertions": [],
      "candidates": [],
      "gold_candidates": [],
      "preferred_candidate": null,
      "ambiguous": false
    }
  ],
  "relations": []
}

Nếu không đảm bảo, trả: {"error": "FAIL_LINKING"}
"""


def build_diagnosis_pool(kb, min_len=25, max_len=110, sample_size=20):
    pool = []
    for code, info in kb.icd10.items():
        name = info.get("vietnamese_name", "") or info.get("english_name", "")
        if min_len < len(name) < max_len:
            pool.append({"canonical": name, "codes": [code]})
    random.shuffle(pool)
    return pool[:sample_size]


def build_drug_pool(kb, sample_size=10):
    pool = []
    for cui, c in kb.rxnorm_concepts.items():
        label = c.get("prefLabel", "")
        if label:
            pool.append({"canonical": label, "codes": [cui]})
    random.shuffle(pool)
    return pool[:sample_size]


def build_pool_user_prompt(kb) -> str:
    """Build a user prompt with pools injected directly (no tool calls)."""
    dx_pool = build_diagnosis_pool(kb)
    rx_pool = build_drug_pool(kb)
    n_symp = min(len(SYMPTOM_POOL), 12)
    n_lab = min(len(LAB_POOL), 5)
    symp_sample = random.sample(SYMPTOM_POOL, n_symp)
    lab_sample = random.sample(LAB_POOL, n_lab)

    def fmt_pool(items):
        return "\n".join(
            f'  - {{"text": "{item["canonical"]}", "codes": {json.dumps(item["codes"])}}}'
            for item in items
        )

    symp_str = "\n".join(f"  - {s}" for s in symp_sample)
    lab_lines = []
    for lb in lab_sample:
        vals = ", ".join(lb["values"])
        lab_lines.append(f'  - {lb["name"]} ({lb["unit"]}): [{vals}]')

    return f"""Hãy viết 1 đoạn ghi chú y khoa tiếng Việt ngắn (3-5 câu) có chứa các thực thể sau.

diagnosis_pool (chọn 1-2 cái):
{fmt_pool(dx_pool)}

drug_pool (chọn 1-2 cái):
{fmt_pool(rx_pool)}

Có thể dùng triệu chứng (chọn từ đây hoặc tự sinh):
{symp_str}

Có thể dùng xét nghiệm:
{chr(10).join(lab_lines)}

Chỉ trả JSON với text và entities."""


# ---------------------------------------------------------------------------
# Generator (dual-mode: agent tool-calling / pool fast)
# ---------------------------------------------------------------------------


class SyntheticGenerator:
    """Generate samples — mode='agent' (tool-calling) or 'pool' (fast injection)."""

    def __init__(self, kb, llm=None, verify=True, mode="pool",
                 openai_client=None, openai_model="qwen-plus",
                 openai_base_url=None, openai_api_key=None):
        self.kb = kb
        self.llm = llm or get_featherless_llm()
        self.verify = verify
        self.mode = mode
        self._openai_client = openai_client
        self._openai_model = openai_model
        self._registry = make_tool_registry(kb)

    def _execute_tool(self, tool_call: dict) -> dict | None:
        """Execute a single tool call. Returns result dict or None."""
        tool_name = tool_call.get("tool")
        args = tool_call.get("args", {})
        handler = self._registry.get(tool_name)
        if handler:
            try:
                return handler(**args)
            except Exception as e:
                return {"error": str(e)}
        return {"error": f"Unknown tool: {tool_name}"}

    def generate_one(self, temperature: float = 0.7, max_retries: int = 3) -> dict | None:
        """Generate one sample. Dispatches to agent or pool mode."""
        if self.mode == "pool":
            return self._generate_one_pool(temperature, max_retries)
        return self._generate_one_agent(temperature, max_retries)

    def _generate_one_pool(self, temperature: float = 0.7, max_retries: int = 3) -> dict | None:
        """Fast generation: inject pools directly, no tool calls."""
        for attempt in range(max_retries):
            user_prompt = build_pool_user_prompt(self.kb)

            if self._openai_client:
                # Use raw OpenAI client (DashScope / Qwen API)
                try:
                    resp = self._openai_client.chat.completions.create(
                        model=self._openai_model,
                        max_tokens=4096,
                        temperature=temperature,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": POOL_SYSTEM},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    raw = resp.choices[0].message.content or ""
                except Exception as e:
                    print(f"Qwen API error: {e}", file=sys.stderr, flush=True)
                    continue
            else:
                raw = self.llm.generate(
                    system_prompt=POOL_SYSTEM,
                    user_prompt=user_prompt,
                    max_tokens=4096,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
            if not raw:
                continue
            parsed = parse_json_output(raw)
            if not parsed:
                continue
            if parsed.get("error") == "FAIL_LINKING":
                continue
            if self.verify:
                errors = validate_linking(parsed, self.kb)
                if errors:
                    continue
            return parsed
        return None

    def _generate_one_agent(self, temperature: float = 0.7, max_retries: int = 3) -> dict | None:
        """Generate one sample via JSON-mode tool simulation."""
        user_prompt = build_agent_user_prompt()

        for attempt in range(max_retries):
            messages = [
                {"role": "system", "content": AGENT_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]
            max_turns = 8

            for turn in range(max_turns):
                # Merge messages into a single prompt for JSON mode
                conv = "\n".join(
                    f"<|{m['role']}|>\n{m['content']}" for m in messages
                )
                combined_prompt = f"{conv}\n<|assistant|>\n"

                raw = self.llm.generate(
                    system_prompt="Bạn là hệ thống sinh dữ liệu y khoa tiếng Việt. Luôn xuất JSON hợp lệ.",
                    user_prompt=combined_prompt,
                    max_tokens=4096,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                if not raw:
                    break

                parsed = parse_json_output(raw)
                if not parsed:
                    break

                # Check for tool call
                if "tool" in parsed:
                    result = self._execute_tool(parsed)
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                    continue

                # Check for final output
                if "text" in parsed:
                    if self.verify:
                        errors = validate_linking(parsed, self.kb)
                        if errors:
                            break  # retry
                    return parsed

                # Check for error
                if parsed.get("error") == "FAIL_LINKING":
                    break

                # Unknown format — continue

            # If we got here without returning, try again

        return None

    def generate_batch(self, n: int, temperature: float = 0.7,
                       max_retries: int = 3, parallel: int = 1,
                       output_path: str = None, save_every: int = 25,
                       verbose: bool = True) -> list[dict]:
        if parallel <= 1:
            return self._generate_sequential(
                n, temperature, max_retries, verbose,
                output_path=output_path, save_every=save_every,
            )
        return self._generate_parallel(n, temperature, max_retries, parallel, verbose)

    def _generate_sequential(self, n: int, temperature: float,
                             max_retries: int, verbose: bool,
                             output_path: str = None,
                             save_every: int = 25) -> list[dict]:
        samples = []
        t0 = time.time()
        for i in range(n):
            sample = self.generate_one(temperature=temperature, max_retries=max_retries)
            if sample:
                samples.append(sample)
                if verbose:
                    print(f"[{i+1}/{n}] OK  types={count_entity_types(sample)}", flush=True)
            else:
                if verbose:
                    print(f"[{i+1}/{n}] FAIL", flush=True)

            # Periodic save
            if output_path and samples and (i + 1) % save_every == 0:
                with open(output_path, "a") as f:
                    for s in samples[-save_every:]:
                        f.write(json.dumps(s, ensure_ascii=False) + "\n")
                if verbose:
                    print(f"  [checkpoint] saved {len(samples)} so far to {output_path}", flush=True)

        # Final save
        if output_path and samples:
            remaining = len(samples) % save_every
            if remaining > 0:
                with open(output_path, "a") as f:
                    for s in samples[-remaining:]:
                        f.write(json.dumps(s, ensure_ascii=False) + "\n")
            if verbose:
                print(f"  [final] saved {len(samples)} samples to {output_path}", flush=True)

        if verbose:
            elapsed = time.time() - t0
            print(f"\nGenerated {len(samples)}/{n} valid samples in {elapsed:.1f}s")
        return samples

    def _generate_parallel(self, n: int, temperature: float,
                           max_retries: int, workers: int,
                           verbose: bool) -> list[dict]:
        """Parallel generation via ThreadPoolExecutor."""
        t0 = time.time()
        samples = []
        fail_count = 0
        lock = __import__("threading").Lock()

        def worker(job_id):
            nonlocal fail_count
            sample = self.generate_one(temperature=temperature, max_retries=max_retries)
            with lock:
                if sample:
                    samples.append(sample)
                    if verbose:
                        print(f"[{job_id+1}/{n}] OK  types={count_entity_types(sample)}", flush=True)
                else:
                    fail_count += 1
                    if verbose:
                        print(f"[{job_id+1}/{n}] FAIL", flush=True)

        with ThreadPoolExecutor(max_workers=workers) as exe:
            futures = [exe.submit(worker, i) for i in range(n)]
            for f in as_completed(futures):
                f.result()  # propagate exceptions

        if verbose:
            elapsed = time.time() - t0
            print(f"\nGenerated {len(samples)}/{n} valid samples in {elapsed:.1f}s ({workers} workers)")
        return samples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Synthetic clinical text generator")
    parser.add_argument("--num", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output", default="data_train/synthetic.jsonl", help="Output JSONL path")
    parser.add_argument("--verify-only", default=None, help="Verify existing JSONL file instead of generating")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", default="pool", choices=["pool", "agent"],
                        help="pool=faster (no tool calls), agent=tool-calling (higher quality)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel workers (default=1, sequential)")
    parser.add_argument("--provider", default="featherless", choices=["featherless", "qwen"],
                        help="LLM provider: featherless (DeepSeek/Qwen via Featherless) or qwen (DashScope)")
    parser.add_argument("--qwen-model", default="qwen-plus",
                        help="Qwen model name (only used with --provider qwen). Default: qwen-plus")
    args = parser.parse_args()

    random.seed(args.seed)
    from kb import KnowledgeBase
    kb = KnowledgeBase()
    kb.load()
    print(f"KB loaded: {len(kb.icd10)} ICD-10, {len(kb.rxnorm_concepts)} RxNorm", flush=True)

    if args.verify_only:
        passed = failed = 0
        with open(args.verify_only) as f:
            for i, line in enumerate(f):
                sample = json.loads(line)
                errors = validate_linking(sample, kb)
                if errors:
                    failed += 1
                    if failed <= 5:
                        print(f"#{i}: {'; '.join(errors[:3])}")
                else:
                    passed += 1
        print(f"\nVerified {passed + failed} samples: {passed} OK, {failed} FAILED")
        return

    from openai import OpenAI as _OpenAI

    openai_client = None
    qwen_model = None

    if args.provider == "qwen":
        qwen_model = args.qwen_model
        api_key = os.getenv("QWEN_API") or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            print("ERROR: QWEN_API or DASHSCOPE_API_KEY not set in .env", file=sys.stderr)
            sys.exit(1)
        openai_client = _OpenAI(
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
        )
        print(f"Provider: qwen | model={qwen_model} | mode={args.mode} | parallel={args.parallel}", flush=True)
    else:
        llm = get_featherless_llm()
        if not llm.is_available():
            print("ERROR: Featherless LLM not available. Check FEATHERLESS_API_KEY in .env", file=sys.stderr)
            sys.exit(1)
        print(f"Provider: featherless | model={llm.model} | mode={args.mode} | parallel={args.parallel}", flush=True)

    gen = SyntheticGenerator(
        kb, llm=llm if args.provider == "featherless" else None,
        verify=True, mode=args.mode,
        openai_client=openai_client, openai_model=qwen_model,
    )
    out_path = str(Path(args.output))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    samples = gen.generate_batch(
        args.num, temperature=args.temperature,
        parallel=args.parallel,
        output_path=out_path, save_every=25,
    )

    if not samples:
        print("No samples generated.")
        sys.exit(1)

    # Save all samples
    with open(out_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved {len(samples)} samples to {out_path}", flush=True)

    all_types = {}
    for s in samples:
        for t, c in count_entity_types(s).items():
            all_types[t] = all_types.get(t, 0) + c
    print("Entity distribution:", json.dumps(all_types, ensure_ascii=False))


if __name__ == "__main__":
    main()
