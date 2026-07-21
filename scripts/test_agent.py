#!/usr/bin/env python3
"""Test the GRPO model with tool calling and validate answer correctness.

Pipeline:
  1. Load test samples from dataset
  2. Call model (vLLM endpoint) with tool calling
  3. Model calls search_icd10 / search_rxnorm tools → server executes against KB
  4. Model produces final entity JSON
  5. Validate: check for missing, extra, wrong entities + candidate accuracy

Usage:
    python3 scripts/test_agent.py \
      --endpoint http://localhost:8000/v1 \
      --dataset data_train/grpo_combined-100.jsonl \
      --num-samples 20
"""

import argparse, json, os, re, sys, time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import requests
from kb import KnowledgeBase

# ── Tools definition (same as training) ────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_icd10",
            "description": "Tra cứu mã ICD-10 theo tên bệnh tiếng Việt hoặc tiếng Anh.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Tên bệnh / chẩn đoán cần tra (tiếng Việt). VD: tăng huyết áp, lao phổi",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_rxnorm",
            "description": "Tra cứu mã RxNorm theo tên thuốc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Tên thuốc cần tra. VD: metformin, atorvastatin"}
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = """Bạn là hệ thống trích xuất thực thể y khoa tiếng Việt.
Nhiệm vụ của bạn là trích xuất tất cả các thực thể y tế từ văn bản lâm sàng.

CÁC LOẠI THỰC THỂ:
- THUỐC: Tên thuốc, có thể kèm liều lượng.
- TRIỆU_CHỨNG: Triệu chứng, dấu hiệu bệnh.
- CHẨN_ĐOÁN: Bệnh, chẩn đoán, hội chứng.
- TÊN_XÉT_NGHIỆM: Xét nghiệm, thủ thuật chẩn đoán.
- KẾT_QUẢ_XÉT_NGHIỆM: Kết quả xét nghiệm, chỉ số.

QUY TRÌNH:
1. Đọc văn bản lâm sàng.
2. Gọi search_icd10 cho mỗi bệnh/chẩn đoán tìm được.
3. Gọi search_rxnorm cho mỗi thuốc tìm được.
4. Tổng hợp kết quả thành JSON array entities.

ĐỊNH DẠNG ĐẦU RA:
[{"entity": "tên thực thể", "type": "LOẠI", "candidates": ["MÃ"]}]

KHÔNG được tự bịa mã ngoài kết quả tra cứu.
"""


# ── Tool execution ──────────────────────────────────────────────────────

def execute_tool(kb, tool_name: str, args: dict) -> list:
    """Execute a tool call against the local KB."""
    if tool_name == "search_icd10":
        return [{"code": r["code"], "name": r["name"], "score": r["score"]}
                for r in kb.search_icd10(args.get("query", ""), top_k=8)]
    elif tool_name == "search_rxnorm":
        return [{"rxcui": r["rxcui"], "name": r["name"], "score": r["score"]}
                for r in kb.search_rxnorm(args.get("query", ""), top_k=8)]
    return []


# ── Agent loop ──────────────────────────────────────────────────────────

def run_agent(endpoint: str, model: str, text: str, kb) -> dict:
    """Run the model with tool calling and return parsed entities."""
    client = requests.Session()

    # Auth: use QWEN_API if endpoint is DashScope, otherwise FIREWORKS_API_KEY
    api_key = os.getenv("QWEN_API") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("FIREWORKS_API_KEY") or ""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}]
    max_turns = 6

    for turn in range(max_turns):
        resp = client.post(
            f"{endpoint}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "tools": TOOLS,
                "temperature": 0.1,
                "max_tokens": 1024,
            },
            headers=headers,
            timeout=120,
        )
        data = resp.json()

        # Handle error responses
        if "error" in data:
            yield {"type": "error", "content": str(data["error"])}
            return
        if "choices" not in data:
            yield {"type": "error", "content": f"Unexpected response: {str(data)[:200]}"}
            return

        choice = data["choices"][0]
        msg = choice["message"]

        # Handle both 'tool_calls' and variations
        tool_calls = msg.get("tool_calls", []) or []
        if not tool_calls and "function_call" in msg:
            tool_calls = [{"function": msg["function_call"]}]

        if tool_calls:
            messages.append(msg)
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"])
                result = execute_tool(kb, fn_name, fn_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": fn_name,
                    "content": json.dumps(result, ensure_ascii=False),
                })
                yield {"type": "tool_call", "name": fn_name, "args": fn_args, "result": result}
            continue

        content = msg.get("content", "")
        yield {"type": "final", "content": content}
        return

    yield {"type": "error", "content": "MAX_TURNS_EXCEEDED"}


# ── Parsing ─────────────────────────────────────────────────────────────

def parse_entities(text: str) -> list[dict]:
    """Parse entity JSON from model output."""
    if not text:
        return []
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        entities = json.loads(m.group())
        return entities if isinstance(entities, list) else []
    except json.JSONDecodeError:
        return []


# ── Validation ──────────────────────────────────────────────────────────

def validate(pred_entities: list, gt_entities: list) -> dict:
    """Validate predictions against ground truth.

    Returns: {
        "correct": int, "missing": int, "extra": int, "wrong_type": int,
        "total_gt": int, "total_pred": int,
        "precision": float, "recall": float, "f1": float,
        "details": list[str],
        "by_type": {type: {correct, missing, extra, wrong_type}},
    }
    """
    def get_text(e):
        return (e.get("entity") or e.get("text") or "").lower().strip()

    def get_type(e):
        return e.get("type", "")

    gt_set = {}  # (text, type) -> entity
    for e in gt_entities:
        key = (get_text(e), get_type(e).upper())
        gt_set[key] = e

    pred_set = {}
    for e in pred_entities:
        key = (get_text(e), get_type(e).upper())
        pred_set[key] = e

    details = []
    type_stats = defaultdict(lambda: {"correct": 0, "missing": 0, "extra": 0, "wrong_type": 0})

    # Check GT entities
    correct, missing, wrong_type = 0, 0, 0
    for key, gt_e in gt_set.items():
        gt_t = key[1]
        # Check if same text exists with ANY type in pred
        pred_text_match = [k for k in pred_set if k[0] == key[0]]

        if key in pred_set:
            correct += 1
            type_stats[gt_t]["correct"] += 1
            # Validate candidates if present
            gt_candidates = set(gt_e.get("candidates", []))
            pred_candidates = set(pred_set[key].get("candidates", []))
            if gt_candidates and not pred_candidates:
                details.append(f"MISSING_CANDIDATES: '{key[0]}' ({key[1]}) — GT has {gt_candidates} but pred has none")
            elif gt_candidates:
                overlap = gt_candidates & pred_candidates
                if len(overlap) == 0:
                    details.append(f"WRONG_CANDIDATES: '{key[0]}' ({key[1]}) — GT={gt_candidates}, pred={pred_candidates}")
                elif len(overlap) < len(gt_candidates):
                    details.append(f"PARTIAL_CANDIDATES: '{key[0]}' ({key[1]}) — matched {overlap}, missing {gt_candidates - pred_candidates}")
        elif pred_text_match:
            wrong_type += 1
            type_stats[gt_t]["wrong_type"] += 1
            details.append(f"WRONG_TYPE: '{key[0]}' — GT={key[1]}, pred={pred_text_match[0][1]}")
        else:
            missing += 1
            type_stats[gt_t]["missing"] += 1
            details.append(f"MISSING: '{key[0]}' ({key[1]})")

    # Check extra predictions
    extra = 0
    for key, pred_e in pred_set.items():
        if key not in gt_set:
            gt_text_match = [k for k in gt_set if k[0] == key[0]]
            if not gt_text_match:
                extra += 1
                type_stats[key[1]]["extra"] += 1
                details.append(f"EXTRA: '{key[0]}' ({key[1]})")

    precision = correct / max(len(pred_set), 1)
    recall = correct / max(len(gt_set), 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "correct": correct, "missing": missing, "extra": extra,
        "wrong_type": wrong_type,
        "total_gt": len(gt_set), "total_pred": len(pred_set),
        "precision": precision, "recall": recall, "f1": f1,
        "details": details,
        "by_type": dict(type_stats),
    }


# ── Main test loop ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test GRPO model with tool calling")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="bibocat/qwen3-ner-grpo-final")
    parser.add_argument("--dataset", default="data_train/grpo_combined-100.jsonl")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    # Load KB
    print("Loading KB...")
    kb = KnowledgeBase()
    kb.load()
    print(f"  {len(kb.icd10)} ICD-10, {len(kb.rxnorm_concepts)} RxNorm")

    # Load dataset
    with open(args.dataset) as f:
        samples = [json.loads(line) for line in f][:args.num_samples]
    print(f"Testing {len(samples)} samples...")
    print("=" * 70)

    all_results = []
    all_tool_calls = Counter()
    total_time = 0

    for i, sample in enumerate(samples):
        msgs = sample.get("messages", [])
        text = ""
        for m in msgs:
            if m["role"] == "user":
                text = m["content"]
                break
        gt_entities = sample.get("entities", [])

        if args.verbose:
            print(f"\n[{i+1}/{len(samples)}] Text: {text[:100]}...")
            print(f"  GT entities: {len(gt_entities)}")

        t0 = time.time()
        agent_output = []
        pred_entities = []

        for step in run_agent(args.endpoint, args.model, text, kb):
            agent_output.append(step)

            if step["type"] == "tool_call":
                all_tool_calls[step["name"]] += 1
                if args.verbose:
                    print(f"  🔧 {step['name']}({step['args']}) → {len(step['result'])} results")

            elif step["type"] == "final":
                pred_entities = parse_entities(step["content"])
                if args.verbose:
                    print(f"  📝 Output: {len(pred_entities)} entities predicted")
                if not pred_entities and args.verbose:
                    print(f"  ⚠️  Raw output (unparseable): {step['content'][:200]}")

        elapsed = time.time() - t0
        total_time += elapsed

        # Validate
        result = validate(pred_entities, gt_entities)
        result["sample_id"] = i
        result["time"] = elapsed
        all_results.append(result)

        if args.verbose:
            print(f"  ✅ correct={result['correct']}/{result['total_gt']} missing={result['missing']} extra={result['extra']} wrong_type={result['wrong_type']}")
            print(f"     P={result['precision']:.3f} R={result['recall']:.3f} F1={result['f1']:.3f} ({elapsed:.1f}s)")
            for d in result["details"][:5]:
                print(f"     {d}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_f1 = sum(r["f1"] for r in all_results) / max(len(all_results), 1)
    avg_p = sum(r["precision"] for r in all_results) / max(len(all_results), 1)
    avg_r = sum(r["recall"] for r in all_results) / max(len(all_results), 1)
    total_correct = sum(r["correct"] for r in all_results)
    total_gt = sum(r["total_gt"] for r in all_results)
    total_pred = sum(r["total_pred"] for r in all_results)
    total_missing = sum(r["missing"] for r in all_results)
    total_extra = sum(r["extra"] for r in all_results)
    total_wrong = sum(r["wrong_type"] for r in all_results)

    print(f"\nOverall (micro):")
    print(f"  Correct:  {total_correct}/{total_gt}")
    print(f"  Missing:  {total_missing}")
    print(f"  Extra:    {total_extra}")
    print(f"  WrongType:{total_wrong}")
    print(f"  Precision: {total_correct/max(total_pred,1):.3f}")
    print(f"  Recall:    {total_correct/max(total_gt,1):.3f}")
    print(f"  F1:        {2*total_correct/(total_pred+total_gt):.3f}" if (total_pred+total_gt) > 0 else f"  F1: 0.000")
    print(f"\nAverage (macro):")
    print(f"  Precision: {avg_p:.3f}")
    print(f"  Recall:    {avg_r:.3f}")
    print(f"  F1:        {avg_f1:.3f}")
    print(f"\nAvg time/sample: {total_time/len(all_results):.1f}s")

    # By entity type
    print(f"\nBy entity type:")
    type_agg = defaultdict(lambda: {"correct": 0, "missing": 0, "extra": 0, "wrong_type": 0})
    for r in all_results:
        for t, s in r.get("by_type", {}).items():
            for k in s:
                type_agg[t][k] += s[k]
    for t in sorted(type_agg):
        s = type_agg[t]
        total = s["correct"] + s["missing"]
        acc = s["correct"] / max(total, 1)
        print(f"  {t:20s}: correct={s['correct']:3d}/{total:3d} ({acc:.1%}) missing={s['missing']:2d} extra={s['extra']:2d} wrong_type={s['wrong_type']}")

    print(f"\nTool calls: {dict(all_tool_calls)}")

    # Save results
    out_path = "data_train/test_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "summary": {
                "samples": len(all_results),
                "avg_f1": avg_f1,
                "total_correct": total_correct,
                "total_gt": total_gt,
                "total_pred": total_pred,
                "tool_calls": dict(all_tool_calls),
            },
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
