#!/usr/bin/env python3
"""Remote agent server for Fireworks RFT — runs the model with tool calling.

The server receives rollouts from Fireworks, calls the model via Fireworks
tracing (tool_calls for search_icd10/search_rxnorm), executes tools against
the local KB, and returns the final response.

Usage:
    export FIREWORKS_API_KEY=fw_xxx
    python3 scripts/remote_agent_server.py --port 8080
    # Expose publicly: ngrok http 8080

Then launch RFT with:
    eval-protocol create rft \
      --remote-server-url https://your-ngrok-url.ngrok.io \
      ...
"""

import json
import logging
import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kb import KnowledgeBase

load_dotenv(Path(__file__).parent.parent / ".env")

# Fireworks tracing
from eval_protocol import FireworksTracingHttpHandler, InitRequest, RolloutIdFilter, Status

logging.getLogger().addHandler(FireworksTracingHttpHandler())

app = FastAPI(title="NER Remote Agent")

# Global KB (loaded once)
kb = None


TOOLS_DEFINITION = [
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


@app.on_event("startup")
def load_kb():
    global kb
    kb = KnowledgeBase()
    kb.load()
    print(f"KB loaded: {len(kb.icd10)} ICD-10, {len(kb.rxnorm_concepts)} RxNorm", flush=True)


def execute_tool(tool_name: str, args: dict) -> list:
    """Execute a tool call against the local KB."""
    results = []
    if tool_name == "search_icd10":
        for r in kb.search_icd10(args.get("query", ""), top_k=8):
            results.append({"code": r["code"], "name": r["name"], "score": r["score"]})
    elif tool_name == "search_rxnorm":
        for r in kb.search_rxnorm(args.get("query", ""), top_k=8):
            results.append({"rxcui": r["rxcui"], "name": r["name"], "score": r["score"]})
    return results


import httpx


def _call_llm(base_url: str, api_key: str, model: str, messages: list,
              tools: list | None = None, temperature: float = 0.7,
              max_tokens: int = 1024) -> dict:
    """Call an OpenAI-compatible LLM endpoint directly via HTTP.

    Uses httpx instead of the OpenAI SDK to avoid LiteLLM interception.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        body["tools"] = tools

    url = base_url.rstrip("/") + "/chat/completions"
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()


def _build_client(model_config: dict, api_key: str):
    """No-op — we use _call_llm directly instead."""
    pass


def run_agent(messages: list, model_config: dict, api_key: str) -> str:
    """Run the model with tool-calling loop.

    - Calls the model through Fireworks traced endpoint (model_base_url)
    - Executes tool calls against the local KB
    - Loops until the model produces a final text response
    """
    base_url = model_config.get("model_base_url", "")
    model_name = model_config.get("model", "qwen-turbo")
    temperature = model_config.get("temperature", 0.7)
    max_tokens = model_config.get("max_tokens", 1024)

    # If no Fireworks URL, fall back to Qwen API
    if "fireworks.ai" not in base_url:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")
        base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        api_key = os.getenv("QWEN_API") or os.getenv("DASHSCOPE_API_KEY")
        model_name = "qwen-turbo"

    system_prompt = f"""Bạn là hệ thống trích xuất thực thể y khoa tiếng Việt.
Bạn có các công cụ sau để tra cứu mã chuẩn:
{json.dumps([t["function"] for t in TOOLS_DEFINITION], ensure_ascii=False, indent=2)}

Quy trình:
1. Đọc văn bản lâm sàng.
2. Gọi search_icd10 cho mỗi bệnh/chẩn đoán tìm được.
3. Gọi search_rxnorm cho mỗi thuốc tìm được.
4. Tổng hợp kết quả thành JSON array entities.

Định dạng tool call:
  {{"name": "search_icd10", "arguments": {{{{"query": "tên bệnh"}}}}}}

Định dạng cuối cùng:
  [{{"entity": "tên", "type": "LOẠI", "candidates": ["MÃ"]}}]
"""

    # Convert Message objects to plain dicts
    plain_messages = []
    for m in messages:
        if isinstance(m, dict):
            plain_messages.append(m)
        elif hasattr(m, "role") and hasattr(m, "content"):
            plain_messages.append({"role": m.role, "content": m.content or ""})
        else:
            plain_messages.append({"role": "user", "content": str(m)})
    conversation = [{"role": "system", "content": system_prompt}] + plain_messages
    max_turns = 6

    for turn in range(max_turns):
        response = _call_llm(
            base_url=base_url, api_key=api_key, model=model_name,
            messages=conversation, tools=TOOLS_DEFINITION,
            temperature=temperature, max_tokens=max_tokens,
        )

        choice = response.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        if not tool_calls:
            return content or ""

        conversation.append(msg)
        for tc in tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
            try:
                fn_args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                fn_args = {}
            result = execute_tool(fn_name, fn_args)
            conversation.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "name": fn_name,
                "content": json.dumps(result, ensure_ascii=False),
            })

    return "FAIL_LINKING"


@app.post("/init")
async def init(request: InitRequest):
    rollout_logger = logging.getLogger(f"ner_remote.{request.metadata.rollout_id}")
    rollout_logger.addFilter(RolloutIdFilter(request.metadata.rollout_id))

    rollout_logger.info(f"Starting NER rollout: {request.metadata.rollout_id}")

    try:
        result = run_agent(
            messages=request.messages,
            model_config={
                "model": request.completion_params.get("model"),
                "temperature": request.completion_params.get("temperature", 0.7),
                "max_tokens": request.completion_params.get("max_tokens", 1024),
                "model_base_url": request.model_base_url,
            },
            api_key=request.api_key,
        )

        rollout_logger.info(
            f"Rollout {request.metadata.rollout_id} completed",
            extra={"status": Status.rollout_finished()},
        )

        return {"status": "success", "result": result}

    except Exception as e:
        rollout_logger.error(
            f"Rollout {request.metadata.rollout_id} failed: {e}",
            extra={"status": Status.rollout_error(str(e))},
        )
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    if not os.getenv("FIREWORKS_API_KEY"):
        print("ERROR: FIREWORKS_API_KEY not set")
        sys.exit(1)

    print(f"Starting NER remote agent on {args.host}:{args.port}")
    print("Expose publicly with: ngrok http " + str(args.port))
    uvicorn.run(app, host=args.host, port=args.port)
