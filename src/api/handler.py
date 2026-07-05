import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import runpod


def handler(job):
    input_data = job.get("input", {})
    text = input_data.get("text", "")
    use_llm = input_data.get("use_llm", False)

    if not text:
        return {"error": "No text provided"}

    try:
        from kb import KnowledgeBase
        from agents.orchestrator import create_orchestrator
        from agents import AgentContext

        base_dir = str(Path(__file__).parent.parent / "database-stuff")
        kb = KnowledgeBase(base_dir=base_dir)
        kb.load()

        logs = []

        def on_log(entry):
            logs.append(entry)
            print(json.dumps(entry), flush=True)

        llm = None
        if use_llm:
            from llm import LLMEngine
            llm = LLMEngine()
            llm.load()

        orchestrator = create_orchestrator(kb=kb, llm=llm, on_log=on_log)
        ctx = AgentContext(text=text, kb=kb)

        ctx = orchestrator.run(ctx)
        result = ctx.to_output()

        return {
            "mentions": result,
            "count": len(result),
            "logs": logs,
            "trace": orchestrator.trace,
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


runpod.serverless.start({"handler": handler})
