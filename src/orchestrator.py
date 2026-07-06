"""Pipeline entry point — delegates to agentic orchestrator."""

import json
import sys
import time
from typing import Callable, Dict, List, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "agents"))

from agents import AgentContext
from agents.orchestrator import create_orchestrator
from tools import validate_output


def run_pipeline(
    note_id: str,
    text: str,
    kb=None,
    llm=None,
    on_log: Optional[Callable] = None,
) -> dict:
    t0 = time.time()
    logs = []

    def log_capture(entry: dict):
        logs.append(entry)
        if on_log:
            on_log(entry)

    ctx = AgentContext(text, kb=kb)
    agent = create_orchestrator(kb=kb, llm=llm, on_log=log_capture)
    ctx = agent.run(ctx)

    output_mentions = ctx.to_output()
    validation = validate_output(output_mentions)

    stats = {
        "total_mentions": len(ctx.mentions),
        "final": len(output_mentions),
        "validation_errors": len(validation["errors"]),
        "elapsed": round(time.time() - t0, 2),
    }

    extracted_sources = {}
    for entry in logs:
        if entry.get("action") == "extracted":
            sources = entry.get("sources", {})
            for k, v in sources.items():
                extracted_sources[k] = len(v)

    return {
        "note_id": note_id,
        "mentions": output_mentions,
        "validation": validation,
        "stats": {**stats, **extracted_sources},
        "logs": logs,
    }
