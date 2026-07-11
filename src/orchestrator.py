"""Pipeline entry point — delegates to agentic orchestrator. Saves all logs to agent_logs/."""

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

LOG_DIR = Path(__file__).parent.parent / "agent_logs"


def _save_log(note_id: str, data: dict):
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"{note_id}.json"
    # Pretty-print with full content (no truncation)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    # Also save human-readable version
    readable_path = LOG_DIR / f"{note_id}_readable.json"
    readable = {
        "file_id": data["file_id"],
        "timestamp": data["timestamp"],
        "elapsed": data["elapsed"],
        "model": data["model"],
        "mention_count": len(data.get("result", {}).get("mentions", [])),
        "stats": data.get("result", {}).get("stats", {}),
        "log_summary": [
            {
                "agent": e.get("agent"),
                "action": e.get("action"),
                "ts": e.get("ts"),
                "tool": e.get("data", {}).get("tool") if isinstance(e.get("data"), dict) else None,
            }
            for e in data.get("logs", [])
            if e.get("action") in ("tool_call", "extracted", "voted", "validated", "complete", "error")
        ],
        "mentions": data.get("result", {}).get("mentions", []),
    }
    readable_path.write_text(json.dumps(readable, ensure_ascii=False, indent=2))


def run_pipeline(
    note_id: str,
    text: str,
    kb=None,
    llm=None,
    on_log: Optional[Callable] = None,
    mode: str = "voting",
) -> dict:
    t0 = time.time()
    logs = []

    def log_capture(entry: dict):
        logs.append(entry)
        if on_log:
            on_log(entry)

    ctx = AgentContext(text, kb=kb)
    agent = create_orchestrator(kb=kb, llm=llm, on_log=log_capture, mode=mode)
    ctx = agent.run(ctx)

    output_mentions = ctx.to_output()
    validation = validate_output(output_mentions)

    # Include ALL logs (full I/O) + final output in the result
    stats = {
        "total_mentions": len(ctx.mentions),
        "final": len(output_mentions),
        "validation_errors": len(validation["errors"]),
        "elapsed": round(time.time() - t0, 2),
    }

    extracted_sources = {}
    for entry in logs:
        if entry.get("action") == "extracted":
            sources = entry.get("data", {}).get("sources", {}) if isinstance(entry.get("data"), dict) else {}
            for k, v in sources.items():
                if isinstance(v, (list, int)):
                    extracted_sources[k] = len(v) if isinstance(v, list) else v

    result = {
        "note_id": note_id,
        "mentions": output_mentions,
        "validation": validation,
        "stats": {**stats, **extracted_sources},
        "logs": logs,
    }

    # Save full log to agent_logs/
    model_name = "runpod" if llm and getattr(llm, 'is_available', lambda: False)() else "rules"
    log_data = {
        "file_id": note_id,
        "timestamp": time.time(),
        "elapsed": round(time.time() - t0, 2),
        "model": model_name,
        "result": result,
        "logs": logs,
    }
    _save_log(note_id, log_data)

    return result
