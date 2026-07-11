"""New orchestrator: parallel sections + per-type agents + summarizer."""

import sys, json, time, os
from pathlib import Path
from typing import Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from agents import AgentContext
from agents.section_parser import parse_sections, get_section_summary
from agents.type_agents import ThuocAgent, TrieuChungAgent, ChanDoanAgent, XetNghiemAgent
from agents.summarizer_agent import SummarizerAgent
from tools import rule_extract, validate_output
from position_fix import fix_positions


# ─── Parallel section runner using tmux ────────────────────────────

def run_section_in_tmux(section: dict, llm, output_file: str):
    """Run extraction on a section in a tmux session. Returns when done."""
    import subprocess
    session_name = f"sec_{section['id']}_{int(time.time())}"
    
    cmd = f"""cd {Path(__file__).parent.parent} && source .venv/bin/activate && python3 -c "
import sys, json
sys.path.insert(0, 'src')
from agents.type_agents import ThuocAgent, TrieuChungAgent, ChanDoanAgent, XetNghiemAgent
from agents.section_parser import get_section_summary

text = {json.dumps(section['text'])}
summary = get_section_summary(text)

results = []
ta = ThuocAgent(llm=None)
results.extend(ta.run(None, section))
xa = XetNghiemAgent(llm=None)
results.extend(xa.run(None, section))

with open('{output_file}', 'w') as f:
    json.dump({{'section_id': '{section['id']}', 'mentions': results}}, f, ensure_ascii=False)
print('Done: {section['id']}')
" 2>&1 | tee /tmp/section_{section['id']}.log"""
    
    subprocess.run(["tmux", "new-session", "-d", "-s", session_name, cmd],
                   capture_output=True, timeout=5)
    return session_name


def wait_tmux_sessions(sessions: list, poll_interval: float = 1.0, timeout: float = 120):
    """Wait for tmux sessions to complete."""
    import subprocess
    import time
    start = time.time()
    remaining = set(sessions)
    while remaining and (time.time() - start) < timeout:
        for s in list(remaining):
            result = subprocess.run(
                ["tmux", "has-session", "-t", s],
                capture_output=True, timeout=2
            )
            if result.returncode != 0:
                remaining.remove(s)
        if remaining:
            time.sleep(poll_interval)
    # Kill any remaining
    for s in remaining:
        subprocess.run(["tmux", "kill-session", "-t", s], capture_output=True, timeout=2)


# ─── Main parallel pipeline ────────────────────────────────────────

def run_parallel_pipeline(
    note_id: str,
    text: str,
    kb=None,
    llm=None,
    on_log: Optional[Callable] = None,
    use_tmux: bool = False,
) -> dict:
    """Run pipeline: sectionize → parallel per-section extraction → merge."""
    t0 = time.time()
    logs = []

    def log(agent, action, **kwargs):
        entry = {"agent": agent, "action": action, "ts": time.time(), **kwargs}
        logs.append(entry)
        if on_log:
            on_log(entry)

    log("system", "start")
    
    # Step 1: Parse sections
    sections = parse_sections(text)
    log("sectionizer", "parsed", sections=[s["id"] for s in sections],
        summaries=[get_section_summary(s["text"]) for s in sections])

    # Step 2: Run summarizer (fast, no LLM needed for basic)
    summarizer = SummarizerAgent(llm=llm)
    ctx = AgentContext(text, kb=kb)
    section_summaries = summarizer.run(ctx, sections)
    log("summarizer", "complete", summaries=section_summaries)

    # Step 3: Parallel extraction per section
    all_mentions = []
    
    if use_tmux and len(sections) > 1:
        # Run each section in background tmux sessions
        tmp_dir = Path("/tmp/section_extractions")
        tmp_dir.mkdir(exist_ok=True)
        sessions = []
        for i, sec in enumerate(sections):
            out_file = str(tmp_dir / f"{note_id}_sec_{sec['id']}.json")
            sess = run_section_in_tmux(sec, llm, out_file)
            sessions.append(sess)
            log("parallel", "section_started", section=sec["id"], tmux_session=sess)
        
        wait_tmux_sessions(sessions, timeout=300)
        log("parallel", "all_sections_done")
        
        # Collect results
        for sec in sections:
            out_file = tmp_dir / f"{note_id}_sec_{sec['id']}.json"
            if out_file.exists():
                data = json.loads(out_file.read_text())
                for m in data.get("mentions", []):
                    m["section_id"] = sec["id"]
                    all_mentions.append(m)
    else:
        # Sequential (no tmux) or single section
        with ThreadPoolExecutor(max_workers=min(4, len(sections))) as ex:
            future_map = {}
            for sec in sections:
                summary = get_section_summary(sec["text"])
                def extract_section(sec=sec):
                    sec_mentions = []
                    for agent_cls in [ThuocAgent, XetNghiemAgent]:
                        agent = agent_cls(llm=llm)
                        sec_mentions.extend(agent.run(ctx, sec))
                    return sec["id"], sec_mentions
                future = ex.submit(extract_section)
                future_map[future] = sec["id"]
            
            for future in as_completed(future_map):
                sec_id, sec_mentions = future.result()
                for m in sec_mentions:
                    m["section_id"] = sec_id
                    all_mentions.extend(sec_mentions)
    
    log("extraction", "complete", total=len(all_mentions), sections={s["id"]: s["title"] for s in sections})

    # Step 4: Position fix
    from position_fix import fix_positions
    all_mentions = fix_positions(all_mentions, text)
    log("position_fix", "done", count=len(all_mentions))

    # Step 5: Dedup by position
    seen = {}
    for m in all_mentions:
        key = tuple(m.get("position", [0, 0]))
        if key not in seen or m.get("confidence", 0) > seen[key].get("confidence", 0):
            seen[key] = m
    final_mentions = list(seen.values())
    log("dedup", "done", before=len(all_mentions), after=len(final_mentions))

    # Step 6: Output
    output = []
    for m in final_mentions:
        out = {
            "text": m.get("text", ""),
            "position": m.get("position", [0, 0]),
            "type": m.get("type", ""),
            "confidence": m.get("confidence", 0.5),
            "section_id": m.get("section_id", ""),
            "assertions": {"isNegated": False, "isFamily": False, "isHistorical": False},
        }
        output.append(out)

    validation = validate_output(output)
    log("system", "complete", elapsed=round(time.time()-t0, 2))
    
    return {
        "note_id": note_id,
        "mentions": output,
        "validation": validation,
        "sections": [{"id": s["id"], "title": s["title"]} for s in sections],
        "summaries": section_summaries,
        "stats": {
            "sections": len(sections),
            "total_mentions": len(all_mentions),
            "final": len(output),
            "elapsed": round(time.time() - t0, 2),
        },
        "logs": logs,
    }
