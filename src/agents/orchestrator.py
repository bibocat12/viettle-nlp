import json
import sys
import time
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from agents import Agent, Tool, AgentContext
from worker_agents import ExtractionAgent, AssertionAgent, LinkingAgent, VerifierAgent


class OrchestratorAgent(Agent):
    def __init__(self, llm=None, kb=None, on_log: Optional[Callable] = None, mode="voting"):
        self.kb = kb
        self.on_log = on_log
        self.mode = mode

        self.extraction = ExtractionAgent(llm=llm, mode=mode)
        self.assertion = AssertionAgent(llm=llm)
        self.linking = LinkingAgent(llm=llm, kb=kb)
        self.verifier = VerifierAgent(llm=llm)

        tools = [
            Tool("run_extraction", "Parallel extraction + voting + position fix", self._run_extraction),
            Tool("run_assertion", "Assertion classification", self._run_assertion),
            Tool("run_linking", "ICD-10 / RxNorm linking", self._run_linking),
            Tool("run_verification", "Schema validation and fix", self._run_verification),
        ]
        super().__init__("orchestrator", llm, tools)

    def _emit_log(self, entry: dict):
        self.trace.append(entry)
        if self.on_log:
            self.on_log(entry)

    def _flush_agent_logs(self, agent):
        for entry in agent.trace:
            if self.on_log:
                self.on_log(entry)
        agent.trace.clear()

    def _run_extraction(self, ctx: AgentContext) -> AgentContext:
        self._emit_log({"agent": "orchestrator", "action": "extraction_start", "ts": time.time()})
        ctx = self.extraction.run(ctx)
        self._flush_agent_logs(self.extraction)
        self._emit_log({"agent": "orchestrator", "action": "extraction_complete",
                        "mentions": len(ctx.mentions), "ts": time.time()})
        return ctx

    def _run_assertion(self, ctx: AgentContext) -> AgentContext:
        self._emit_log({"agent": "orchestrator", "action": "assertion_start", "ts": time.time()})
        ctx = self.assertion.run(ctx)
        self._flush_agent_logs(self.assertion)
        self._emit_log({"agent": "orchestrator", "action": "assertion_complete", "ts": time.time()})
        return ctx

    def _run_linking(self, ctx: AgentContext) -> AgentContext:
        self._emit_log({"agent": "orchestrator", "action": "linking_start", "ts": time.time()})
        ctx = self.linking.run(ctx)
        self._flush_agent_logs(self.linking)
        linked = len([m for m in ctx.mentions if m.get("candidates")])
        self._emit_log({"agent": "orchestrator", "action": "linking_complete",
                        "linked": linked, "ts": time.time()})
        return ctx

    def _run_verification(self, ctx: AgentContext) -> AgentContext:
        self._emit_log({"agent": "orchestrator", "action": "verification_start", "ts": time.time()})
        ctx = self.verifier.run(ctx)
        self._flush_agent_logs(self.verifier)
        self._emit_log({"agent": "orchestrator", "action": "verification_complete",
                        "kept": len(ctx.mentions), "ts": time.time()})
        return ctx

    def run(self, ctx: AgentContext) -> AgentContext:
        t0 = time.time()
        self._emit_log({"agent": "orchestrator", "action": "start",
                        "text_len": len(ctx.text), "ts": t0})

        ctx = self.call_tool("run_extraction", ctx=ctx)
        ctx = self.call_tool("run_assertion", ctx=ctx)
        ctx = self.call_tool("run_linking", ctx=ctx)
        ctx = self.call_tool("run_verification", ctx=ctx)

        elapsed = time.time() - t0
        self._emit_log({"agent": "orchestrator", "action": "complete",
                        "total_mentions": len(ctx.mentions),
                        "elapsed": round(elapsed, 2), "ts": time.time()})
        return ctx


def create_orchestrator(kb=None, llm=None, on_log=None, mode="voting") -> OrchestratorAgent:
    return OrchestratorAgent(llm=llm, kb=kb, on_log=on_log, mode=mode)
