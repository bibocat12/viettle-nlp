import json
import sys
import time
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from agents import Agent, Tool, AgentContext, LLMEngine
from worker_agents import ExtractionAgent, AssertionAgent, LinkingAgent, VerifierAgent


class OrchestratorAgent(Agent):
    def __init__(self, llm=None, kb=None, on_log: Optional[Callable] = None):
        self.kb = kb
        self.on_log = on_log

        self.extraction = ExtractionAgent(llm=llm)
        self.assertion = AssertionAgent(llm=llm)
        self.linking = LinkingAgent(llm=llm, kb=kb)
        self.verifier = VerifierAgent(llm=llm)

        tools = [
            Tool("extract", "Run extraction agent", self._run_extraction),
            Tool("assert", "Run assertion agent", self._run_assertion),
            Tool("link", "Run linking agent", self._run_linking),
            Tool("verify", "Run verifier agent", self._run_verify),
            Tool("plan", "LLM decides next step", self._plan),
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
        self._emit_log({"agent": "orchestrator", "action": "call_tool", "tool": "extract", "ts": time.time()})
        ctx = self.extraction.run(ctx)
        self._flush_agent_logs(self.extraction)
        self._emit_log({"agent": "extraction", "action": "complete", "mentions": len(ctx.mentions), "ts": time.time()})
        return ctx

    def _run_assertion(self, ctx: AgentContext) -> AgentContext:
        self._emit_log({"agent": "orchestrator", "action": "call_tool", "tool": "assert", "ts": time.time()})
        ctx = self.assertion.run(ctx)
        self._flush_agent_logs(self.assertion)
        self._emit_log({"agent": "assertion", "action": "complete", "ts": time.time()})
        return ctx

    def _run_linking(self, ctx: AgentContext) -> AgentContext:
        self._emit_log({"agent": "orchestrator", "action": "call_tool", "tool": "link", "ts": time.time()})
        ctx = self.linking.run(ctx)
        self._flush_agent_logs(self.linking)
        linked = len([m for m in ctx.mentions if m.get("candidates")])
        self._emit_log({"agent": "linking", "action": "complete", "linked": linked, "ts": time.time()})
        return ctx

    def _run_verify(self, ctx: AgentContext) -> AgentContext:
        self._emit_log({"agent": "orchestrator", "action": "call_tool", "tool": "verify", "ts": time.time()})
        valid_types = {"THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}
        issues = []
        for i, m in enumerate(ctx.mentions):
            if m.get("type") not in valid_types:
                issues.append(i)
        self._emit_log({"agent": "verifier", "action": "validated", "total": len(ctx.mentions), "issues": len(issues), "ts": time.time()})
        return ctx

    def _plan(self, ctx: AgentContext) -> dict:
        if self.llm is None:
            return {"steps": ["extract", "assert", "link", "verify"]}
        uncertain = ctx.get_uncertain_mentions(0.7)
        needs_rerank = len([m for m in ctx.mentions if m.get("candidates") and len(m.get("candidates", [])) > 2])
        plan = ["extract", "assert", "link"]
        if uncertain:
            plan.append("review_uncertain")
        if needs_rerank:
            plan.append("rerank")
        plan.append("verify")
        return {"steps": plan, "uncertain": len(uncertain), "needs_rerank": needs_rerank}

    def run(self, ctx: AgentContext) -> AgentContext:
        self._emit_log({"agent": "orchestrator", "action": "start", "text_len": len(ctx.text), "ts": time.time()})

        plan = self.call_tool("plan", ctx=ctx)
        self._emit_log({"agent": "orchestrator", "action": "plan", "steps": plan.get("steps", []), "ts": time.time()})

        for step in plan.get("steps", []):
            if step == "extract":
                ctx = self.call_tool("extract", ctx=ctx)
            elif step == "assert":
                ctx = self.call_tool("assert", ctx=ctx)
            elif step == "link":
                ctx = self.call_tool("link", ctx=ctx)
            elif step == "verify":
                ctx = self.call_tool("verify", ctx=ctx)

        self._emit_log({"agent": "orchestrator", "action": "complete", "total_mentions": len(ctx.mentions), "ts": time.time()})
        return ctx


def create_orchestrator(kb=None, llm=None, on_log=None) -> OrchestratorAgent:
    return OrchestratorAgent(llm=llm, kb=kb, on_log=on_log)
