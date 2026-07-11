"""Per-type extraction agents. Each agent extracts ONE entity type.
All LLM prompts and responses are logged for debugging."""

import sys, re, json
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents import Agent, Tool, AgentContext
from prompt import (
    THUOC_EXTRACT_SYSTEM, THUOC_EXTRACT_USER,
    TRIEUCHUNG_EXTRACT_SYSTEM, TRIEUCHUNG_EXTRACT_USER,
    CHANDOAN_EXTRACT_SYSTEM, CHANDOAN_EXTRACT_USER,
    XETNGHIEM_EXTRACT_SYSTEM, XETNGHIEM_EXTRACT_USER,
)


class ThuocAgent(Agent):
    """Extract THUỐC (medications) only."""
    
    SYSTEM_PROMPT = THUOC_EXTRACT_SYSTEM
    USER_TEMPLATE = THUOC_EXTRACT_USER

    def __init__(self, llm=None):
        tools = [
            Tool("rule_extract_drugs", "Extract drugs via regex", self._rule_drugs),
            Tool("llm_extract_drugs", "Extract drugs via LLM", self._llm_drugs),
        ]
        super().__init__("thuoc_agent", llm, tools)

    def _rule_drugs(self, text: str) -> list:
        from tools import DRUG_PATTERN
        results = []
        for m in DRUG_PATTERN.finditer(text):
            name = m.group("name").strip()
            dose = m.group("dose") or ""
            unit = m.group("unit") or ""
            if len(name) >= 3:
                results.append({
                    "text": f"{name} {dose}{unit}".strip(),
                    "position": [m.start(), m.end()],
                    "type": "THUỐC",
                    "confidence": 0.98,
                    "source": "rule",
                })
        return results

    def _llm_drugs(self, text: str, summary: str) -> list:
        if not self.llm:
            return []
        user_prompt = self.USER_TEMPLATE.format(text=text, summary=summary)
        self.log("llm_call", {
            "input": {"system": self.SYSTEM_PROMPT, "user": user_prompt},
        })
        resp = self.llm.generate(self.SYSTEM_PROMPT, user_prompt, 200)
        self.log("llm_response", {
            "input": {"system": self.SYSTEM_PROMPT, "user": user_prompt},
            "output": resp,
        })
        try:
            return json.loads(resp) if isinstance(json.loads(resp), list) else []
        except:
            return []

    def run(self, ctx: AgentContext, section: dict) -> list:
        self.log("start", {"section": section.get("id"), "type": "THUỐC"})
        meds = []
        meds.extend(self.call_tool("rule_extract_drugs", text=section["text"]))
        if self.llm:
            llm_meds = self.call_tool("llm_extract_drugs",
                text=section["text"], summary=get_section_summary(section.get("text","")))
            seen = {m["text"].lower() for m in meds}
            for m in llm_meds:
                txt = m.get("entity", "").strip().lower()
                if txt and txt not in seen:
                    meds.append({"text": m["entity"], "type": "THUỐC",
                                 "source": "llm", "confidence": 0.85})
                    seen.add(txt)
        self.log("complete", {"found": len(meds)})
        return meds


class TrieuChungAgent(Agent):
    """Extract TRIỆU_CHỨNG (symptoms) only."""
    
    SYSTEM_PROMPT = TRIEUCHUNG_EXTRACT_SYSTEM
    USER_TEMPLATE = TRIEUCHUNG_EXTRACT_USER

    def __init__(self, llm=None):
        tools = [
            Tool("llm_extract_symptoms", "Extract symptoms via LLM", self._llm_symptoms),
        ]
        super().__init__("trieuchung_agent", llm, tools)

    def _llm_symptoms(self, text: str, summary: str) -> list:
        if not self.llm:
            return []
        user_prompt = self.USER_TEMPLATE.format(text=text, summary=summary)
        self.log("llm_call", {
            "input": {"system": self.SYSTEM_PROMPT, "user": user_prompt},
        })
        resp = self.llm.generate(self.SYSTEM_PROMPT, user_prompt, 200)
        self.log("llm_response", {
            "input": {"system": self.SYSTEM_PROMPT, "user": user_prompt},
            "output": resp,
        })
        try:
            return json.loads(resp) if isinstance(json.loads(resp), list) else []
        except:
            return []

    def run(self, ctx: AgentContext, section: dict) -> list:
        self.log("start", {"section": section.get("id"), "type": "TRIỆU_CHỨNG"})
        if not self.llm:
            return []
        return self.call_tool("llm_extract_symptoms",
            text=section["text"], summary=get_section_summary(section.get("text","")))


class ChanDoanAgent(Agent):
    """Extract CHẨN_ĐOÁN (diagnoses) only."""
    
    SYSTEM_PROMPT = CHANDOAN_EXTRACT_SYSTEM
    USER_TEMPLATE = CHANDOAN_EXTRACT_USER

    def __init__(self, llm=None):
        tools = [
            Tool("llm_extract_diagnoses", "Extract diagnoses via LLM", self._llm_diag),
        ]
        super().__init__("chanDoan_agent", llm, tools)

    def _llm_diag(self, text: str, summary: str) -> list:
        if not self.llm:
            return []
        user_prompt = self.USER_TEMPLATE.format(text=text, summary=summary)
        self.log("llm_call", {
            "input": {"system": self.SYSTEM_PROMPT, "user": user_prompt},
        })
        resp = self.llm.generate(self.SYSTEM_PROMPT, user_prompt, 200)
        self.log("llm_response", {
            "input": {"system": self.SYSTEM_PROMPT, "user": user_prompt},
            "output": resp,
        })
        try:
            return json.loads(resp) if isinstance(json.loads(resp), list) else []
        except:
            return []

    def run(self, ctx: AgentContext, section: dict) -> list:
        self.log("start", {"section": section.get("id"), "type": "CHẨN_ĐOÁN"})
        if not self.llm:
            return []
        return self.call_tool("llm_extract_diagnoses",
            text=section["text"], summary=get_section_summary(section.get("text","")))


class XetNghiemAgent(Agent):
    """Extract TÊN_XÉT_NGHIỆM (test names) + KẾT_QUẢ_XÉT_NGHIỆM (results)."""
    
    SYSTEM_PROMPT = XETNGHIEM_EXTRACT_SYSTEM
    USER_TEMPLATE = XETNGHIEM_EXTRACT_USER

    def __init__(self, llm=None):
        tools = [
            Tool("rule_extract_tests", "Extract tests via regex", self._rule_tests),
            Tool("rule_extract_results", "Extract results via regex", self._rule_results),
            Tool("llm_extract_tests", "Extract tests+results via LLM", self._llm_tests),
        ]
        super().__init__("xetnghiem_agent", llm, tools)

    def _rule_tests(self, text: str) -> list:
        from tools import TEST_PATTERNS
        results = []
        for pattern in TEST_PATTERNS:
            for m in re.finditer(re.escape(pattern), text, re.IGNORECASE):
                if m.end() - m.start() >= 2:
                    results.append({
                        "text": text[m.start():m.end()],
                        "position": [m.start(), m.end()],
                        "type": "TÊN_XÉT_NGHIỆM", "confidence": 0.95, "source": "rule",
                    })
        return results

    def _rule_results(self, text: str) -> list:
        from tools import NORMAL_RESULTS, LAB_VALUE_PATTERNS
        results = []
        for pattern in NORMAL_RESULTS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                results.append({
                    "text": text[m.start():m.end()],
                    "position": [m.start(), m.end()],
                    "type": "KẾT_QUẢ_XÉT_NGHIỆM", "confidence": 0.90, "source": "rule",
                })
        for pattern, _ in LAB_VALUE_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                if m.lastindex and m.lastindex >= 2:
                    results.append({
                        "text": m.group(m.lastindex),
                        "position": [m.start(m.lastindex), m.end(m.lastindex)],
                        "type": "KẾT_QUẢ_XÉT_NGHIỆM", "confidence": 0.95, "source": "rule",
                    })
        return results

    def _llm_tests(self, text: str, summary: str) -> list:
        if not self.llm:
            return []
        user_prompt = self.USER_TEMPLATE.format(text=text, summary=summary)
        self.log("llm_call", {
            "input": {"system": self.SYSTEM_PROMPT, "user": user_prompt},
        })
        resp = self.llm.generate(self.SYSTEM_PROMPT, user_prompt, 250)
        self.log("llm_response", {
            "input": {"system": self.SYSTEM_PROMPT, "user": user_prompt},
            "output": resp,
        })
        try:
            return json.loads(resp) if isinstance(json.loads(resp), list) else []
        except:
            return []

    def run(self, ctx: AgentContext, section: dict) -> list:
        self.log("start", {"section": section.get("id"), "type": "XÉT_NGHIỆM"})
        res = []
        res.extend(self.call_tool("rule_extract_tests", text=section["text"]))
        res.extend(self.call_tool("rule_extract_results", text=section["text"]))
        if self.llm:
            llm_res = self.call_tool("llm_extract_tests",
                text=section["text"], summary=get_section_summary(section.get("text","")))
            seen = {f"{m['text']}_{m['type']}" for m in res}
            for m in llm_res:
                key = f"{m.get('entity','')}_{m.get('type','')}"
                if key not in seen:
                    res.append({"text": m["entity"], "type": m["type"],
                                "source": "llm", "confidence": 0.85})
                    seen.add(key)
        self.log("complete", {"found": len(res)})
        return res


def get_section_summary(text: str) -> str:
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    summary = ' | '.join(lines[:3]) if lines else text[:150]
    return summary[:200] + '...' if len(summary) > 200 else summary
