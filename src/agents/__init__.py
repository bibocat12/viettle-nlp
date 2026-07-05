import json
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm import LLMEngine


class Tool:
    def __init__(self, name: str, description: str, func, schema: Optional[dict] = None):
        self.name = name
        self.description = description
        self.func = func
        self.schema = schema or {}

    def run(self, **kwargs) -> Any:
        return self.func(**kwargs)


class Agent(ABC):
    def __init__(self, name: str, llm=None, tools=None):
        self.name = name
        self.llm = llm
        self.tools = {t.name: t for t in (tools or [])}
        self.trace: List[Dict] = []

    def add_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def log(self, action: str, data: dict):
        self.trace.append({"agent": self.name, "action": action, "data": data})

    def get_tools_description(self) -> str:
        lines = []
        for t in self.tools.values():
            lines.append(f"- {t.name}: {t.description}")
        return "\n".join(lines)

    def call_tool(self, tool_name: str, **kwargs) -> Any:
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found. Available: {list(self.tools.keys())}")
        result = self.tools[tool_name].run(**kwargs)
        self.log("tool_call", {"tool": tool_name, "args": kwargs, "result_type": type(result).__name__})
        return result

    def llm_decide(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> dict:
        if self.llm is None:
            return {}
        response = self.llm.generate(system_prompt, user_prompt, max_tokens)
        return self.llm._parse_json(response)

    @abstractmethod
    def run(self, context):
        pass


class AgentContext:
    def __init__(self, text: str, kb=None):
        self.text = text
        self.normalized = text
        self.kb = kb
        self.mentions: List[Dict] = []
        self.rules: List[Dict] = []
        self.ner: List[Dict] = []
        self.assertions: Dict[str, Dict] = {}
        self.candidates: Dict[str, List] = {}
        self.conflicts: List[Dict] = []
        self.trace: List[Dict] = []
        self.errors: List[str] = []

    def add_mention(self, m: dict):
        self.mentions.append(m)

    def get_mentions_by_type(self, mtype: str) -> List[Dict]:
        return [m for m in self.mentions if m.get("type") == mtype]

    def get_uncertain_mentions(self, threshold: float = 0.7) -> List[Dict]:
        return [m for m in self.mentions if m.get("confidence", 1.0) < threshold]

    def get_mentions_with_candidates(self) -> List[Dict]:
        return [m for m in self.mentions if m.get("candidates")]

    def to_output(self) -> List[Dict]:
        valid_types = {"THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}
        result = []
        for m in self.mentions:
            if m.get("type") not in valid_types:
                continue
            pos = m.get("position", [0, 0])
            if not isinstance(pos, list) or len(pos) != 2:
                continue
            if pos[1] <= pos[0]:
                continue
            out = {
                "text": m.get("text", ""),
                "position": pos,
                "type": m["type"],
                "assertions": {
                    "isNegated": bool(m.get("assertions", {}).get("isNegated", False)),
                    "isFamily": bool(m.get("assertions", {}).get("isFamily", False)),
                    "isHistorical": bool(m.get("assertions", {}).get("isHistorical", False)),
                },
            }
            if m["type"] in ("CHẨN_ĐOÁN", "THUỐC"):
                out["candidates"] = m.get("candidates", [])
            result.append(out)
        return result
