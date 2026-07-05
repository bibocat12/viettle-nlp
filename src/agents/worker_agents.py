from typing import Any, Dict, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from agents import Agent, Tool, AgentContext


class ExtractionAgent(Agent):
    def __init__(self, llm=None):
        tools = [
            Tool("rule_extract", "Extract mentions using regex rules", self._rule_extract),
            Tool("classify_type", "Classify mention type using LLM", self._classify_type),
        ]
        super().__init__("extraction", llm, tools)

    def _rule_extract(self, text: str) -> list:
        import sys
        sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
        from rules import extract_rules
        return extract_rules(text)

    def _classify_type(self, mention: str, context: str) -> dict:
        if self.llm is None:
            return {"type": "TRIỆU_CHỨNG"}
        system = """Classify this Vietnamese medical mention into one of:
- THUỐC: Medications
- TRIỆU_CHỨNG: Symptoms
- CHẨN_ĐOÁN: Diagnoses
- TÊN_XÉT_NGHIỆM: Test names
- KẾT_QUẢ_XÉT_NGHIỆM: Test results

Return JSON: {"type": "TYPE", "confidence": 0.0-1.0}"""
        user = f"Mention: \"{mention}\"\nContext: \"{context[:300]}\""
        return self.llm_decide(system, user, 100)

    def run(self, ctx: AgentContext) -> AgentContext:
        self.log("start", {"text_len": len(ctx.text)})

        rules = self.call_tool("rule_extract", text=ctx.normalized)
        ctx.rules = rules
        ctx.mentions = rules

        self.log("complete", {"rule_mentions": len(rules)})
        return ctx


class AssertionAgent(Agent):
    NEGATION = [
        r"không\b", r"chưa\b", r"không có\b", r"chưa có\b", r"không ghi nhận\b",
        r"không có gì\b", r"bình thường\b", r"không có bất thường\b",
        r"không có gì đáng\b", r"không buồn nôn\b", r"không đánh trống ngực\b",
    ]
    FAMILY = [r"bố\b", r"mẹ\b", r"cha\b", r"người nhà\b", r"gia đình\b", r"theo lời\b"]
    HISTORICAL = [r"tiền sử\b", r"trước đây\b", r"trước khi\b", r"đã điều trị\b", r"đã dùng\b"]

    def __init__(self, llm=None):
        tools = [
            Tool("check_assertion", "Check assertions using rules", self._check_rules),
            Tool("llm_assertion", "LLM-based assertion for hard cases", self._check_llm),
        ]
        super().__init__("assertion", llm, tools)

    def _check_rules(self, text: str, start: int, end: int, window: int = 40) -> dict:
        import re
        ctx_start = max(0, start - window)
        ctx_end = min(len(text), end + window)
        prefix = text[ctx_start:start].lower()
        is_neg = any(re.search(p, prefix) for p in self.NEGATION)
        is_fam = any(re.search(p, prefix) for p in self.FAMILY)
        is_hist = any(re.search(p, prefix) for p in self.HISTORICAL)
        return {"isNegated": is_neg, "isFamily": is_fam, "isHistorical": is_hist}

    def _check_llm(self, mention: str, mtype: str, context: str) -> dict:
        if self.llm is None:
            return {"isNegated": False, "isFamily": False, "isHistorical": False}
        system = """Determine assertions for Vietnamese medical mention:
- isNegated: negated (không, chưa)
- isFamily: family history (bố, mẹ, người nhà)
- isHistorical: past/historical (tiền sử, trước đây)

Return JSON: {"isNegated": bool, "isFamily": bool, "isHistorical": bool}"""
        user = f"Mention: \"{mention}\"\nType: {mtype}\nContext: \"{context[:400]}\""
        result = self.llm_decide(system, user, 100)
        if isinstance(result, dict) and "isNegated" in result:
            return result
        return {"isNegated": False, "isFamily": False, "isHistorical": False}

    def _is_uncertain(self, assertions: dict) -> bool:
        neg, fam, hist = assertions.get("isNegated"), assertions.get("isFamily"), assertions.get("isHistorical")
        if fam and hist:
            return True
        if neg and (fam or hist):
            return True
        return False

    def run(self, ctx: AgentContext) -> AgentContext:
        self.log("start", {"mentions": len(ctx.mentions)})

        for m in ctx.mentions:
            start, end = m["position"][0], m["position"][1]
            assertions = self.call_tool("check_assertion", text=ctx.text, start=start, end=end)

            if self._is_uncertain(assertions) and self.llm:
                llm_result = self.call_tool("llm_assertion",
                    mention=m.get("text", ""), mtype=m.get("type", ""),
                    context=ctx.text[max(0, start - 150):end + 150])
                if llm_result:
                    assertions = llm_result

            m["assertions"] = assertions

        self.log("complete", {"processed": len(ctx.mentions)})
        return ctx


class LinkingAgent(Agent):
    def __init__(self, llm=None, kb=None):
        self.kb = kb
        tools = [
            Tool("search_icd10", "Search ICD-10 candidates", self._search_icd10),
            Tool("search_rxnorm", "Search RxNorm candidates", self._search_rxnorm),
            Tool("rerank", "LLM rerank candidates", self._rerank),
        ]
        super().__init__("linking", llm, tools)

    def _search_icd10(self, query: str, top_k: int = 5) -> list:
        if self.kb is None:
            return []
        results = self.kb.search_icd10(query, top_k)
        return [r["code"] for r in results]

    def _search_rxnorm(self, query: str, top_k: int = 5) -> list:
        if self.kb is None:
            return []
        results = self.kb.search_rxnorm(query, top_k)
        return [r["rxcui"] for r in results]

    def _rerank(self, mention: str, context: str, candidates: list, code_type: str) -> list:
        if self.llm is None or not candidates:
            return candidates[:3]
        if len(candidates) <= 1:
            return candidates

        if code_type == "icd10" and self.kb:
            cand_desc = "\n".join([f"- {c}: {self.kb.get_icd10_code(c).get('vietnamese_name', '')[:50]}" for c in candidates[:6]])
        elif code_type == "rxnorm" and self.kb:
            cand_desc = "\n".join([f"- {c}: {self.kb.get_rxnorm_concept(c).get('prefLabel', '')[:50]}" for c in candidates[:6]])
        else:
            cand_desc = "\n".join([f"- {c}" for c in candidates[:6]])

        system = f"""Select best code(s) from candidates for this {code_type} mention.
Return JSON: {{"selected": ["code1"], "reason": "brief"}}"""
        user = f"Mention: \"{mention}\"\nContext: \"{context[:200]}\"\nCandidates:\n{cand_desc}"
        result = self.llm_decide(system, user, 200)
        if isinstance(result, dict) and "selected" in result:
            valid = [c for c in result["selected"] if c in candidates]
            if valid:
                return valid[:3]
        return candidates[:3]

    def run(self, ctx: AgentContext) -> AgentContext:
        self.log("start", {"mentions": len(ctx.mentions)})

        for m in ctx.mentions:
            mtype = m.get("type", "")
            text = m.get("text", "")
            if mtype == "CHẨN_ĐOÁN":
                candidates = self.call_tool("search_icd10", query=text)
                ctx_start = max(0, m["position"][0] - 150)
                ctx_end = min(len(ctx.text), m["position"][1] + 150)
                context = ctx.text[ctx_start:ctx_end]
                if candidates and self.llm:
                    candidates = self.call_tool("rerank", mention=text, context=context, candidates=candidates, code_type="icd10")
                m["candidates"] = candidates
            elif mtype == "THUỐC":
                import re
                ingredient = re.sub(r"\d+(?:\.\d+)?\s*(?:mg|ml|g|mcg|IU|U)\b", "", text, flags=re.IGNORECASE).strip()
                candidates = self.call_tool("search_rxnorm", query=ingredient or text)
                ctx_start = max(0, m["position"][0] - 150)
                ctx_end = min(len(ctx.text), m["position"][1] + 150)
                context = ctx.text[ctx_start:ctx_end]
                if candidates and self.llm:
                    candidates = self.call_tool("rerank", mention=text, context=context, candidates=candidates, code_type="rxnorm")
                m["candidates"] = candidates

        self.log("complete", {"linked": len([m for m in ctx.mentions if m.get("candidates")])})
        return ctx


class VerifierAgent(Agent):
    VALID_TYPES = {"THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}

    def __init__(self, llm=None):
        tools = [
            Tool("validate_schema", "Validate output schema", self._validate),
            Tool("fix_mention", "LLM fix invalid mention", self._fix_llm),
        ]
        super().__init__("verifier", llm, tools)

    def _validate(self, mentions: list) -> dict:
        issues = []
        for i, m in enumerate(mentions):
            if m.get("type") not in self.VALID_TYPES:
                issues.append({"idx": i, "issue": f"invalid type: {m.get('type')}"})
            pos = m.get("position", [0, 0])
            if not isinstance(pos, list) or len(pos) != 2 or pos[1] <= pos[0]:
                issues.append({"idx": i, "issue": "invalid position"})
            if m.get("type") in ("CHẨN_ĐOÁN", "THUỐC") and not m.get("candidates"):
                issues.append({"idx": i, "issue": "missing candidates"})
        return {"valid": len(issues) == 0, "issues": issues}

    def _fix_llm(self, mention: dict, issue: str) -> dict:
        if self.llm is None:
            return mention
        system = f"""Fix this invalid medical mention extraction.
Valid types: {', '.join(self.VALID_TYPES)}
Issue: {issue}
Return JSON with fixed fields: {{"type": "...", "assertions": {{...}}, "candidates": [...]}}"""
        user = f"Mention: {mention}"
        result = self.llm_decide(system, user, 200)
        if isinstance(result, dict):
            if result.get("type") in self.VALID_TYPES:
                mention["type"] = result["type"]
            if "assertions" in result:
                mention["assertions"] = result["assertions"]
            if "candidates" in result and isinstance(result["candidates"], list):
                mention["candidates"] = result["candidates"]
        return mention

    def run(self, ctx: AgentContext) -> AgentContext:
        self.log("start", {"mentions": len(ctx.mentions)})

        validation = self.call_tool("validate_schema", mentions=ctx.mentions)
        if not validation["valid"]:
            self.log("issues_found", {"count": len(validation["issues"])})
            for issue in validation["issues"]:
                idx = issue["idx"]
                if idx < len(ctx.mentions):
                    ctx.mentions[idx] = self.call_tool("fix_mention", mention=ctx.mentions[idx], issue=issue["issue"])

        final_validation = self.call_tool("validate_schema", mentions=ctx.mentions)
        self.log("complete", {"valid": final_validation["valid"], "issues": len(final_validation["issues"])})
        return ctx
