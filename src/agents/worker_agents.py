from typing import Any, Dict, List, Optional
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import Agent, Tool, AgentContext
from tools import preprocess_note, rule_extract, validate_output
from ner_tool import ner_and_type
from healthbert import classify_type as healthbert_classify_type
from position_fix import fix_positions
from voting import merge_with_voting, llm_resolve_conflicts
from assertion_agent import classify_assertions
from linker_agent import link_candidates
from verifier_agent import verify_all


class ExtractionAgent(Agent):
    HEALTHBERT_SKIP_STARTS = {
        'tiền sử', 'bệnh sử', 'kết quả', 'các', 'đặc điểm', 'thời điểm',
        'tình trạng', 'lý do', 'triệu chứng', 'yếu tố', 'thuốc trước',
        'các bệnh', 'các yếu tố', 'các triệu chứng', 'các diễn biến',
        'các sự kiện', 'các thủ thuật', 'đánh giá', 'xử trí',
    }

    def __init__(self, llm=None, mode="voting"):
        self.mode = mode  # "voting" (healthbert+llm) or "llm_only"
        tools = [
            Tool("parallel_extract", "Run extraction models", self._parallel_extract),
            Tool("vote_merge", "Merge mentions with voting-based confidence", self._vote_merge),
            Tool("resolve_conflicts", "LLM resolve low-confidence mentions", self._resolve_conflicts),
            Tool("fix_positions", "Re-locate positions in original text", self._fix_positions),
            Tool("llm_verify", "LLM verify low-confidence types", self._llm_verify),
        ]
        super().__init__("extraction", llm, tools)

    def _run_rule(self, text: str) -> list:
        return rule_extract(text)["mentions"]

    def _run_qwen(self, text: str) -> list:
        if self.llm is None:
            return []
        from agents.section_parser import parse_sections
        from concurrent.futures import ThreadPoolExecutor, as_completed

        sections = parse_sections(text)
        if len(sections) <= 1:
            # No sections found, infer on full text
            result = ner_and_type(text, llm=self.llm)
            return result.get("mentions", [])

        # Parallel extraction per section
        all_mentions = []
        with ThreadPoolExecutor(max_workers=min(3, len(sections))) as ex:
            def extract_section(sec):
                mentions = ner_and_type(sec["text"], llm=self.llm).get("mentions", [])
                # Offset positions by section start
                for m in mentions:
                    if m.get("position"):
                        m["position"][0] += sec["start"]
                        m["position"][1] += sec["start"]
                return mentions

            futures = {ex.submit(extract_section, sec): sec["id"] for sec in sections}
            for future in as_completed(futures):
                try:
                    all_mentions.extend(future.result())
                except Exception as e:
                    self.log("error", {"section": futures[future], "message": str(e)})

        return all_mentions

    def _run_healthbert(self, text: str) -> list:
        import re
        mentions = []
        cand_pattern = re.compile(
            r'(?:^|[\s,;:\(\)"])([A-ZÀ-ỹa-zà-ỹ][a-zà-ỹ]+(?:\s+[a-zà-ỹ]+){2,5})(?=[\s,;:\-\.\)]|$)'
        )
        seen = set()
        for m in cand_pattern.finditer(text):
            span_text = m.group(1).strip()
            if len(span_text) < 8 or len(span_text) > 60:
                continue
            first_word = span_text.lower().split()[0] if span_text.split() else ''
            if first_word in self.HEALTHBERT_SKIP_STARTS:
                continue
            s, e = m.start(1), m.end(1)
            pos_key = (s, e)
            if pos_key in seen:
                continue
            seen.add(pos_key)
            ctx_start = max(0, s - 50)
            ctx_end = min(len(text), e + 50)
            context = text[ctx_start:ctx_end]
            result = healthbert_classify_type(span_text, context)
            mtype = result.get("type", "TRIỆU_CHỨNG")
            if mtype == "KHÔNG_PHẢI":
                continue
            mentions.append({
                "text": span_text,
                "position": [s, e],
                "type": mtype,
                "type_candidates": [{"label": mtype, "score": result.get("score", 0.5)}],
            })
        return mentions

    def _parallel_extract(self, text: str) -> dict:
        results = {"qwen": None, "healthbert": None}
        errors = []

        if self.mode == "llm_only":
            results["qwen"] = self._run_qwen(text) or []
        else:
            # Mode "voting": HealthBERT + LLM in parallel
            with ThreadPoolExecutor(max_workers=2) as ex:
                futures = {
                    ex.submit(self._run_qwen, text): "qwen",
                    ex.submit(self._run_healthbert, text): "healthbert",
                }
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        results[name] = future.result()
                    except Exception as e:
                        errors.append(f"{name}: {e}")
                        results[name] = []

        return {
            "rule": [], "qwen": results["qwen"] or [],
            "healthbert": results["healthbert"] or [],
            "errors": errors,
        }

    def _vote_merge(self, all_mentions: list) -> list:
        return merge_with_voting(all_mentions)

    def _resolve_conflicts(self, merged: list, text: str) -> list:
        vote_conflicts = sum(1 for m in merged if m.get("needs_review"))
        if vote_conflicts > 0 and self.llm:
            return llm_resolve_conflicts(merged, text, self.llm)
        return merged

    def _fix_positions(self, mentions: list, text: str) -> list:
        return fix_positions(mentions, text)

    def _llm_verify(self, mention: dict, context: str) -> bool:
        if self.llm is None:
            return False
        system = """Bạn là chuyên gia kiểm tra phân loại thực thể y khoa.
Một hệ thống keyword-based đã phân loại một mention. Hãy xác nhận hoặc sửa lại.

5 loại: THUỐC, TRIỆU_CHỨNG, CHẨN_ĐOÁN, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM

Trả về JSON: {"correct": true/false, "corrected_type": "...", "reason": "..."}"""
        user = f"Mention: \"{mention['text']}\"\nCurrent type: {mention['type']}\nContext: \"{context}\""
        result = self.llm_decide(system, user, 150)
        if isinstance(result, dict) and result.get("correct") is False and result.get("corrected_type"):
            valid = ("THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM")
            if result["corrected_type"] in valid:
                mention["type"] = result["corrected_type"]
                mention["llm_corrected"] = True
                return True
        return False

    def run(self, ctx: AgentContext) -> AgentContext:
        self.log("start", {"text_len": len(ctx.text)})

        preprocessed = preprocess_note(ctx.text, ctx.text)
        normalized = preprocessed["normalized_text"]
        ctx.normalized = normalized
        self.log("preprocessed", {"normalized_len": len(normalized)})

        parallel = self.call_tool("parallel_extract", text=normalized)

        # Flatten all mentions with source tracking
        all_mentions = []
        for source_name in ("qwen", "healthbert"):
            for m in parallel.get(source_name, []):
                if "type_candidates" in m and "type" not in m:
                    candidates = m["type_candidates"]
                    if candidates:
                        best = max(candidates, key=lambda x: x.get("score", 0))
                        m["type"] = best.get("label", "TRIỆU_CHỨNG")
                m["extraction_source"] = source_name
                all_mentions.append(m)

        source_counts = {k: len(v or []) for k, v in parallel.items() if k != "errors"}
        self.log("extracted", {"sources": source_counts, "total": len(all_mentions)})

        # Voting merge
        merged = self.call_tool("vote_merge", all_mentions=all_mentions)
        self.log("voted", {"merged": len(merged)})

        # LLM resolve conflicts
        merged = self.call_tool("resolve_conflicts", merged=merged, text=normalized)

        # Position fix
        merged = self.call_tool("fix_positions", mentions=merged, text=normalized)
        n_fixed = sum(1 for m in merged if m.get("position_fixed"))
        self.log("positions_fixed", {"fixed": n_fixed})

        # LLM verify low-confidence
        llm_corrected = 0
        for m in merged:
            if m.get("needs_review") and not m.get("llm_resolved"):
                ctx_start = max(0, m["position"][0] - 50)
                ctx_end = min(len(normalized), m["position"][1] + 50)
                context = normalized[ctx_start:ctx_end]
                if self.call_tool("llm_verify", mention=m, context=context):
                    llm_corrected += 1
        self.log("llm_verified", {"corrected": llm_corrected})

        ctx.mentions = merged
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
            Tool("rule_assert", "Rule-based assertion check", self._rule_assert),
            Tool("llm_assert", "LLM assertion for all mentions", self._llm_assert),
        ]
        super().__init__("assertion", llm, tools)

    def _rule_assert(self, text: str, start: int, end: int) -> dict:
        import re
        window = 60
        ctx_start = max(0, start - window)
        prefix = text[ctx_start:start].lower()
        return {
            "isNegated": any(re.search(p, prefix) for p in self.NEGATION),
            "isFamily": any(re.search(p, prefix) for p in self.FAMILY),
            "isHistorical": any(re.search(p, prefix) for p in self.HISTORICAL),
        }

    def _llm_assert(self, mention: str, mtype: str, context: str) -> dict:
        if self.llm is None:
            return {"isNegated": False, "isFamily": False, "isHistorical": False}
        system = """Xác định assertion cho thực thể y khoa tiếng Việt:
- isNegated: phủ định (không, chưa)
- isFamily: tiền sử gia đình (bố, mẹ, người nhà)
- isHistorical: tiền sử bản thân (tiền sử, trước đây)

Trả về JSON: {"isNegated": bool, "isFamily": bool, "isHistorical": bool,
"confidence": 0.0-1.0, "reason": "..."}"""
        user = f"Thực thể: \"{mention}\"\nLoại: {mtype}\nNgữ cảnh: \"{context[:400]}\""
        result = self.llm_decide(system, user, 150)
        if isinstance(result, dict) and "isNegated" in result:
            return result
        return {"isNegated": False, "isFamily": False, "isHistorical": False}

    def run(self, ctx: AgentContext) -> AgentContext:
        self.log("start", {"mentions": len(ctx.mentions)})

        for m in ctx.mentions:
            if m.get("type") not in ("CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"):
                m["assertions"] = {"isNegated": False, "isFamily": False, "isHistorical": False}
                continue

            rule_assert = self.call_tool("rule_assert",
                text=ctx.normalized, start=m["position"][0], end=m["position"][1])

            if self.llm and m.get("type") in ("CHẨN_ĐOÁN", "THUỐC"):
                ctx_start = max(0, m["position"][0] - 150)
                ctx_end = min(len(ctx.normalized), m["position"][1] + 150)
                context = ctx.normalized[ctx_start:ctx_end]
                llm_result = self.call_tool("llm_assert",
                    mention=m["text"], mtype=m["type"], context=context)
                m["assertions"] = llm_result
            elif self.llm and m.get("type") == "TRIỆU_CHỨNG" and any(rule_assert.values()):
                # Symptom with detected assertion (negation/family/historical) — LLM verify
                ctx_start = max(0, m["position"][0] - 150)
                ctx_end = min(len(ctx.normalized), m["position"][1] + 150)
                context = ctx.normalized[ctx_start:ctx_end]
                llm_result = self.call_tool("llm_assert",
                    mention=m["text"], mtype=m["type"], context=context)
                m["assertions"] = llm_result
            else:
                m["assertions"] = rule_assert
            m["assertion_confidence"] = m.get("assertions", {}).get("confidence", 0.85)

        self.log("complete", {"processed": len(ctx.mentions), "llm_checked": sum(1 for m in ctx.mentions if m.get("assertions", {}).get("reason"))})
        return ctx


class LinkingAgent(Agent):
    def __init__(self, llm=None, kb=None):
        self.kb = kb
        tools = [
            Tool("link_entity", "Link entity to ICD-10 or RxNorm", self._link_entity),
        ]
        super().__init__("linking", llm, tools)

    def _link_entity(self, mention: dict, context: str) -> dict:
        if self.kb is None:
            return {"candidates": [], "confidence": 0}
        return link_candidates(mention, context, self.kb, llm=self.llm)

    def run(self, ctx: AgentContext) -> AgentContext:
        self.log("start", {"mentions": len(ctx.mentions)})

        linked = 0
        for m in ctx.mentions:
            mtype = m.get("type", "")
            if mtype not in ("CHẨN_ĐOÁN", "THUỐC"):
                continue
            ctx_start = max(0, m["position"][0] - 150)
            ctx_end = min(len(ctx.normalized), m["position"][1] + 150)
            context = ctx.normalized[ctx_start:ctx_end]
            result = self.call_tool("link_entity", mention=m, context=context)
            m["candidates"] = result.get("candidates", [])
            m["link_confidence"] = result.get("confidence", 0)
            if m["candidates"]:
                linked += 1

        self.log("complete", {"linked": linked})
        return ctx


class VerifierAgent(Agent):
    VALID_TYPES = {"THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}

    def __init__(self, llm=None):
        tools = [
            Tool("verify_schema", "Validate and fix output", self._verify),
        ]
        super().__init__("verifier", llm, tools)

    def _verify(self, mentions: list, text: str) -> dict:
        return verify_all(mentions, text, llm=self.llm)

    def run(self, ctx: AgentContext) -> AgentContext:
        self.log("start", {"mentions": len(ctx.mentions)})

        result = self.call_tool("verify_schema", mentions=ctx.mentions, text=ctx.normalized)
        kept = result.get("final_predictions", [])
        dropped = result.get("dropped", 0)

        # Dedup by position: remove overlapping entities, keep highest confidence
        kept_sorted = sorted(kept, key=lambda x: (x.get("position", [0, 0])[0], -x.get("confidence", 0)))
        deduped = []
        for m in kept_sorted:
            s, e = m.get("position", [0, 0])
            overlap = False
            for i, prev in enumerate(deduped):
                ps, pe = prev.get("position", [0, 0])
                # Check overlap: if starts within previous span or previous starts within this span
                if (s >= ps and s < pe) or (ps >= s and ps < e):
                    overlap = True
                    # Keep the one with higher confidence
                    if m.get("confidence", 0) > prev.get("confidence", 0):
                        deduped[i] = m
                    break
            if not overlap:
                deduped.append(m)
        extra_dropped = len(kept) - len(deduped)

        self.log("validated", {"kept": len(deduped), "dropped": (dropped if isinstance(dropped, int) else len(dropped)) + extra_dropped})
        ctx.mentions = deduped
        return ctx
