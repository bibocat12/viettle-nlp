import json
import sys
import time
from typing import Callable, Dict, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))

from tools import preprocess_note, rule_extract, validate_output
from ner_tool import ner_and_type
from assertion_agent import classify_assertions
from linker_agent import link_candidates
from verifier_agent import verify_all
from healthbert import classify_type as healthbert_classify_type
from healthbert import classify_assertion as healthbert_classify_assertion


NER_LLM_VERIFY_SYSTEM = """Bạn là chuyên gia kiểm tra phân loại thực thể y khoa.
Một hệ thống keyword-based đã phân loại một mention. Hãy xác nhận hoặc sửa lại.

5 loại: THUỐC, TRIỆU_CHỨNG, CHẨN_ĐOÁN, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM

Trả về JSON: {"correct": true/false, "corrected_type": "...", "reason": "..."}"""


class Orchestrator:
    def __init__(self, kb=None, llm=None, on_log: Optional[Callable] = None):
        self.kb = kb
        self.llm = llm
        self.on_log = on_log

    def _log(self, agent: str, action: str, **kwargs):
        entry = {"agent": agent, "action": action, "ts": time.time(), **kwargs}
        if self.on_log:
            self.on_log(entry)
        return entry

    def _merge_mentions(self, mentions_list: List[list]) -> list:
        """Merge mentions from multiple sources, deduplicating by position."""
        all_mentions = []
        seen_positions = set()
        for mentions in mentions_list:
            for m in mentions:
                pos = tuple(m.get("position", [0, 0]))
                if pos in seen_positions:
                    continue
                seen_positions.add(pos)
                all_mentions.append(m)

        all_mentions.sort(key=lambda x: x["position"][0])
        result = []
        for m in all_mentions:
            if not result:
                result.append(m)
                continue
            prev = result[-1]
            if m["position"][0] >= prev["position"][1]:
                result.append(m)
            else:
                cur_len = m["position"][1] - m["position"][0]
                prev_len = prev["position"][1] - prev["position"][0]
                if cur_len > prev_len:
                    result[-1] = m
        return result

    def _llm_verify_healthbert(self, mention: dict, context: str) -> dict:
        """Use LLM to verify/correct ViHealthBERT keyword-based type classification."""
        if not self.llm or not self.llm.is_available():
            return mention

        text = mention.get("text", "")
        mtype = mention.get("type", "")

        user = f"""Mention: "{text}"
Keyword-classified as: {mtype}
Context: "{context[:200]}"

Is this classification correct?"""

        response = self.llm.generate(NER_LLM_VERIFY_SYSTEM, user, 200)
        result = self.llm.parse_json(response)

        if isinstance(result, dict) and not result.get("correct", True):
            corrected = result.get("corrected_type", "")
            if corrected in ("THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"):
                mention["type"] = corrected
                mention["type_candidates"] = [{"label": corrected, "score": 0.85}]
                mention["llm_corrected"] = True
                self._log("llm_verify", "corrected",
                          input={"text": text, "old_type": mtype},
                          output={"new_type": corrected, "reason": result.get("reason", "")})
        return mention

    def run(self, note_id: str, text: str) -> dict:
        self._log("orchestrator", "start", input={"note_id": note_id, "text_len": len(text)})

        preprocessed = preprocess_note(note_id, text)
        normalized = preprocessed["normalized_text"]
        self._log("orchestrator", "preprocessed", input={}, output={"len": len(normalized)})

        # ===== PARALLEL EXTRACTION =====
        # Run all extraction methods simultaneously
        results = {"rule": None, "ner_qwen": None, "ner_healthbert": None}
        errors = []

        def run_rule():
            r = rule_extract(normalized)
            self._log("orchestrator", "rule_extracted", output={"mentions": len(r["mentions"])})
            return r["mentions"]

        def run_ner_qwen():
            r = ner_and_type(normalized, llm=self.llm)
            self._log("orchestrator", "ner_qwen_extracted", output={"mentions": len(r["mentions"])})
            return r["mentions"]

        def run_ner_healthbert():
            import re
            mentions = []
            # Only capture longer, more specific phrases (3-6 words)
            # Skip sentence fragments
            skip_starts = {'tiền sử', 'bệnh sử', 'kết quả', 'các', 'đặc điểm', 'thời điểm',
                          'tình trạng', 'lý do', 'triệu chứng', 'yếu tố', 'thuốc trước',
                          'các bệnh', 'các yếu tố', 'các triệu chứng', 'các diễn biến',
                          'các sự kiện', 'các thủ thuật', 'đánh giá', 'xử trí'}
            cand_pattern = re.compile(
                r'(?:^|[\s,;:\-\(\)])([A-ZÀ-ỹ][a-zà-ỹ]+(?:\s+[a-zà-ỹ]+){2,5})(?=[\s,;:\-\.\)]|$)'
            )
            seen = set()
            for m in cand_pattern.finditer(normalized):
                span_text = m.group(1).strip()
                if len(span_text) < 8 or len(span_text) > 60:
                    continue
                first_word = span_text.lower().split()[0] if span_text.split() else ''
                if first_word in skip_starts:
                    continue
                s, e = m.start(1), m.end(1)
                pos_key = (s, e)
                if pos_key in seen:
                    continue
                seen.add(pos_key)
                ctx_start = max(0, s - 50)
                ctx_end = min(len(normalized), e + 50)
                context = normalized[ctx_start:ctx_end]
                result = healthbert_classify_type(span_text, context)
                mentions.append({
                    "text": span_text,
                    "position": [s, e],
                    "type": result["type"],
                    "type_candidates": [{"label": result["type"], "score": result["score"]}],
                    "source": "healthbert",
                    "confidence": result["score"],
                })
                if len(mentions) >= 15:
                    break
            self._log("orchestrator", "ner_healthbert_extracted", output={"mentions": len(mentions)})
            return mentions

        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = {
                ex.submit(run_rule): "rule",
                ex.submit(run_ner_qwen): "ner_qwen",
                ex.submit(run_ner_healthbert): "ner_healthbert",
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    errors.append(f"{name}: {e}")
                    results[name] = []
                    self._log("error", f"{name}_failed", message=str(e))

        if errors:
            self._log("orchestrator", "parallel_errors", errors=errors)

        # ===== MERGE =====
        # Convert type_candidates to type for mentions that don't have 'type'
        all_mentions = []
        for source_name, mentions in results.items():
            if mentions:
                for m in mentions:
                    if "type_candidates" in m and "type" not in m:
                        candidates = m["type_candidates"]
                        if candidates:
                            best = max(candidates, key=lambda x: x.get("score", 0))
                            m["type"] = best.get("label", "TRIỆU_CHỨNG")
                    m["extraction_source"] = source_name
                    all_mentions.append(m)

        merged = self._merge_mentions([all_mentions])
        self._log("orchestrator", "merged", output={"total": len(merged), "from": {k: len(v or []) for k, v in results.items()}})

        # ===== LLM REASONING on ViHealthBERT output (only low-confidence) =====
        llm_corrected = 0
        for i, m in enumerate(merged):
            if m.get("confidence", 1.0) < 0.85 and m.get("extraction_source") == "ner_healthbert":
                ctx_start = max(0, m["position"][0] - 50)
                ctx_end = min(len(normalized), m["position"][1] + 50)
                context = normalized[ctx_start:ctx_end]
                self._llm_verify_healthbert(m, context)
                if m.get("llm_corrected"):
                    llm_corrected += 1

        self._log("orchestrator", "llm_verified", output={"healthbert_mentions": sum(1 for m in merged if m.get("extraction_source") == "ner_healthbert"), "corrected": llm_corrected})

        # ===== ASSERTION =====
        for i, mention in enumerate(merged):
            assertion_result = classify_assertions(normalized, mention, llm=self.llm)
            mention["assertions"] = {
                "isNegated": assertion_result["isNegated"],
                "isFamily": assertion_result["isFamily"],
                "isHistorical": assertion_result["isHistorical"],
            }
            mention["assertion_confidence"] = assertion_result.get("confidence", 0.85)

        self._log("orchestrator", "assertions_classified", count=len(merged))

        # ===== LINKING =====
        for i, mention in enumerate(merged):
            mtype = mention.get("type", "")
            if mtype in ("CHẨN_ĐOÁN", "THUỐC"):
                ctx_start = max(0, mention["position"][0] - 150)
                ctx_end = min(len(normalized), mention["position"][1] + 150)
                context = normalized[ctx_start:ctx_end]

                link_result = link_candidates(mention, context, self.kb, llm=self.llm)
                mention["candidates"] = link_result.get("candidates", [])
                mention["link_confidence"] = link_result.get("confidence", 0)

        self._log("orchestrator", "linked", count=len([m for m in merged if m.get("candidates")]))

        # ===== VERIFICATION =====
        verify_result = verify_all(merged, normalized, llm=self.llm)
        final_mentions = verify_result["final_predictions"]
        dropped = verify_result["dropped"]
        self._log("orchestrator", "verified", kept=len(final_mentions), dropped=dropped)

        # ===== OUTPUT =====
        output_mentions = []
        for m in final_mentions:
            out = {
                "text": m.get("text", ""),
                "position": m.get("position", [0, 0]),
                "type": m.get("type", ""),
                "assertions": m.get("assertions", {
                    "isNegated": False,
                    "isFamily": False,
                    "isHistorical": False,
                }),
            }
            if m.get("type") in ("CHẨN_ĐOÁN", "THUỐC"):
                out["candidates"] = m.get("candidates", [])
            output_mentions.append(out)

        validation = validate_output(output_mentions)
        self._log("orchestrator", "validated", is_valid=validation["is_valid"], errors=len(validation["errors"]))
        self._log("orchestrator", "complete", total_mentions=len(output_mentions))

        return {
            "note_id": note_id,
            "mentions": output_mentions,
            "validation": validation,
            "stats": {
                "rule_mentions": len(results.get("rule", [])),
                "ner_qwen_mentions": len(results.get("ner_qwen", [])),
                "ner_healthbert_mentions": len(results.get("ner_healthbert", [])),
                "merged": len(merged),
                "final": len(output_mentions),
                "dropped": dropped,
                "llm_corrected": llm_corrected,
            },
        }


def run_pipeline(note_id: str, text: str, kb=None, llm=None, on_log=None) -> dict:
    orchestrator = Orchestrator(kb=kb, llm=llm, on_log=on_log)
    return orchestrator.run(note_id, text)
