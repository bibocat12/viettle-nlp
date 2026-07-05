import json
import sys
import time
from typing import Callable, Dict, List, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tools import preprocess_note, rule_extract, validate_output
from ner_tool import ner_and_type
from assertion_agent import classify_assertions
from linker_agent import link_candidates
from verifier_agent import verify_all


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

    def _merge_mentions(self, rule_mentions: list, ner_mentions: list) -> list:
        all_mentions = []
        for m in rule_mentions:
            m["source"] = "rule"
            all_mentions.append(m)
        for m in ner_mentions:
            if "type_candidates" in m and "type" not in m:
                candidates = m["type_candidates"]
                if candidates:
                    best = max(candidates, key=lambda x: x.get("score", 0))
                    m["type"] = best.get("label", "TRIỆU_CHỨNG")
                    m["confidence"] = best.get("score", 0.6)
                else:
                    m["type"] = "TRIỆU_CHỨNG"
                    m["confidence"] = 0.6
            overlap = False
            for existing in all_mentions:
                es, ee = existing["position"]
                ns, ne = m["position"]
                if ns < ee and ne > es:
                    overlap = True
                    break
            if not overlap:
                m["source"] = "ner"
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

    def run(self, note_id: str, text: str) -> dict:
        self._log("orchestrator", "start", input={"note_id": note_id, "text_len": len(text), "text_preview": text[:200]})

        # 1. Preprocess
        preprocessed = preprocess_note(note_id, text)
        normalized = preprocessed["normalized_text"]
        self._log("orchestrator", "preprocessed",
                  input={"text_len": len(text)},
                  output={"sentences": len(preprocessed["sentences"]), "normalized_len": len(normalized)})

        # 2. Rule extraction
        rule_result = rule_extract(normalized)
        rule_mentions = rule_result["mentions"]
        assertion_cues = rule_result["assertion_cues"]
        self._log("orchestrator", "rule_extracted",
                  input={"text_len": len(normalized)},
                  output={"mentions": len(rule_mentions), "cues": len(assertion_cues),
                          "mention_types": {m.get("type"): sum(1 for x in rule_mentions if x.get("type") == m.get("type")) for m in rule_mentions}})

        for m in rule_mentions:
            self._log("rules", "found",
                      input={"text": m.get("text", "")},
                      output={"type": m.get("type"), "position": m.get("position"), "confidence": m.get("confidence", 0)})

        # 3. NER extraction
        ner_result = ner_and_type(normalized, llm=self.llm)
        ner_mentions = ner_result["mentions"]
        self._log("orchestrator", "ner_extracted",
                  input={"text_len": len(normalized), "llm_used": self.llm is not None},
                  output={"mentions": len(ner_mentions)})

        for m in ner_mentions[:10]:
            mtype = m.get("type", m.get("type_candidates", [{}])[0].get("label", ""))
            self._log("ner", "found",
                      input={"text": m.get("text", ""), "source": m.get("source", "")},
                      output={"type": mtype, "position": m.get("position"), "candidates": m.get("type_candidates", [])})

        # 4. Merge mentions
        merged = self._merge_mentions(rule_mentions, ner_mentions)
        self._log("orchestrator", "merged",
                  input={"rule_count": len(rule_mentions), "ner_count": len(ner_mentions)},
                  output={"merged_count": len(merged), "dropped": len(rule_mentions) + len(ner_mentions) - len(merged)})

        # 5. Classify assertions for each mention
        for i, mention in enumerate(merged):
            assertion_result = classify_assertions(normalized, mention, llm=self.llm)
            mention["assertions"] = {
                "isNegated": assertion_result["isNegated"],
                "isFamily": assertion_result["isFamily"],
                "isHistorical": assertion_result["isHistorical"],
            }
            mention["assertion_confidence"] = assertion_result.get("confidence", 0.85)
            mention["assertion_evidence"] = assertion_result.get("evidence", "")

            self._log("assertion", "classified",
                      input={"mention": mention.get("text", ""), "type": mention.get("type", ""),
                             "context": normalized[max(0, mention["position"][0]-50):mention["position"][1]+50]},
                      output={"assertions": mention["assertions"], "confidence": mention["assertion_confidence"],
                              "evidence": mention.get("assertion_evidence", "")})

        # 6. Link candidates for CHẨN_ĐOÁN and THUỐC
        for i, mention in enumerate(merged):
            mtype = mention.get("type", "")
            if mtype in ("CHẨN_ĐOÁN", "THUỐC"):
                ctx_start = max(0, mention["position"][0] - 150)
                ctx_end = min(len(normalized), mention["position"][1] + 150)
                context = normalized[ctx_start:ctx_end]

                link_result = link_candidates(mention, context, self.kb, llm=self.llm)
                mention["candidates"] = link_result.get("candidates", [])
                mention["link_confidence"] = link_result.get("confidence", 0)

                self._log("linker", "linked",
                          input={"mention": mention.get("text", ""), "type": mtype, "context": context[:200]},
                          output={"candidates": mention["candidates"][:5], "confidence": mention["link_confidence"]})

        self._log("orchestrator", "linked",
                  input={"total_mentions": len(merged)},
                  output={"linked_count": len([m for m in merged if m.get("candidates")])})

        # 7. Verify all mentions
        verify_result = verify_all(merged, normalized, llm=self.llm)
        final_mentions = verify_result["final_predictions"]
        dropped = verify_result["dropped"]
        self._log("orchestrator", "verified",
                  input={"total": len(merged)},
                  output={"kept": len(final_mentions), "dropped": dropped,
                          "actions": [r.get("action") for r in verify_result.get("actions", [])]})

        # 8. Build output
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

        # 9. Validate output
        validation = validate_output(output_mentions)
        self._log("orchestrator", "validated",
                  input={"mention_count": len(output_mentions)},
                  output={"is_valid": validation["is_valid"], "errors": validation["errors"], "warnings": validation["warnings"]})

        if not validation["is_valid"]:
            self._log("orchestrator", "validation_failed", errors=validation["errors"])

        self._log("orchestrator", "complete",
                  input={"note_id": note_id},
                  output={"total_mentions": len(output_mentions), "stats": {
                      "rule_mentions": len(rule_mentions),
                      "ner_mentions": len(ner_mentions),
                      "merged": len(merged),
                      "final": len(output_mentions),
                      "dropped": dropped,
                  }})

        return {
            "note_id": note_id,
            "mentions": output_mentions,
            "validation": validation,
            "stats": {
                "rule_mentions": len(rule_mentions),
                "ner_mentions": len(ner_mentions),
                "merged": len(merged),
                "final": len(output_mentions),
                "dropped": dropped,
            },
        }


def run_pipeline(note_id: str, text: str, kb=None, llm=None, on_log=None) -> dict:
    orchestrator = Orchestrator(kb=kb, llm=llm, on_log=on_log)
    return orchestrator.run(note_id, text)
