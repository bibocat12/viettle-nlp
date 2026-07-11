import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))


VALID_TYPES = {"THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}
VALID_ASSERTIONS = {"isNegated", "isFamily", "isHistorical"}


def verify_mention(mention: dict, original_text: str, llm=None) -> dict:
    """Verify a single mention and return action.

    Possible actions: accept, retry_assertion, retry_linking, drop_prediction
    """
    errors = []
    warnings = []

    # Check type
    mtype = mention.get("type", "")
    if mtype not in VALID_TYPES:
        return {"action": "drop_prediction", "reason": f"invalid type: {mtype}"}

    # Check position
    pos = mention.get("position", [0, 0])
    if not isinstance(pos, list) or len(pos) != 2:
        return {"action": "drop_prediction", "reason": "invalid position"}
    if pos[1] <= pos[0]:
        return {"action": "drop_prediction", "reason": "position end <= start"}
    if pos[1] > len(original_text):
        return {"action": "drop_prediction", "reason": "position exceeds text length"}

    # Check span text matches original
    span_text = original_text[pos[0]:pos[1]]
    if mention.get("text", "") != span_text:
        # Try to find the text nearby
        nearby = original_text[max(0, pos[0]-20):pos[1]+20]
        if mention.get("text", "") not in nearby:
            return {"action": "drop_prediction", "reason": "span text mismatch"}

    # Check assertions — strip extra keys instead of dropping
    assertions = mention.get("assertions", {})
    for key in list(assertions.keys()):
        if key not in VALID_ASSERTIONS:
            del assertions[key]

    # Check candidates
    if mtype in ("CHẨN_ĐOÁN", "THUỐC"):
        candidates = mention.get("candidates", [])
        if not candidates:
            warnings.append("no candidates")

    # Check assertion confidence
    assertion_conf = mention.get("assertion_confidence", 1.0)
    if assertion_conf < 0.6:
        return {"action": "retry_assertion", "reason": "low assertion confidence"}

    # Use LLM for complex review if available
    if llm and llm.is_available():
        # Check for impossible type-assertion combos
        if assertions.get("isNegated") and mtype == "THUỐC":
            # Negated drugs are unusual - verify
            ctx_start = max(0, pos[0] - 100)
            ctx_end = min(len(original_text), pos[1] + 100)
            context = original_text[ctx_start:ctx_end]

            system = """Review this medical mention. Is the assertion correct?
Return JSON: {"correct": true/false, "reason": "brief"}"""
            user = f"Mention: \"{mention['text']}\"\nType: {mtype}\nAssertions: {assertions}\nContext: \"{context}\""

            response = llm.generate(system, user, 100)
            result = llm.parse_json(response)
            if isinstance(result, dict) and not result.get("correct", True):
                return {"action": "retry_assertion", "reason": result.get("reason", "LLM review")}

    return {
        "action": "accept",
        "fixes": [],
        "warnings": warnings,
    }


def verify_all(predictions: list, original_text: str, llm=None) -> dict:
    """Verify all predictions and return results."""
    results = []
    final_predictions = []

    for pred in predictions:
        result = verify_mention(pred, original_text, llm)
        results.append(result)

        if result["action"] == "accept":
            final_predictions.append(pred)
        elif result["action"] == "retry_assertion":
            # Retry with LLM if available
            if llm and llm.is_available():
                from assertion_agent import classify_assertions
                pred["assertions"] = classify_assertions(original_text, pred, llm)
                final_predictions.append(pred)
            else:
                final_predictions.append(pred)
        elif result["action"] == "retry_linking":
            # Keep the mention but mark for re-linking
            final_predictions.append(pred)
        # drop_prediction: don't include

    return {
        "actions": results,
        "final_predictions": final_predictions,
        "dropped": len(predictions) - len(final_predictions),
    }
