"""Fireworks-compatible evaluator for Medical NER GRPO.

Matches the evaluate(messages, **kwargs) -> dict signature
expected by the Fireworks RL evaluation framework.

Metrics (same as the reward function):
  - text_score:      1 - WER (char-level Levenshtein) on entity text
  - assertions_score: Jaccard similarity of assertion sets
  - candidates_score: weighted Jaccard similarity of candidate sets
  - final_score:     0.3 * text + 0.3 * assertions + 0.4 * candidates
"""

import json
import math
import re


# ── Character-level WER (Levenshtein) ────────────────────────────────────

def wer_char(ref: str, hyp: str) -> float:
    if not ref and not hyp:
        return 0.0
    if not ref or not hyp:
        return 1.0
    r, c = len(ref) + 1, len(hyp) + 1
    dp = [[0] * c for _ in range(r)]
    for i in range(r):
        dp[i][0] = i
    for j in range(c):
        dp[0][j] = j
    for i in range(1, r):
        for j in range(1, c):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1] / max(len(ref), len(hyp))


# ── Jaccard similarity ───────────────────────────────────────────────────

def jaccard(gt_set: set, pred_set: set) -> float:
    if not gt_set and not pred_set:
        return 1.0
    if not gt_set:
        return 0.0
    return len(gt_set & pred_set) / len(gt_set | pred_set)


# ── Parsing ──────────────────────────────────────────────────────────────

def parse_entities(text: str) -> list[dict]:
    """Extract JSON entity array from model output."""
    if not text:
        return []
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if not m:
        return []
    try:
        pred = json.loads(m.group())
        return pred if isinstance(pred, list) else []
    except json.JSONDecodeError:
        return []


def get_entity_text(e: dict) -> str:
    return e.get("entity") or e.get("text") or ""


# ── Core scoring ─────────────────────────────────────────────────────────

def score_entities(pred_entities: list[dict],
                   gt_entities: list[dict]) -> dict:
    """Score prediction against ground truth.

    Returns a dict of component scores and the final.
    """
    if not pred_entities and not gt_entities:
        return _result(1.0, 1.0, 1.0, 1.0, "both empty")

    if not pred_entities:
        # No prediction: each GT entity is a one-sided key → 0 for all
        n = max(len(gt_entities), 1)
        return _result(0.0, 0.0, 0.0, 0.0, f"no prediction ({len(gt_entities)} GT entities missed)")

    # Build keyed sets (entity_text.lower(), type)
    def keyed(entities):
        d = {}
        for e in entities:
            k = (get_entity_text(e).lower().strip(), e.get("type", ""))
            d.setdefault(k, []).append(e)
        return d

    gt_by_key = keyed(gt_entities)
    pred_by_key = keyed(pred_entities)
    all_keys = set(gt_by_key) | set(pred_by_key)

    text_scores, assertion_scores = [], []
    candidate_scores, candidate_weights = [], []

    for key in all_keys:
        gt_list = gt_by_key.get(key, [])
        pred_list = pred_by_key.get(key, [])

        if not gt_list or not pred_list:
            text_scores.append(0.0)
            assertion_scores.append(1.0 if not gt_list and not pred_list else 0.0)
            candidate_scores.append(0.0)
            candidate_weights.append(1)
            continue

        gt, pred = gt_list[0], pred_list[0]

        gt_text = get_entity_text(gt)
        pred_text = pred.get("entity", "")
        text_scores.append(1.0 - wer_char(gt_text, pred_text))

        ga = set(gt.get("assertions", []))
        pa = set(pred.get("assertions", []))
        assertion_scores.append(jaccard(ga, pa))

        gc = set(gt.get("candidates", []))
        pc = set(pred.get("candidates", []))
        cs = jaccard(gc, pc)
        w = len(gc) + 1
        candidate_scores.append(cs)
        candidate_weights.append(w)

    n = max(len(text_scores), 1)
    text_score = sum(text_scores) / n
    assertions_score = sum(assertion_scores) / n
    candidates_score = (
        sum(s * w for s, w in zip(candidate_scores, candidate_weights))
        / max(sum(candidate_weights), 1)
        if candidate_weights else 0.0
    )
    final = 0.3 * text_score + 0.3 * assertions_score + 0.4 * candidates_score

    return _result(final, text_score, assertions_score, candidates_score)


def _result(final, text, assertions, candidates, reason=""):
    return {
        "score": final,
        "is_score_valid": True,
        "reason": reason or f"text={text:.3f} assertions={assertions:.3f} candidates={candidates:.3f}",
        "extracted_completion_answer": "",
        "extracted_ground_truth_answer": "",
        "metrics": {
            "text_score": {
                "is_score_valid": True,
                "score": text,
                "reason": "1 - WER on entity text",
            },
            "assertions_score": {
                "is_score_valid": True,
                "score": assertions,
                "reason": "Jaccard on assertion sets",
            },
            "candidates_score": {
                "is_score_valid": True,
                "score": candidates,
                "reason": "Weighted Jaccard on candidate sets",
            },
        },
    }


# ── Required entry point ─────────────────────────────────────────────────

def evaluate(messages: list[dict], **kwargs) -> dict:
    """Evaluate a single entry. Required by Fireworks evaluator framework.

    Args:
        messages: Dataset row — last message is the model completion.
        kwargs: May contain 'ground_truth' (list of entity dicts).

    Returns:
        dict with is_score_valid, score, reason, metrics.
    """
    ground_truth = kwargs.get("ground_truth", [])
    completion = messages[-1]["content"] if messages else ""

    pred_entities = parse_entities(completion)

    # Parse ground truth if it's a JSON string
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            ground_truth = []

    return score_entities(pred_entities, ground_truth)
