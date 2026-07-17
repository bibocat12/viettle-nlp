"""Custom GRPO evaluator for Unsloth CLI.

Place this file in your training directory and reference it with:
    --reward-function reward.py
"""

import json
import re


def parse_completion(completion: str) -> list:
    """Parse JSON entity array from model output."""
    if not completion:
        return []
    m = re.search(r"\[.*\]", completion, re.DOTALL)
    if not m:
        return []
    try:
        pred = json.loads(m.group())
        return pred if isinstance(pred, list) else []
    except json.JSONDecodeError:
        return []


def get_entity_text(e: dict) -> str:
    return e.get("entity") or e.get("text") or ""


def jaccard(gt_set: set, pred_set: set) -> float:
    if not gt_set and not pred_set:
        return 1.0
    if not gt_set:
        return 0.0
    return len(gt_set & pred_set) / len(gt_set | pred_set)


def wer_char(ref: str, hyp: str) -> float:
    """Character-level Levenshtein WER."""
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


# ── Required function: evaluate(completions, **kwargs) → list[float] ──

def evaluate(completions: list[str], **kwargs) -> list[float]:
    """Score each completion against ground truth.

    Called by Unsloth/TRL GRPOTrainer. Returns one float per completion.

    Scores: 0.3 * text_WER + 0.3 * assertions_Jaccard + 0.4 * candidates_Jaccard
    """
    ground_truth_list = kwargs.get("ground_truth", [])
    if isinstance(ground_truth_list, dict) and "entities" in ground_truth_list:
        ground_truth_list = ground_truth_list["entities"]

    scores = []
    for completion in completions:
        pred_entities = parse_completion(completion)

        if not pred_entities and not ground_truth_list:
            scores.append(1.0)
            continue
        if not pred_entities:
            scores.append(0.0)
            continue

        # Align by (text, type) key
        def keyed(entities):
            d = {}
            for e in entities:
                k = (get_entity_text(e).lower().strip(), e.get("type", ""))
                d.setdefault(k, []).append(e)
            return d

        gt_k = keyed(ground_truth_list)
        pred_k = keyed(pred_entities)
        all_keys = set(gt_k) | set(pred_k)

        text_scores, assert_scores, cand_scores, cand_weights = [], [], [], []

        for key in all_keys:
            gt_list = gt_k.get(key, [])
            pred_list = pred_k.get(key, [])

            if not gt_list or not pred_list:
                text_scores.append(0.0)
                assert_scores.append(0.0)
                cand_scores.append(0.0)
                cand_weights.append(1)
                continue

            gt, pred = gt_list[0], pred_list[0]

            # Text score
            text_scores.append(1.0 - wer_char(get_entity_text(gt), pred.get("entity", "")))

            # Assertion score
            assert_scores.append(jaccard(
                set(gt.get("assertions", [])),
                set(pred.get("assertions", [])),
            ))

            # Candidate score (weighted by GT count)
            gc = set(gt.get("candidates", []))
            pc = set(pred.get("candidates", []))
            cs = jaccard(gc, pc)
            w = len(gc) + 1
            cand_scores.append(cs)
            cand_weights.append(w)

        n = max(len(text_scores), 1)
        text_score = sum(text_scores) / n
        assertions_score = sum(assert_scores) / n
        candidates_score = (
            sum(s * w for s, w in zip(cand_scores, cand_weights))
            / max(sum(cand_weights), 1)
            if cand_weights else 0.0
        )
        final = 0.3 * text_score + 0.3 * assertions_score + 0.4 * candidates_score
        scores.append(final)

    return scores
