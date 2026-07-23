#!/usr/bin/env python3
"""Span-aware GRPO reward for the Viettel medical NER v2 datasets."""

from __future__ import annotations

import json
from difflib import SequenceMatcher


VALID_TYPES = {
    "THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN",
    "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM",
}
VALID_ASSERTIONS = {"isNegated", "isFamily", "isHistorical"}


def parse_completion(completion) -> list[dict]:
    if isinstance(completion, list):
        if completion and isinstance(completion[0], dict) and "content" in completion[0]:
            completion = completion[0].get("content", "")
        elif all(isinstance(item, dict) for item in completion):
            return completion
    if isinstance(completion, dict):
        completion = completion.get("content", "")
    if not isinstance(completion, str):
        return []
    decoder = json.JSONDecoder()
    for index, char in enumerate(completion):
        if char != "[":
            continue
        try:
            value, _ = decoder.raw_decode(completion[index:])
            return value if isinstance(value, list) else []
        except json.JSONDecodeError:
            continue
    return []


def entity_text(entity: dict) -> str:
    return str(entity.get("entity") or entity.get("text") or "")


def valid_position(entity: dict) -> tuple[int, int] | None:
    position = entity.get("position")
    if (
        isinstance(position, list) and len(position) == 2
        and all(isinstance(value, int) for value in position)
        and 0 <= position[0] < position[1]
    ):
        return position[0], position[1]
    return None


def span_iou(left: tuple[int, int] | None, right: tuple[int, int] | None) -> float:
    if left is None or right is None:
        return 0.0
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union else 0.0


def text_similarity(left: str, right: str) -> float:
    left, right = left.casefold().strip(), right.casefold().strip()
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def jaccard(left: set, right: set) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right) if left | right else 1.0


def assertion_labels(assertions: object) -> set[str] | None:
    """Normalize list and boolean-object assertion forms for reward scoring.

    The competition schema requires a list of label strings, but Qwen3/vLLM
    sometimes emits ``{"isNegated": true, "isFamily": false}``.  Treat that
    object as semantically equivalent for the assertion component so a single
    malformed format cannot crash or erase the whole GRPO signal.  The schema
    component still requires the canonical list form and penalizes the object.
    """
    if isinstance(assertions, list):
        labels: set[str] = set()
        for label in assertions:
            if not isinstance(label, str) or label not in VALID_ASSERTIONS:
                return None
            labels.add(label)
        return labels
    if isinstance(assertions, dict):
        labels = set()
        for label, value in assertions.items():
            if label not in VALID_ASSERTIONS or not isinstance(value, bool):
                return None
            if value:
                labels.add(label)
        return labels
    return None


def schema_score(predictions: list) -> float:
    if not isinstance(predictions, list):
        return 0.0
    if not predictions:
        return 1.0
    valid = 0
    for entity in predictions:
        if not isinstance(entity, dict):
            continue
        assertions = entity.get("assertions", [])
        candidates = entity.get("candidates", [])
        if (
            entity_text(entity)
            and entity.get("type") in VALID_TYPES
            and valid_position(entity) is not None
            and isinstance(assertions, list)
            and assertion_labels(assertions) is not None
            and isinstance(candidates, list)
            and len(candidates) <= 1
        ):
            valid += 1
    return valid / len(predictions)


def align_entities(predictions: list[dict], ground_truth: list[dict]) -> list[tuple[int, int, float]]:
    edges = []
    for gt_index, gt in enumerate(ground_truth):
        for pred_index, pred in enumerate(predictions):
            if gt.get("type") != pred.get("type"):
                continue
            text_score = text_similarity(entity_text(gt), entity_text(pred))
            overlap = span_iou(valid_position(gt), valid_position(pred))
            if text_score < 0.45 and overlap == 0:
                continue
            score = 0.70 * text_score + 0.30 * overlap
            edges.append((score, gt_index, pred_index, text_score))
    edges.sort(reverse=True)
    used_gt, used_pred, matches = set(), set(), []
    for _, gt_index, pred_index, text_score in edges:
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append((gt_index, pred_index, text_score))
    return matches


def score_entities(predictions: list[dict], ground_truth: list[dict]) -> dict[str, float]:
    predictions = [entity for entity in predictions if isinstance(entity, dict)]
    ground_truth = [entity for entity in ground_truth if isinstance(entity, dict)]
    if not predictions and not ground_truth:
        return {"score": 1.0, "text": 1.0, "assertions": 1.0, "candidates": 1.0, "schema": 1.0}
    if not predictions:
        return {"score": 0.0, "text": 0.0, "assertions": 0.0, "candidates": 0.0, "schema": 1.0}

    matches = align_entities(predictions, ground_truth)
    matched_gt = {gt for gt, _, _ in matches}
    matched_pred = {pred for _, pred, _ in matches}
    union_count = max(len(ground_truth) + len(predictions) - len(matches), 1)

    text_total = sum(similarity for _, _, similarity in matches)
    text_score = text_total / union_count

    assertion_total = 0.0
    for gt_index, pred_index, _ in matches:
        gt, pred = ground_truth[gt_index], predictions[pred_index]
        if gt.get("_assertion_supervised", True):
            gt_assertions = assertion_labels(gt.get("assertions", []))
            pred_assertions = assertion_labels(pred.get("assertions", []))
            if gt_assertions is not None and pred_assertions is not None:
                assertion_total += jaccard(gt_assertions, pred_assertions)
        else:
            assertion_total += 1.0
    assertion_score = assertion_total / union_count

    candidate_numerator = 0.0
    candidate_denominator = 0.0
    for gt_index, pred_index, _ in matches:
        gt, pred = ground_truth[gt_index], predictions[pred_index]
        if not gt.get("_candidate_supervised", True):
            continue
        gt_codes = set(map(str, gt.get("candidates", [])))
        pred_codes = set(map(str, pred.get("candidates", [])))
        weight = len(gt_codes) + 1
        candidate_numerator += weight * jaccard(gt_codes, pred_codes)
        candidate_denominator += weight
    for gt_index, gt in enumerate(ground_truth):
        if gt_index in matched_gt or not gt.get("_candidate_supervised", True):
            continue
        candidate_denominator += len(gt.get("candidates", [])) + 1
    for pred_index, pred in enumerate(predictions):
        if pred_index not in matched_pred and pred.get("candidates"):
            candidate_denominator += 1
    candidate_score = (
        candidate_numerator / candidate_denominator if candidate_denominator else 1.0
    )

    format_score = schema_score(predictions)
    leaderboard_score = 0.30 * text_score + 0.30 * assertion_score + 0.40 * candidate_score
    final = leaderboard_score * (0.90 + 0.10 * format_score)
    return {
        "score": final,
        "text": text_score,
        "assertions": assertion_score,
        "candidates": candidate_score,
        "schema": format_score,
    }


def evaluate(completions: list, **kwargs) -> list[float]:
    """TRL/Unsloth-compatible batch reward entry point."""
    ground = kwargs.get("ground_truth", kwargs.get("entities", kwargs.get("gt_entities", [])))
    batched_ground = (
        isinstance(ground, list) and len(ground) == len(completions)
        and (not ground or isinstance(ground[0], list))
    )
    scores = []
    for index, completion in enumerate(completions):
        truth = ground[index] if batched_ground else ground
        scores.append(score_entities(parse_completion(completion), truth or [])["score"])
    return scores
