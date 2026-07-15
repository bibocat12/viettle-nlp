import json
import logging
import os
import re

from eval_protocol.models import EvaluateResult, EvaluationRow, MetricResult
from eval_protocol.pytest import RemoteRolloutProcessor, evaluation_test

# Remote agent URL — set REMOTE_AGENT_URL env var before uploading
# Default is localhost for testing, use ngrok URL for real RFT
REMOTE_URL = os.getenv("REMOTE_AGENT_URL", "http://localhost:8080")
print(f"Using remote agent URL: {REMOTE_URL}")
ROLLOUT_PROCESSOR = RemoteRolloutProcessor(remote_base_url=REMOTE_URL)

logger = logging.getLogger(__name__)


def extract_entities(text: str) -> list:
    """Parse JSON entity array from model output.  Falls back to single-entity
    extraction from free text so the evaluator gets non-zero signal even
    before the model learns the JSON format."""
    if not text:
        return []

    # Try JSON array first
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            pred = json.loads(m.group())
            if isinstance(pred, list):
                return pred
        except json.JSONDecodeError:
            pass

    # Fallback: extract any entity-like patterns from free text
    # Look for patterns like "entity: X, type: Y" or "đau ngực (TRIỆU_CHỨNG)"
    entities = []

    # Pattern 1: "text (TYPE)" or "text - TYPE"
    pairs = re.findall(r'[""]?([^""]+?)[""]?\s*[\(\-–]\s*(\w+)\s*[\)]?', text)
    for entity_text, etype in pairs:
        entity_text = entity_text.strip()
        etype = etype.strip().upper()
        if etype in ("THUỐC","TRIỆU_CHỨNG","CHẨN_ĐOÁN","TÊN_XÉT_NGHIỆM","KẾT_QUẢ_XÉT_NGHIỆM","THỦ_THUẬT"):
            entities.append({"entity": entity_text, "type": etype})

    # Pattern 2: look for known medical terms from the text
    known_terms = [
        ("đau", "TRIỆU_CHỨNG"), ("sốt", "TRIỆU_CHỨNG"), ("ho", "TRIỆU_CHỨNG"),
        ("khó thở", "TRIỆU_CHỨNG"), ("mệt", "TRIỆU_CHỨNG"), ("buồn nôn", "TRIỆU_CHỨNG"),
        ("tăng huyết áp", "CHẨN_ĐOÁN"), ("đái tháo đường", "CHẨN_ĐOÁN"),
        ("viêm phổi", "CHẨN_ĐOÁN"), ("lao", "CHẨN_ĐOÁN"),
        ("WBC", "TÊN_XÉT_NGHIỆM"), ("CRP", "TÊN_XÉT_NGHIỆM"), ("ALT", "TÊN_XÉT_NGHIỆM"),
    ]
    found_texts = {e["entity"].lower() for e in entities}
    for term, etype in known_terms:
        if term in text.lower() and term not in found_texts:
            entities.append({"entity": term, "type": etype})
            found_texts.add(term)

    return entities


def get_entity_text(e: dict) -> str:
    return e.get("entity") or e.get("text") or ""


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


def jaccard(gt_set: set, pred_set: set) -> float:
    if not gt_set and not pred_set:
        return 1.0
    if not gt_set:
        return 0.0
    return len(gt_set & pred_set) / len(gt_set | pred_set)


def score_entities(pred_entities: list[dict], gt_entities: list[dict]) -> dict:
    """Compute final score = 0.3 * text + 0.3 * assertions + 0.4 * candidates."""

    if not pred_entities and not gt_entities:
        return {"score": 1.0, "text": 1.0, "assertions": 1.0, "candidates": 1.0}
    if not pred_entities:
        return {"score": 0.0, "text": 0.0, "assertions": 0.0, "candidates": 0.0}

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
            assertion_scores.append(0.0)
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

    return {"score": final, "text": text_score, "assertions": assertions_score, "candidates": candidates_score}


JSONL_PATH = os.path.abspath("./data_train/grpo_combined-100.jsonl")


@evaluation_test(
    input_dataset=[JSONL_PATH],
    completion_params=[{"model": "accounts/fireworks/models/qwen3-8b", "max_tokens": 1024, "temperature": 0.0}],
    max_dataset_rows=100,
    passed_threshold=0.5,
    rollout_processor=ROLLOUT_PROCESSOR,
    mode="pointwise",
)
def test_ner_accuracy(row: EvaluationRow, **kwargs) -> EvaluationRow:
    logger.info(f"Starting NER rollout: {row.execution_metadata.rollout_id}")

    assistant = row.messages[-1]
    content = assistant.get("content", "") if isinstance(assistant, dict) else (assistant.content or "")

    pred_entities = extract_entities(content)

    ground_truth = row.ground_truth
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            ground_truth = []

    result = score_entities(pred_entities, ground_truth)

    row.evaluation_result = EvaluateResult(
        score=result["score"],
        is_score_valid=True,
        reason=f"text={result['text']:.3f} assertions={result['assertions']:.3f} candidates={result['candidates']:.3f} | {len(pred_entities)} pred vs {len(ground_truth)} GT",
    )
    return row
