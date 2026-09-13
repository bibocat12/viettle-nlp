#!/usr/bin/env python3
"""Add high-precision exact mentions learned from labeled GRPO-v2 data.

For every labeled surface, empirical precision is measured as:

    labeled exact mentions / all word-bounded occurrences in extraction_train

An Aho-Corasick automaton makes the denominator a true corpus scan instead of
assuming that every occurrence of a medical-looking word is an annotation.
This is particularly important for short Vietnamese words such as ``đau`` and
``phù``.  The scored submission is used only as an immutable anchor.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from build_submission import validate_submission
from grpo_v2_fusion import _strict_assertions
from reward_v2 import score_entities
from run_score40_consensus import _aliases


VALID_TYPES = {
    "THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN",
    "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM",
}
LINKABLE = {"THUỐC", "CHẨN_ĐOÁN"}
HARD_REJECT = {
    "chó cắn", "điều trị", "phẫu thuật", "bệnh này", "viêm",
    "rượu", "bia", "nước có ga", "dịch sinh học", "ăn ngủ được",
}


def _normalise(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split()).strip()


def _word_character(value: str) -> bool:
    return bool(value) and (value.isalnum() or value == "_" or unicodedata.category(value).startswith("M"))


def _bounded(text: str, start: int, end: int) -> bool:
    return (
        (start == 0 or not _word_character(text[start - 1]))
        and (end == len(text) or not _word_character(text[end]))
    )


class Automaton:
    """Small dependency-free Aho-Corasick implementation."""

    def __init__(self, terms: list[str]):
        self.terms = terms
        self.next: list[dict[str, int]] = [{}]
        self.fail = [0]
        self.output: list[list[int]] = [[]]
        for term_id, term in enumerate(terms):
            state = 0
            for character in term:
                child = self.next[state].get(character)
                if child is None:
                    child = len(self.next)
                    self.next[state][character] = child
                    self.next.append({})
                    self.fail.append(0)
                    self.output.append([])
                state = child
            self.output[state].append(term_id)
        queue: deque[int] = deque()
        for child in self.next[0].values():
            queue.append(child)
        while queue:
            state = queue.popleft()
            for character, child in self.next[state].items():
                queue.append(child)
                fallback = self.fail[state]
                while fallback and character not in self.next[fallback]:
                    fallback = self.fail[fallback]
                self.fail[child] = self.next[fallback].get(character, 0)
                self.output[child].extend(self.output[self.fail[child]])

    def matches(self, value: str):
        state = 0
        for end_index, character in enumerate(value):
            while state and character not in self.next[state]:
                state = self.fail[state]
            state = self.next[state].get(character, 0)
            for term_id in self.output[state]:
                start = end_index + 1 - len(self.terms[term_id])
                end = end_index + 1
                if _bounded(value, start, end):
                    yield start, end, term_id


def _text(row: dict) -> str:
    for message in row.get("messages", []):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return str(row.get("text", ""))


def _surface(entity: dict) -> str:
    return _normalise(str(entity.get("entity") or entity.get("text") or ""))


def _read_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _candidate_votes(paths: list[Path]) -> dict[tuple[str, str], tuple[str, float, int]]:
    votes: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for path in paths:
        if not path.exists():
            continue
        for row in _read_rows(path):
            for entity in row.get("entities", []):
                mention_type = entity.get("type")
                codes = list(map(str, entity.get("candidates", [])))
                if (
                    mention_type not in LINKABLE
                    or entity.get("_candidate_supervised") is False
                    or len(codes) != 1
                ):
                    continue
                surface = _surface(entity)
                if surface:
                    votes[(surface, mention_type)][codes[0]] += 1
    result = {}
    for key, counter in votes.items():
        code, count = counter.most_common(1)[0]
        total = sum(counter.values())
        result[key] = (code, count / total, count)
    return result


def build_lexicon(
    train_path: Path,
    candidate_paths: list[Path],
    threshold: float,
    minimum_labels: int,
) -> tuple[dict[str, dict], dict]:
    rows = list(_read_rows(train_path))
    label_types: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for entity in row.get("entities", []):
            surface = _surface(entity)
            mention_type = str(entity.get("type", ""))
            if surface and mention_type in VALID_TYPES:
                label_types[surface][mention_type] += 1

    terms = sorted(label_types)
    automaton = Automaton(terms)
    occurrences = Counter()
    for row in rows:
        # NFC keeps the automaton consistent with normalized labels.  It is
        # used only for corpus counts, not target offsets.
        value = unicodedata.normalize("NFC", _text(row)).casefold()
        for _, _, term_id in automaton.matches(value):
            occurrences[terms[term_id]] += 1

    candidate_map = _candidate_votes(candidate_paths)
    lexicon = {}
    rejected = Counter()
    for surface, counter in label_types.items():
        mention_type, type_count = counter.most_common(1)[0]
        total_labels = sum(counter.values())
        bounded_occurrences = occurrences[surface]
        precision = type_count / bounded_occurrences if bounded_occurrences else 0.0
        purity = type_count / total_labels
        reason = None
        if surface in HARD_REJECT:
            reason = "hard_reject"
        elif len(surface) < 3 or not any(character.isalpha() for character in surface):
            reason = "too_short"
        elif type_count < minimum_labels:
            reason = "label_count"
        elif purity < 0.80:
            reason = "type_purity"
        elif precision < threshold:
            reason = "empirical_precision"
        if reason:
            rejected[reason] += 1
            continue
        code = None
        candidate = candidate_map.get((surface, mention_type))
        if candidate and candidate[1] >= 0.90 and candidate[2] >= 2:
            code = candidate[0]
        lexicon[surface] = {
            "type": mention_type,
            "labeled_count": type_count,
            "bounded_occurrences": bounded_occurrences,
            "precision": precision,
            "purity": purity,
            "candidate": code,
        }
    report = {
        "labeled_surfaces": len(label_types),
        "accepted_surfaces": len(lexicon),
        "rejected": dict(rejected),
    }
    return lexicon, report


def _scan(text: str, automaton: Automaton, lexicon: dict[str, dict]):
    # Vietnamese case folding preserves code-point offsets.  Do not NFC the
    # target here: some source notes intentionally contain decomposed accents.
    value = text.casefold()
    for start, end, term_id in automaton.matches(value):
        surface = automaton.terms[term_id]
        metadata = lexicon[surface]
        yield start, end, metadata


def _make_entity(text: str, start: int, end: int, metadata: dict,
                 icd: dict[str, str], rxnorm: dict[str, str]) -> dict:
    mention_type = metadata["type"]
    result = {
        "text": text[start:end],
        "type": mention_type,
        "assertions": [],
        "position": [start, end],
    }
    result["assertions"] = _strict_assertions(text, result)
    if mention_type in LINKABLE:
        aliases = icd if mention_type == "CHẨN_ĐOÁN" else rxnorm
        surface = _normalise(result["text"])
        code = aliases.get(surface) or metadata.get("candidate")
        result["candidates"] = [code] if code else []
    return result


def evaluate_split(path: Path, lexicon: dict[str, dict], first_only: bool) -> dict:
    terms = sorted(lexicon)
    automaton = Automaton(terms)
    icd, rxnorm = _aliases()
    totals = Counter()
    metrics = Counter()
    records = 0
    for row in _read_rows(path):
        text = _text(row)
        predictions = []
        seen_surface_type = set()
        for start, end, metadata in _scan(text, automaton, lexicon):
            key = (_normalise(text[start:end]), metadata["type"])
            if first_only and key in seen_surface_type:
                continue
            seen_surface_type.add(key)
            predictions.append(_make_entity(text, start, end, metadata, icd, rxnorm))
        truth = row.get("entities", [])
        truth_keys = {
            (tuple(e.get("position", [])), e.get("type"))
            for e in truth if len(e.get("position", [])) == 2
        }
        pred_keys = {(tuple(e["position"]), e["type"]) for e in predictions}
        totals["tp"] += len(truth_keys & pred_keys)
        totals["fp"] += len(pred_keys - truth_keys)
        totals["fn"] += len(truth_keys - pred_keys)
        score = score_entities(predictions, truth)
        for key in ("score", "text", "assertions", "candidates"):
            metrics[key] += score[key]
        records += 1
    precision = totals["tp"] / max(1, totals["tp"] + totals["fp"])
    recall = totals["tp"] / max(1, totals["tp"] + totals["fn"])
    return {
        "records": records,
        **dict(totals),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "mean_metrics": {key: metrics[key] / max(records, 1) for key in metrics},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--train", type=Path,
        default=ROOT / "data_train/grpo_v2/extraction_train.jsonl",
    )
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--minimum-labels", type=int, default=1)
    parser.add_argument("--first-only", action="store_true")
    parser.add_argument(
        "--lexicon-only", action="store_true",
        help="build an independent prediction instead of unioning with --source",
    )
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    candidate_paths = [
        ROOT / "data_train/grpo_v2/full_train.jsonl",
        ROOT / "data_train/grpo_v2/linking_train.jsonl",
    ]
    lexicon, report = build_lexicon(
        args.train, candidate_paths, args.threshold, args.minimum_labels
    )
    if args.evaluate:
        for split in ("validation", "test"):
            path = ROOT / f"data_train/grpo_v2/extraction_{split}.jsonl"
            report[f"{split}_all"] = evaluate_split(path, lexicon, False)
            report[f"{split}_first"] = evaluate_split(path, lexicon, True)

    terms = sorted(lexicon)
    automaton = Automaton(terms)
    icd, rxnorm = _aliases()
    args.output.mkdir(parents=True, exist_ok=True)
    additions = Counter()
    audit = []
    for record in range(1, 101):
        text = (args.inputs / f"{record}.txt").read_text(encoding="utf-8")
        entities = [] if args.lexicon_only else json.loads(
            (args.source / f"{record}.json").read_text(encoding="utf-8")
        )
        exact = {
            (e["position"][0], e["position"][1], e["type"])
            for e in entities
        }
        seen_surface_type = {
            (_normalise(e["text"]), e["type"]) for e in entities
        }
        record_additions = []
        for start, end, metadata in _scan(text, automaton, lexicon):
            exact_key = (start, end, metadata["type"])
            surface_key = (_normalise(text[start:end]), metadata["type"])
            if exact_key in exact or (args.first_only and surface_key in seen_surface_type):
                continue
            entity = _make_entity(text, start, end, metadata, icd, rxnorm)
            entities.append(entity)
            exact.add(exact_key)
            seen_surface_type.add(surface_key)
            record_additions.append(entity)
            additions[metadata["type"]] += 1
            if entity.get("candidates"):
                additions["coded"] += 1
        entities.sort(key=lambda e: (e["position"][0], e["position"][1], e["type"]))
        (args.output / f"{record}.json").write_text(
            json.dumps(entities, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if record_additions:
            audit.append({"record": record, "additions": record_additions})

    errors = validate_submission(args.output, args.inputs)
    if errors:
        raise RuntimeError("; ".join(errors[:10]))
    run_dir = args.output.parent
    (run_dir / "lexicon_report.json").write_text(
        json.dumps({**report, "target_additions": dict(additions)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "audit.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit),
        encoding="utf-8",
    )
    zip_path = run_dir / "output.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for record in range(1, 101):
            archive.write(args.output / f"{record}.json", f"output/{record}.json")
    print(json.dumps({**report, "target_additions": dict(additions), "zip": str(zip_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
