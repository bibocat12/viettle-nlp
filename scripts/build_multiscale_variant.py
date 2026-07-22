#!/usr/bin/env python3
"""Build a conservative submission from multiscale recall proposals.

Only non-overlapping diagnosis/drug mentions with an exact reviewed ICD-10 or
RxNorm alias are added to the scored base. No record IDs or note text are
hard-coded, so the same gate can be rerun on private test notes.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def normalize(value: str) -> str:
    return " ".join(value.casefold().split()).strip(" .,:;()")


def load_aliases(filenames: tuple[str, ...]) -> dict[str, str]:
    aliases = {}
    for filename in filenames:
        values = json.loads(
            (ROOT / "database-stuff" / filename).read_text(encoding="utf-8")
        )
        aliases.update({normalize(key): str(code).upper() for key, code in values.items()})
    return aliases


def load_base(path: Path) -> dict[int, list[dict]]:
    with zipfile.ZipFile(path) as archive:
        return {
            index: json.loads(archive.read(f"output/{index}.json"))
            for index in range(1, 101)
        }


def add_reviewed_proposals(
    records: dict[int, list[dict]], proposal_dir: Path
) -> list[tuple[int, dict]]:
    icd = load_aliases(("icd10_aliases.json", "icd10_aliases_round2.json"))
    rxnorm = load_aliases(("rxnorm_aliases.json", "rxnorm_aliases_round2.json"))
    additions = []
    for index in range(1, 101):
        proposals = json.loads((proposal_dir / f"{index}.json").read_text(encoding="utf-8"))
        for proposal in proposals:
            mention_type = proposal.get("type")
            aliases = icd if mention_type == "CHẨN_ĐOÁN" else rxnorm if mention_type == "THUỐC" else {}
            code = aliases.get(normalize(proposal.get("text", "")))
            if not code:
                continue
            start, end = proposal.get("position", [0, 0])
            if any(
                min(end, entity["position"][1]) > max(start, entity["position"][0])
                for entity in records[index]
            ):
                continue
            added = {
                "text": proposal["text"],
                "type": mention_type,
                "assertions": list(proposal.get("assertions", [])),
                "position": [start, end],
                "candidates": [code],
            }
            records[index].append(added)
            additions.append((index, added))
        records[index].sort(key=lambda item: (item["position"][0], item["position"][1]))
    return additions


def validate(records: dict[int, list[dict]]) -> None:
    for index, entities in records.items():
        source = (ROOT / "input" / f"{index}.txt").read_text(encoding="utf-8")
        for entity in entities:
            start, end = entity["position"]
            if source[start:end] != entity["text"]:
                raise ValueError(f"{index}.json: text-position mismatch for {entity!r}")
            if len(entity.get("candidates", [])) > 1:
                raise ValueError(f"{index}.json: multiple candidates for {entity!r}")


def write_submission(records: dict[int, list[dict]], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    # Use real files + archive.write, matching the packaging metadata of the
    # previously accepted lb15 archive. Some competition extractors mishandle
    # writestr entries whose external attributes do not mark a regular file.
    with tempfile.TemporaryDirectory(prefix="viettle_submission_") as temp_dir:
        staging = Path(temp_dir) / "output"
        staging.mkdir()
        for index in range(1, 101):
            (staging / f"{index}.json").write_text(
                json.dumps(records[index], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index in range(1, 101):
                archive.write(staging / f"{index}.json", arcname=f"output/{index}.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = load_base(args.base)
    additions = add_reviewed_proposals(records, args.proposals)
    validate(records)
    write_submission(records, args.output)
    print(f"Added {len(additions)} reviewed multiscale mentions")
    for index, entity in additions:
        print(index, json.dumps(entity, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
