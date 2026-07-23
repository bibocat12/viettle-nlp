#!/usr/bin/env python3
"""Map prediction offsets from NFC-normalized text back to raw input text."""

import argparse
import json
import unicodedata
from pathlib import Path


def normalized_text_and_boundaries(raw: str) -> tuple[str, dict[int, int]]:
    """Return pipeline-normalized text and normalized-to-raw cluster boundaries."""
    chunks: list[str] = []
    boundaries: dict[int, int] = {0: 0}
    normalized_offset = 0
    raw_offset = 0

    while raw_offset < len(raw):
        cluster_end = raw_offset + 1
        while cluster_end < len(raw) and unicodedata.combining(raw[cluster_end]):
            cluster_end += 1

        raw_cluster = raw[raw_offset:cluster_end]
        normalized_cluster = unicodedata.normalize("NFC", raw_cluster).replace("\u00a0", " ")
        chunks.append(normalized_cluster)

        # Ordinary characters retain all internal boundaries. A decomposed
        # grapheme only exposes its outer boundaries, which is exactly where
        # entity spans may begin/end.
        if len(normalized_cluster) == len(raw_cluster):
            for delta in range(1, len(normalized_cluster) + 1):
                boundaries[normalized_offset + delta] = raw_offset + delta
        else:
            boundaries[normalized_offset + len(normalized_cluster)] = cluster_end

        normalized_offset += len(normalized_cluster)
        raw_offset = cluster_end

    return "".join(chunks), boundaries


def nearest_occurrence(text: str, needle: str, anchor: int) -> tuple[int, int] | None:
    starts: list[int] = []
    offset = 0
    while True:
        position = text.find(needle, offset)
        if position < 0:
            break
        starts.append(position)
        offset = position + 1
    if not starts:
        return None
    start = min(starts, key=lambda value: abs(value - anchor))
    return start, start + len(needle)


def remap_file(prediction_path: Path, input_path: Path, destination: Path) -> int:
    raw = input_path.read_text(encoding="utf-8")
    normalized, boundaries = normalized_text_and_boundaries(raw)
    expected = unicodedata.normalize("NFC", raw).replace("\u00a0", " ")
    if normalized != expected:
        raise ValueError(f"normalization map mismatch for {input_path.name}")

    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    changed = 0
    for index, prediction in enumerate(predictions):
        start, end = prediction["position"]
        entity = prediction["text"]
        if normalized[start:end] != entity:
            relocated = nearest_occurrence(normalized, entity, start)
            if relocated is None:
                raise ValueError(
                    f"{prediction_path.name}[{index}]: entity absent from normalized input: {entity!r}"
                )
            start, end = relocated

        if start not in boundaries or end not in boundaries:
            raise ValueError(
                f"{prediction_path.name}[{index}]: span splits a Unicode grapheme ({start}, {end})"
            )
        raw_start, raw_end = boundaries[start], boundaries[end]
        raw_entity = raw[raw_start:raw_end]
        if [raw_start, raw_end] != prediction["position"] or raw_entity != entity:
            changed += 1
        prediction["position"] = [raw_start, raw_end]
        prediction["text"] = raw_entity

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    total_changed = 0
    for index in range(1, 101):
        total_changed += remap_file(
            args.predictions / f"{index}.json",
            args.inputs / f"{index}.txt",
            args.output / f"{index}.json",
        )
    print(f"Remapped 100 files; updated {total_changed} entity spans")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
