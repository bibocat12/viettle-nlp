#!/usr/bin/env python3
"""Strictly validate 100 prediction files and package output.zip."""

import argparse
import json
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tools import validate_output


def validate_submission(output_dir: Path, input_dir: Path) -> list[str]:
    errors = []
    expected = {f"{index}.json" for index in range(1, 101)}
    actual = {path.name for path in output_dir.glob("*.json")}
    for name in sorted(expected - actual, key=lambda value: int(Path(value).stem)):
        errors.append(f"missing {name}")
    for name in sorted(actual - expected):
        errors.append(f"unexpected JSON file {name}")

    for index in range(1, 101):
        prediction_path = output_dir / f"{index}.json"
        input_path = input_dir / f"{index}.txt"
        if not prediction_path.exists() or not input_path.exists():
            if not input_path.exists():
                errors.append(f"missing input {input_path.name}")
            continue
        try:
            predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{prediction_path.name}: invalid JSON ({exc})")
            continue
        if not isinstance(predictions, list):
            errors.append(f"{prediction_path.name}: top level must be a list")
            continue
        non_objects = [idx for idx, value in enumerate(predictions) if not isinstance(value, dict)]
        if non_objects:
            errors.append(
                f"{prediction_path.name}: entities must be objects (bad indexes: {non_objects[:10]})"
            )
            continue

        validation = validate_output(predictions)
        errors.extend(f"{prediction_path.name}: {message}" for message in validation["errors"])
        source = input_path.read_text(encoding="utf-8")
        for entity_index, prediction in enumerate(predictions):
            allowed = {"text", "type", "assertions", "position"}
            if prediction.get("type") in ("THUỐC", "CHẨN_ĐOÁN"):
                allowed.add("candidates")
            extra = set(prediction) - allowed
            if extra:
                errors.append(
                    f"{prediction_path.name}[{entity_index}]: extra fields {sorted(extra)}"
                )
            position = prediction.get("position")
            if (
                isinstance(position, list)
                and len(position) == 2
                and all(isinstance(value, int) for value in position)
            ):
                start, end = position
                if not (0 <= start < end <= len(source)):
                    errors.append(
                        f"{prediction_path.name}[{entity_index}]: position outside input"
                    )
                elif source[start:end] != prediction.get("text"):
                    errors.append(
                        f"{prediction_path.name}[{entity_index}]: text does not match position"
                    )
            candidates = prediction.get("candidates", [])
            if not isinstance(candidates, list) or len(candidates) > 1:
                errors.append(
                    f"{prediction_path.name}[{entity_index}]: candidates must contain at most one code"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "input")
    parser.add_argument("--zip", dest="zip_path", type=Path, default=ROOT / "output.zip")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    errors = validate_submission(args.output_dir, args.input_dir)
    if errors:
        print(f"Submission invalid: {len(errors)} error(s)", file=sys.stderr)
        for error in errors[:100]:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Submission valid: 100 JSON files, offsets and schema verified")
    if args.check_only:
        return 0

    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(1, 101):
            source = args.output_dir / f"{index}.json"
            archive.write(source, arcname=f"output/{index}.json")
    print(f"Wrote {args.zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
