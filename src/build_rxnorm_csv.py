#!/usr/bin/env python3
"""
Pre-process RxNorm TTL into CSV for fast loading.

Parses database-stuff/RXNORM.ttl and exports to database-stuff/rxnorm_concepts.csv.

Usage:
    python3 src/build_rxnorm_csv.py
"""

import csv
import re
import sys
from pathlib import Path


def parse_rxnorm_to_csv(ttl_path: str, csv_path: str):
    """Parse RxNorm TTL and save to CSV."""
    concepts = {}
    current_cui = None

    print(f"Parsing {ttl_path}...")
    with open(ttl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()

            if "a owl:Class" in line:
                m = re.search(r"/RXNORM/(\d+)", line)
                if m:
                    current_cui = m.group(1)
                    if current_cui not in concepts:
                        concepts[current_cui] = {
                            "rxcui": current_cui,
                            "prefLabel": "",
                            "altLabels": [],
                            "ingredient_of": [],
                            "has_ingredient": [],
                            "tradename_of": [],
                            "has_tradename": [],
                            "has_dose_form": [],
                            "isa": [],
                            "ndc": "",
                            "strength": "",
                        }
            elif current_cui is not None:
                if "skos:prefLabel" in line:
                    m = re.search(r'"""(.+?)"""', line)
                    if m:
                        concepts[current_cui]["prefLabel"] = m.group(1)
                elif "skos:altLabel" in line:
                    m = re.search(r'"""(.+?)"""', line)
                    if m:
                        concepts[current_cui]["altLabels"].append(m.group(1))
                elif "ingredient_of" in line:
                    m = re.search(r"/RXNORM/(\d+)", line)
                    if m:
                        concepts[current_cui]["ingredient_of"].append(m.group(1))
                elif "has_ingredient" in line:
                    m = re.search(r"/RXNORM/(\d+)", line)
                    if m:
                        concepts[current_cui]["has_ingredient"].append(m.group(1))
                elif "tradename_of" in line:
                    m = re.search(r"/RXNORM/(\d+)", line)
                    if m:
                        concepts[current_cui]["tradename_of"].append(m.group(1))
                elif "has_tradename" in line:
                    m = re.search(r"/RXNORM/(\d+)", line)
                    if m:
                        concepts[current_cui]["has_tradename"].append(m.group(1))
                elif "has_dose_form" in line:
                    m = re.search(r"/RXNORM/(\d+)", line)
                    if m:
                        concepts[current_cui]["has_dose_form"].append(m.group(1))
                elif "isa" in line and "inverse_isa" not in line:
                    m = re.search(r"/RXNORM/(\d+)", line)
                    if m:
                        concepts[current_cui]["isa"].append(m.group(1))
                elif "NDC" in line and "skos:" not in line:
                    m = re.search(r'"""(.+?)"""', line)
                    if m:
                        concepts[current_cui]["ndc"] = m.group(1)
                elif "RXN_AVAILABLE_STRENGTH" in line:
                    m = re.search(r'"""(.+?)"""', line)
                    if m:
                        concepts[current_cui]["strength"] = m.group(1)
                elif line == ".":
                    current_cui = None

            if line_num % 100000 == 0 and line_num > 0:
                print(f"  Parsed {line_num} lines, {len(concepts)} concepts...", file=sys.stderr)

    print(f"Total: {len(concepts)} concepts")

    # Write CSV
    fieldnames = [
        "rxcui", "prefLabel", "altLabels",
        "ingredient_of", "has_ingredient",
        "tradename_of", "has_tradename",
        "has_dose_form", "isa", "ndc", "strength"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cui, data in concepts.items():
            row = {
                "rxcui": data["rxcui"],
                "prefLabel": data["prefLabel"],
                "altLabels": "|".join(data["altLabels"]),
                "ingredient_of": "|".join(data["ingredient_of"]),
                "has_ingredient": "|".join(data["has_ingredient"]),
                "tradename_of": "|".join(data["tradename_of"]),
                "has_tradename": "|".join(data["has_tradename"]),
                "has_dose_form": "|".join(data["has_dose_form"]),
                "isa": "|".join(data["isa"]),
                "ndc": data["ndc"],
                "strength": data["strength"],
            }
            writer.writerow(row)

    print(f"Saved to {csv_path}")


def main():
    base_dir = Path(__file__).parent.parent / "database-stuff"
    ttl_path = base_dir / "RXNORM.ttl"
    csv_path = base_dir / "rxnorm_concepts.csv"

    if not ttl_path.exists():
        print(f"Error: {ttl_path} not found", file=sys.stderr)
        sys.exit(1)

    parse_rxnorm_to_csv(str(ttl_path), str(csv_path))

    # Verify
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Verification: {len(rows)} rows in CSV")

    # Show sample
    print("\nSample entries:")
    for row in rows[:5]:
        print(f"  {row['rxcui']:10s} | {row['prefLabel'][:40]}")


if __name__ == "__main__":
    main()
