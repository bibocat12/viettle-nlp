#!/usr/bin/env python3
"""
Extract ICD-10 codes with Vietnamese and English names from the MoH PDF.

Parses database-stuff/06-byt-kem.pdf (Vietnamese MoH ICD-10 code list)
and exports to database-stuff/icd10_vietnamese.csv.

Usage:
    python3 src/extract_icd10_pdf.py [--method pymupdf|pdftotext]
"""

import csv
import re
import subprocess
import sys
from pathlib import Path

VN_CHARS = set(
    "àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉị"
    "òóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳýỹỷỵ"
    "ÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊ"
    "ÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲÝỸỶỴ"
)


def has_vietnamese(text: str) -> bool:
    return any(c in VN_CHARS for c in text)


def extract_pymupdf(pdf_path: str) -> list:
    """Extract using PyMuPDF block positions for accurate column matching."""
    import pymupdf

    doc = pymupdf.open(pdf_path)
    entries = []
    seen_codes = set()

    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")

        # Find code blocks (contain ICD-10 code pattern)
        code_blocks = []
        vn_blocks = []

        for bi, b in enumerate(blocks):
            if len(b) < 5:
                continue
            x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], str(b[4]).strip()
            if not text:
                continue

            # Code block: contains code like "I10\nI10\nEssential..." or "I10.0\nI110\n..."
            # Match both 3-char (I10) and 4+ char (I10.0) codes
            code_match = re.search(r"\b([A-Z]\d{2}(?:\.\d{1,2})?)\b", text)
            if code_match and x0 > 400:  # Code blocks are at x~473
                code = code_match.group(1)
                # For 3-char codes, check if block has pattern "CODE\nCODE\n..."
                if len(code) == 3:
                    # 3-char code blocks have format "I10\nI10\nEnglish name"
                    if not re.search(r"\b" + code + r"\s*\n\s*" + code + r"\b", text):
                        continue
                code_blocks.append({
                    "bi": bi, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "code": code, "text": text
                })

            # Vietnamese block: at x~591, contains Vietnamese text
            if x0 > 550 and has_vietnamese(text):
                vn_blocks.append({
                    "bi": bi, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "text": text
                })

        # Match code blocks to Vietnamese blocks by y-position (same row)
        for cb in code_blocks:
            code = cb["code"]
            if code in seen_codes:
                continue

            # Find Vietnamese block at same y-position (within 20 units)
            best_vn = None
            best_dist = 999
            for vb in vn_blocks:
                y_dist = abs(vb["y0"] - cb["y0"])
                if y_dist < 20 and y_dist < best_dist:
                    best_dist = y_dist
                    best_vn = vb

            if best_vn is None:
                continue

            # Extract Vietnamese name from the block
            vn_text = best_vn["text"].strip()
            # Remove trailing code references
            vn_text = re.sub(r"\s*\([A-Z]\d{2}(?:\.\d{1,2})?[\.\-†]*\)\s*$", "", vn_text).strip()
            vn_text = re.sub(r"\s+[A-Z]\d{2}(?:\.\d{1,2})?\s*$", "", vn_text).strip()
            vn_text = re.sub(r"\s+", " ", vn_text).strip()

            # Extract English name from code block
            en_text = cb["text"]
            # Remove the code lines
            en_text = re.sub(r"^[A-Z]\d{2}\.\d{1,2}\s*\n\s*[A-Z]\d{2,4}\s*\n", "", en_text).strip()
            en_text = re.sub(r"\s+", " ", en_text).strip()

            if len(vn_text) < 3:
                continue

            entries.append({
                "icd10_code": code,
                "icd10_code_no_dot": code.replace(".", ""),
                "english_name": en_text,
                "vietnamese_name": vn_text,
            })
            seen_codes.add(code)

        if page_num % 100 == 0 and page_num > 0:
            print(f"  Processed {page_num} pages...", file=sys.stderr)

    return entries


def extract_pdftotext(pdf_path: str) -> list:
    """Extract using pdftotext with regex-based column detection."""
    result = subprocess.run(
        ["pdftotext", "-layout", pdf_path, "-"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error: pdftotext failed", file=sys.stderr)
        return []

    lines = result.stdout.split("\n")
    entries = []
    seen_codes = set()

    for line in lines:
        if len(line) < 200:
            continue

        m = re.search(r"([A-Z]\d{2}\.\d{1,2})", line)
        if not m:
            continue

        code = m.group(1)
        if code in seen_codes:
            continue

        rest = line[m.end():]
        vn_start = None
        for i, ch in enumerate(rest):
            if ch in VN_CHARS:
                vn_start = i
                break

        if vn_start is None:
            continue

        en_name = rest[:vn_start].strip()
        vn_name = rest[vn_start:].strip()

        # Find where VN name ends (next code or end)
        vn_end = re.search(r"\s+[A-Z]\d{2}(?:\.\d{1,2})?[\.\-†]*\s*$", vn_name)
        if vn_end:
            vn_name = vn_name[:vn_end.start()].strip()

        vn_name = re.sub(r"\s+", " ", vn_name).strip()
        en_name = re.sub(r"\s+", " ", en_name).strip()
        vn_name = re.sub(r"\s*\([A-Z]\d{2}(?:\.\d{1,2})?[\.\-†]*\)\s*$", "", vn_name).strip()

        if len(vn_name) < 3:
            continue

        entries.append({
            "icd10_code": code,
            "icd10_code_no_dot": code.replace(".", ""),
            "english_name": en_name,
            "vietnamese_name": vn_name,
        })
        seen_codes.add(code)

    return entries


def save_csv(entries: list, output_path: str):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "icd10_code", "icd10_code_no_dot", "english_name", "vietnamese_name"
        ])
        writer.writeheader()
        writer.writerows(entries)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["pymupdf", "pdftotext"], default="pymupdf")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    pdf_path = base_dir / "database-stuff" / "06-byt-kem.pdf"
    csv_path = base_dir / "database-stuff" / "icd10_vietnamese.csv"

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting ICD-10 codes from: {pdf_path}")
    print(f"Method: {args.method}")

    if args.method == "pymupdf":
        entries = extract_pymupdf(str(pdf_path))
    else:
        entries = extract_pdftotext(str(pdf_path))

    print(f"Extracted {len(entries)} unique ICD-10 codes")
    save_csv(entries, str(csv_path))
    print(f"Saved to: {csv_path}")

    print("\nSample entries:")
    for e in entries[:15]:
        print(f"  {e['icd10_code']:10s} | {e['vietnamese_name'][:60]}")

    print("\nSpecific codes:")
    for code in ["I10", "J18", "E11", "N18", "J43", "C34.9", "B67.3", "A00.0"]:
        found = [e for e in entries if e["icd10_code"] == code]
        if found:
            print(f"  {code}: {found[0]['vietnamese_name'][:60]}")
        else:
            print(f"  {code}: NOT FOUND")

    codes_with_vn = sum(1 for e in entries if e["vietnamese_name"])
    codes_with_en = sum(1 for e in entries if e["english_name"])
    print(f"\nStats:")
    print(f"  Total codes: {len(entries)}")
    print(f"  With Vietnamese name: {codes_with_vn}")
    print(f"  With English name: {codes_with_en}")


if __name__ == "__main__":
    main()
