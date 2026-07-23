#!/usr/bin/env python3
"""Repair high-confidence assertion/type/linking errors in round-2 output."""

import argparse
import json
import re
import unicodedata
from pathlib import Path


ICD_REPAIRS = {
    "benh dai": "A82.9",
    "may day vo can": "L50.1",
    "may day man": "L50.8",
    "may day man tinh": "L50.8",
    "toan trang suy kiet": "R64",
    "u xo tuyen vu": "D24",
    "u nang tuyen vu": "N60.0",
    "ung thu": "C80.1",
    "tieu duong": "E11.9",
    "tien san giat": "O14.9",
    "gian thung tinh": "I86.1",
    "o loet trong bao tu": "K25.9",
    "viem hang vi sung huyet": "K29.7",
    "viem sung huyet hang vi da day": "K29.7",
    "viem bao tu": "K29.7",
    "ung thu mau": "C95.9",
    "tu dich/tu mau duoi mang cung man tinh": "I62.0",
    "nang mang nhen": "G93.0",
    "nhiem trung chi duoi": "L08.9",
    "benh gout": "M10.9",
    "benh gut": "M10.9",
    "nao ung tuy": "G91.9",
    "roi loan chuyen hoa tinh bot (amyloidosis)": "E85",
    "amyloidosis chuoi nhe": "E85.8",
    "benh daithuong": "A82.9",
    "beo phi": "E66.9",
    "nhiem khuan duong tiet nieu": "N39.0",
    "nhiem trung duong tiet nieu": "N39.0",
    "tram cam": "F32.9",
    "mun trung ca": "L70.9",
    "huyet khoi": "I82.9",
}

TEST_NAMES = {
    "bach cau", "hong cau", "tieu cau", "hemoglobin", "hematocrit",
    "bilirubin", "creatinine", "troponin", "duong huyet",
}

TYPE_REPAIRS = {
    "toan trang suy kiet": "TRIỆU_CHỨNG",
    "suy kiet": "TRIỆU_CHỨNG",
    "suy nhuoc": "TRIỆU_CHỨNG",
    "beo phi": "CHẨN_ĐOÁN",
    "nhiem khuan duong tiet nieu": "CHẨN_ĐOÁN",
    "nhiem trung duong tiet nieu": "CHẨN_ĐOÁN",
    "tram cam": "CHẨN_ĐOÁN",
    "mun trung ca": "CHẨN_ĐOÁN",
    "huyet khoi": "CHẨN_ĐOÁN",
    "corticoid lieu cao keo dai": "THUỐC",
    "corticoid lieu thap": "THUỐC",
    "thinh luc": "TÊN_XÉT_NGHIỆM",
}


def key(value: str) -> str:
    value = unicodedata.normalize("NFD", value.casefold())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", value).strip(" .,:;()")


def clause_prefix(raw: str, start: int, width: int = 160) -> str:
    prefix = raw[max(0, start - width):start]
    normalized = unicodedata.normalize("NFC", prefix).casefold()
    boundary = max(normalized.rfind(token) for token in ("\n", ".", ";", "!", "?"))
    if boundary >= 0:
        normalized = normalized[boundary + 1:]
    contrast = list(re.finditer(r"\b(?:nhưng|tuy nhiên|song|mặc dù)\b", normalized))
    if contrast:
        normalized = normalized[contrast[-1].end():]
    return normalized


def strict_assertions(raw: str, entity: dict) -> list[str]:
    if entity["type"] not in {"THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN"}:
        return []

    start = entity["position"][0]
    prefix = clause_prefix(raw, start)
    labels: list[str] = []

    if re.search(
        r"(?:\bkhông(?:\s+có|\s+ghi nhận|\s+còn)?|\bchưa(?:\s+có)?|"
        r"\bphủ nhận|\bâm tính(?:\s+với)?)\b.{0,70}$",
        prefix,
        flags=re.DOTALL,
    ):
        labels.append("isNegated")

    family_context = re.search(r"\btiền sử gia đình\b.{0,120}$", prefix, re.DOTALL)
    relative_context = re.search(
        r"\b(?:bố|mẹ|cha|chị gái|anh trai|ông|bà|con)\b.{0,45}"
        r"\b(?:bị|mắc|có|được chẩn đoán)\b.{0,45}$",
        prefix,
        flags=re.DOTALL,
    )
    if family_context or relative_context:
        labels.append("isFamily")

    before = unicodedata.normalize("NFC", raw[max(0, start - 1800):start]).casefold()
    historical_headers = (
        "tiền sử bệnh nội khoa", "tiền sử bệnh\n", "tiền sử:",
        "các bệnh mãn tính", "thuốc trước nhập viện", "thuốc trước khi nhập viện",
    )
    current_headers = (
        "tiền sử bệnh hiện tại", "bệnh sử hiện tại", "lý do nhập viện",
        "triệu chứng hiện tại", "đánh giá tại bệnh viện", "tình trạng ngay trước khi nhập viện",
    )
    last_historical = max((before.rfind(marker) for marker in historical_headers), default=-1)
    last_current = max((before.rfind(marker) for marker in current_headers), default=-1)
    explicit_history = re.search(
        r"\b(?:tiền sử|trước đây|đã từng|đã dùng|trước nhập viện|trước khi nhập viện)\b.{0,100}$",
        prefix,
        flags=re.DOTALL,
    )
    is_indication = entity["type"] == "TRIỆU_CHỨNG" and re.search(
        r"\b(?:điều trị|do|vì)\b.{0,45}$", prefix, flags=re.DOTALL
    )
    if not is_indication and (explicit_history or last_historical > last_current):
        labels.append("isHistorical")

    return labels


def process(prediction_path: Path, input_path: Path, destination: Path) -> dict[str, int]:
    raw = input_path.read_text(encoding="utf-8")
    entities = json.loads(prediction_path.read_text(encoding="utf-8"))
    stats = {"candidate_repairs": 0, "type_repairs": 0, "assertion_changes": 0}

    for entity in entities:
        normalized = key(entity["text"])
        repaired_type = TYPE_REPAIRS.get(normalized)
        if repaired_type is None and entity["type"] == "TRIỆU_CHỨNG" and normalized in TEST_NAMES:
            repaired_type = "TÊN_XÉT_NGHIỆM"
        if repaired_type and entity["type"] != repaired_type:
            entity["type"] = repaired_type
            entity.pop("candidates", None)
            stats["type_repairs"] += 1

        repaired_code = ICD_REPAIRS.get(normalized)
        if entity["type"] == "CHẨN_ĐOÁN" and repaired_code:
            if entity.get("candidates") != [repaired_code]:
                entity["candidates"] = [repaired_code]
                stats["candidate_repairs"] += 1

        assertions = strict_assertions(raw, entity)
        if entity.get("assertions", []) != assertions:
            entity["assertions"] = assertions
            stats["assertion_changes"] += 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(entities, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    totals = {"candidate_repairs": 0, "type_repairs": 0, "assertion_changes": 0}
    for index in range(1, 101):
        stats = process(
            args.predictions / f"{index}.json",
            args.inputs / f"{index}.txt",
            args.output / f"{index}.json",
        )
        for name, value in stats.items():
            totals[name] += value
    print(json.dumps(totals, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
