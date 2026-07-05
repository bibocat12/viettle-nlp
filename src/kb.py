import csv
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Optional


def load_rxnorm_csv(csv_path: str) -> dict:
    """Load pre-processed RxNorm concepts from CSV."""
    concepts = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cui = row["rxcui"]
            concepts[cui] = {
                "rxcui": cui,
                "prefLabel": row["prefLabel"],
                "altLabels": [x for x in row["altLabels"].split("|") if x],
                "ingredient_of": [x for x in row["ingredient_of"].split("|") if x],
                "has_ingredient": [x for x in row["has_ingredient"].split("|") if x],
                "tradename_of": [x for x in row["tradename_of"].split("|") if x],
                "has_tradename": [x for x in row["has_tradename"].split("|") if x],
                "has_dose_form": [x for x in row["has_dose_form"].split("|") if x],
                "isa": [x for x in row["isa"].split("|") if x],
                "ndc": row["ndc"],
                "strength": row["strength"],
            }
    return concepts


def build_rxnorm_index(concepts: dict) -> dict:
    """Build name-to-CUI index for RxNorm lookup."""
    name_to_cui = defaultdict(list)
    for cui, c in concepts.items():
        label = c["prefLabel"].lower().strip()
        if label:
            name_to_cui[label].append(cui)
        for alt in c["altLabels"]:
            if alt:
                name_to_cui[alt.lower().strip()].append(cui)
    return dict(name_to_cui)


def load_icd10_csv(csv_path: str) -> dict:
    """Load ICD-10 codes with Vietnamese names from CSV."""
    icd10 = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get("icd10_code", "").strip()
            if not code:
                continue
            vn = row.get("vietnamese_name", "").strip()
            en = row.get("english_name", "").strip()
            code_no_dot = row.get("icd10_code_no_dot", "").strip()
            if code not in icd10:
                icd10[code] = {
                    "code": code,
                    "code_no_dot": code_no_dot,
                    "english_name": en,
                    "vietnamese_name": vn,
                    "billable": True,
                    "synonyms": [],
                }
            elif vn and not icd10[code]["vietnamese_name"]:
                icd10[code]["vietnamese_name"] = vn
    return icd10


def build_icd10_index(icd10: dict) -> dict:
    """Build name-to-code index for ICD-10 lookup."""
    name_to_code = defaultdict(list)
    for code, info in icd10.items():
        en = info["english_name"].lower().strip()
        if en:
            name_to_code[en].append(code)
        vn = info["vietnamese_name"].lower().strip()
        if vn:
            name_to_code[vn].append(code)
        for syn in info["synonyms"]:
            if syn:
                name_to_code[syn.lower().strip()].append(code)
    return dict(name_to_code)


class KnowledgeBase:
    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            base_dir = str(Path(__file__).parent.parent / "database-stuff")
        self.base_dir = Path(base_dir)
        self.rxnorm_concepts = {}
        self.rxnorm_name_index = {}
        self.icd10 = {}
        self.icd10_name_index = {}
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        # Load RxNorm from pre-processed CSV
        rxnorm_csv = self.base_dir / "rxnorm_concepts.csv"
        if rxnorm_csv.exists():
            self.rxnorm_concepts = load_rxnorm_csv(str(rxnorm_csv))
            self.rxnorm_name_index = build_rxnorm_index(self.rxnorm_concepts)
        else:
            print(f"Warning: {rxnorm_csv} not found. Run build_rxnorm_csv.py first.", file=__import__('sys').stderr)

        # Load ICD-10 from pre-processed CSV
        icd10_csv = self.base_dir / "icd10_vietnamese.csv"
        if icd10_csv.exists():
            self.icd10 = load_icd10_csv(str(icd10_csv))
            self.icd10_name_index = build_icd10_index(self.icd10)
        else:
            print(f"Warning: {icd10_csv} not found. Run extract_icd10_pdf.py first.", file=__import__('sys').stderr)

        self._loaded = True

    def search_icd10(self, query: str, top_k: int = 10) -> list:
        q = query.lower().strip()
        results = []
        if q in self.icd10_name_index:
            for code in self.icd10_name_index[q]:
                results.append({"code": code, "score": 1.0, "name": self.icd10[code].get("vietnamese_name") or self.icd10[code].get("english_name", "")})
        for name, codes in self.icd10_name_index.items():
            if q in name or name in q:
                for code in codes:
                    if not any(r["code"] == code for r in results):
                        ratio = len(q) / max(len(name), 1)
                        results.append({"code": code, "score": min(ratio, 0.9), "name": self.icd10[code].get("vietnamese_name") or self.icd10[code].get("english_name", "")})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def search_rxnorm(self, query: str, top_k: int = 10) -> list:
        q = query.lower().strip()
        results = []
        if q in self.rxnorm_name_index:
            for cui in self.rxnorm_name_index[q]:
                results.append({"rxcui": cui, "score": 1.0, "name": self.rxnorm_concepts[cui]["prefLabel"]})
        for name, cuis in self.rxnorm_name_index.items():
            if q in name or name in q:
                for cui in cuis:
                    if not any(r["rxcui"] == cui for r in results):
                        ratio = len(q) / max(len(name), 1)
                        results.append({"rxcui": cui, "score": min(ratio, 0.9), "name": self.rxnorm_concepts[cui]["prefLabel"]})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_icd10_code(self, code: str) -> dict:
        return self.icd10.get(code, {})

    def get_rxnorm_concept(self, rxcui: str) -> dict:
        return self.rxnorm_concepts.get(rxcui, {})
