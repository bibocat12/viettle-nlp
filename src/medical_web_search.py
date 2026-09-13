"""Privacy-preserving, cached public medical terminology lookup.

Only the entity string is sent to public sources; never the clinical note or
patient context.  The response is evidence for a reviewer, never an authority
to emit a code outside the local ICD-10/RxNorm KB.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote_plus

import requests


class MedicalWebSearch:
    def __init__(self, cache_dir: Path, timeout: int = 8):
        self.cache_dir = cache_dir
        self.timeout = timeout

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")[:100] or "query"

    def _cache(self, source: str, query: str) -> Path:
        return self.cache_dir / source / f"{self._slug(query)}.json"

    def _get(self, source: str, query: str, url: str) -> dict:
        path = self._cache(source, query)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        try:
            data = requests.get(url, timeout=self.timeout, headers={"User-Agent": "ViettleNLP/1.0"}).json()
        except Exception:
            data = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    def drug(self, term: str, local_codes: set[str]) -> list[dict]:
        data = self._get(
            "rxnav", term,
            f"https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term={quote_plus(term)}&maxEntries=5",
        )
        hits = []
        for item in data.get("approximateGroup", {}).get("candidate", [])[:8]:
            code = str(item.get("rxcui", ""))
            if code in local_codes and not any(hit["code"] == code for hit in hits):
                hits.append({"code": code, "name": item.get("name", ""), "source": "RxNav"})
        return hits

    def diagnosis(self, term: str) -> list[dict]:
        data = self._get(
            "mesh", term,
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=mesh&term={quote_plus(term)}&retmode=json&retmax=3",
        )
        translated = data.get("esearchresult", {}).get("translationset", [])
        return [{"term": item.get("to", ""), "source": "NCBI MeSH"} for item in translated[:2] if item.get("to")]
