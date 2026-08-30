"""NCBI client: the literature backbone of the project.

Wraps the four NCBI services in the plan:
  1. E-utilities (esearch)        -> topic string  ->  list of PMIDs
  2. BioC                         -> PMID/PMCID     ->  clean article text (JSON)
  3. PMC ID Converter             -> PMID           ->  PMCID (only for full text)
  4. Literature Citation Exporter -> PMID/PMCID     ->  formatted citation (RIS/MEDLINE)

Abstracts only need a PMID (BioC pubmed mode). Open-access full text needs a
PMCID, which is why the ID Converter exists.
"""
from __future__ import annotations

from typing import Any

from app.config import settings
from .base import ApiError, make_client

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BIOC_PUBMED = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pubmed.cgi"
BIOC_PMCOA = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi"
IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
CITATION = "https://api.ncbi.nlm.nih.gov/lit/ctxp/v1"


def _common_params() -> dict[str, str]:
    """tool + email + api_key that NCBI asks be sent on every request."""
    params: dict[str, str] = {"tool": settings.ncbi_tool}
    if settings.ncbi_email:
        params["email"] = settings.ncbi_email
    if settings.ncbi_api_key:
        params["api_key"] = settings.ncbi_api_key
    return params


class NcbiClient:
    def __init__(self) -> None:
        self._http = make_client()

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "NcbiClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # 1. E-utilities: topic -> PMIDs --------------------------------------
    def search_pmids(self, term: str, retmax: int = 20, sort: str = "relevance") -> list[str]:
        params = {
            **_common_params(),
            "db": "pubmed",
            "term": term,
            "retmax": str(retmax),
            "retmode": "json",
            "sort": sort,
        }
        r = self._http.get(f"{EUTILS}/esearch.fcgi", params=params)
        r.raise_for_status()
        data = r.json()
        return data.get("esearchresult", {}).get("idlist", [])

    # 1b. E-utilities: PMIDs -> titles (one call) --------------------------
    def esummary_titles(self, pmids: list[str]) -> dict[str, str]:
        """Map each PMID to its article title in a single esummary call (clean study listing)."""
        if not pmids:
            return {}
        params = {
            **_common_params(),
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        }
        r = self._http.get(f"{EUTILS}/esummary.fcgi", params=params)
        r.raise_for_status()
        result = r.json().get("result", {})
        out: dict[str, str] = {}
        for uid in result.get("uids", []):
            title = (result.get(uid, {}) or {}).get("title")
            if title:
                out[uid] = title.strip().rstrip(".")
        return out

    # 2. BioC: PMID -> abstract text --------------------------------------
    def fetch_abstract_bioc(self, pmid: str, encoding: str = "unicode") -> dict[str, Any]:
        url = f"{BIOC_PUBMED}/BioC_json/{pmid}/{encoding}"
        r = self._http.get(url)
        r.raise_for_status()
        return r.json()

    # 2b. BioC: PMCID -> full-text -----------------------------------------
    def fetch_fulltext_bioc(self, pmcid: str, encoding: str = "unicode") -> dict[str, Any]:
        url = f"{BIOC_PMCOA}/BioC_json/{pmcid}/{encoding}"
        r = self._http.get(url)
        r.raise_for_status()
        return r.json()

    # 3. PMC ID Converter: PMID -> PMCID -----------------------------------
    def pmid_to_pmcid(self, pmids: list[str]) -> dict[str, str | None]:
        params = {
            **_common_params(),
            "ids": ",".join(pmids),
            "format": "json",
            "versions": "no",
        }
        r = self._http.get(IDCONV, params=params)
        r.raise_for_status()
        out: dict[str, str | None] = {}
        for rec in r.json().get("records", []):
            out[rec.get("pmid", "")] = rec.get("pmcid")  # pmcid is None if not open-access
        return out

    def pmcid_to_pmid(self, pmcid: str) -> str | None:
        """Resolve a PMCID (e.g. PMC5155684) to its PMID via the ID converter."""
        params = {**_common_params(), "ids": pmcid, "format": "json", "versions": "no"}
        r = self._http.get(IDCONV, params=params)
        r.raise_for_status()
        for rec in r.json().get("records", []):
            if rec.get("pmid"):
                return rec["pmid"]
        return None

    # 4. Citation Exporter: PMID -> formatted citation ---------------------
    def citation(self, pmid: str, fmt: str = "ris", db: str = "pubmed") -> str:
        # fmt: "ris" | "medline" | "csl" ; db: "pubmed" | "pmc"
        params = {"format": fmt, "id": pmid}
        r = self._http.get(f"{CITATION}/{db}/", params=params)
        r.raise_for_status()
        return r.text


def extract_text_from_bioc(bioc: Any) -> str:
    """Flatten a BioC JSON response down to a single readable string.

    The response nests as: [collection] -> documents -> passages -> text.
    Both the pubmed (abstract) and pmcoa (full-text) endpoints share this shape,
    so we walk any collection wrappers down to the documents.
    """
    collections = bioc if isinstance(bioc, list) else [bioc]
    documents: list[dict] = []
    for node in collections:
        if "documents" in node:
            documents.extend(node["documents"])
        else:
            documents.append(node)  # already a document

    chunks: list[str] = []
    for doc in documents:
        for passage in doc.get("passages", []):
            text = passage.get("text")
            if text:
                section = passage.get("infons", {}).get("section_type") or passage.get("infons", {}).get("type")
                chunks.append(f"[{section}] {text}" if section else text)
    if not chunks:
        raise ApiError("No text passages found in BioC document")
    return "\n\n".join(chunks)
