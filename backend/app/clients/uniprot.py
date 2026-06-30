"""UniProt client: enrich a target gene with protein-level facts.

Open Targets gives us Ensembl gene ids + symbols; UniProt turns a symbol into
the human protein's function, names and cross-references, which feeds the
mechanism-of-action explanations.
"""
from __future__ import annotations

from typing import Any

from .base import make_client

BASE = "https://rest.uniprot.org/uniprotkb"


class UniProtClient:
    def __init__(self) -> None:
        self._http = make_client(headers={"Accept": "application/json"})

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "UniProtClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def lookup_by_gene(self, gene_symbol: str, organism_id: int = 9606) -> dict[str, Any] | None:
        """Return a compact summary of the reviewed (Swiss-Prot) human entry."""
        params = {
            "query": f"gene_exact:{gene_symbol} AND organism_id:{organism_id} AND reviewed:true",
            "fields": "accession,id,protein_name,gene_primary,cc_function,length",
            "format": "json",
            "size": "1",
        }
        r = self._http.get(f"{BASE}/search", params=params)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None
        entry = results[0]
        function = ""
        for comment in entry.get("comments", []):
            if comment.get("commentType") == "FUNCTION":
                texts = comment.get("texts", [])
                if texts:
                    function = texts[0].get("value", "")
                break
        protein = entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value")
        return {
            "accession": entry.get("primaryAccession"),
            "uniprot_id": entry.get("uniProtkbId"),
            "protein_name": protein,
            "function": function,
            "length": entry.get("sequence", {}).get("length"),
        }
