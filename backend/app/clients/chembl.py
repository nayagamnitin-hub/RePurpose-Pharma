"""ChEMBL client: drug-level details to complement Open Targets' known drugs.

Used to resolve a drug name to its ChEMBL id and pull max clinical phase,
molecule type and (later) chemistry for molecular-similarity features.
"""
from __future__ import annotations

from typing import Any

from .base import make_client

BASE = "https://www.ebi.ac.uk/chembl/api/data"


class ChemblClient:
    def __init__(self) -> None:
        self._http = make_client(headers={"Accept": "application/json"})

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ChemblClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def search_molecule(self, name: str, limit: int = 1) -> list[dict[str, Any]]:
        params = {"q": name, "limit": str(limit), "format": "json"}
        r = self._http.get(f"{BASE}/molecule/search", params=params)
        r.raise_for_status()
        out = []
        for m in r.json().get("molecules", []):
            out.append(
                {
                    "chembl_id": m.get("molecule_chembl_id"),
                    "pref_name": m.get("pref_name"),
                    "max_phase": m.get("max_phase"),
                    "molecule_type": m.get("molecule_type"),
                    "first_approval": m.get("first_approval"),
                }
            )
        return out
