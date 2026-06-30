"""Open Targets Platform client (GraphQL).

This is the structured backbone that PubMed alone can't give us:
  - resolve a free-text disease name -> an EFO disease id
  - resolve a gene symbol -> an Ensembl target id (for target-driven queries)
  - disease id -> ranked associated targets (proteins/genes) with evidence scores
  - target id -> drugs that act on it, with mechanism, clinical stage, adverse
    events, warnings and approved indications (everything we need for the
    safety / effectiveness labels, in one round trip per target)
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from .base import ApiError, make_client

ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"

_SEARCH = """
query Search($q: String!, $entity: String!, $size: Int!) {
  search(queryString: $q, entityNames: [$entity], page: {index: 0, size: $size}) {
    hits { id name entity description }
  }
}
"""

_DISEASE_TARGETS = """
query DiseaseTargets($efoId: String!, $size: Int!) {
  disease(efoId: $efoId) {
    id
    name
    description
    associatedTargets(page: {index: 0, size: $size}) {
      count
      rows {
        score
        target { id approvedSymbol approvedName }
      }
    }
  }
}
"""

# Open Targets v26 (beta): drugs live under drugAndClinicalCandidates; mechanism,
# adverse events, warnings and indications are nested on the drug.
_TARGET_DRUGS = """
query TargetDrugs($ensgId: String!) {
  target(ensemblId: $ensgId) {
    id
    approvedSymbol
    approvedName
    drugAndClinicalCandidates {
      count
      rows {
        maxClinicalStage
        drug {
          id
          name
          drugType
          maximumClinicalStage
          mechanismsOfAction { rows { mechanismOfAction actionType } }
          adverseEvents { count rows { name count } }
          drugWarnings { warningType toxicityClass }
          indications { count rows { disease { id name } } }
        }
        diseases { disease { id name } }
      }
    }
  }
}
"""


class OpenTargetsClient:
    def __init__(self) -> None:
        self._http = make_client(headers={"Content-Type": "application/json"})

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "OpenTargetsClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _query(self, query: str, variables: dict[str, Any], retries: int = 3) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                r = self._http.post(ENDPOINT, json={"query": query, "variables": variables})
                r.raise_for_status()
                payload = r.json()
                if "errors" in payload:
                    # Open Targets occasionally throws a transient "Internal server error".
                    raise ApiError(f"Open Targets GraphQL error: {payload['errors']}")
                return payload["data"]
            except (httpx.HTTPError, ApiError) as e:
                last_err = e
                if attempt < retries - 1:
                    time.sleep(0.6 * (attempt + 1))  # brief backoff, then retry
        raise last_err  # type: ignore[misc]

    # --- search / resolve -------------------------------------------------
    def search_disease(self, name: str, size: int = 5) -> list[dict[str, Any]]:
        data = self._query(_SEARCH, {"q": name, "entity": "disease", "size": size})
        return data["search"]["hits"]

    def resolve_disease_id(self, name: str) -> dict[str, Any] | None:
        hits = self.search_disease(name, size=1)
        return hits[0] if hits else None

    def resolve_target(self, symbol: str) -> dict[str, Any] | None:
        """Gene symbol (e.g. 'FGFR3') -> {id: Ensembl gene id, name, ...}."""
        data = self._query(_SEARCH, {"q": symbol, "entity": "target", "size": 1})
        hits = data["search"]["hits"]
        return hits[0] if hits else None

    # --- associations -----------------------------------------------------
    def disease_targets(self, efo_id: str, size: int = 25) -> dict[str, Any]:
        data = self._query(_DISEASE_TARGETS, {"efoId": efo_id, "size": size})
        return data["disease"]

    def target_drugs(self, ensembl_id: str) -> dict[str, Any]:
        data = self._query(_TARGET_DRUGS, {"ensgId": ensembl_id})
        return data["target"]
