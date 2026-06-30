"""Live smoke test of every external client + the full pipeline.

Run from the backend/ directory:
    python -m scripts.smoke_test
Hits real APIs (no key required), so it needs internet.
"""
from __future__ import annotations

import sys

from app.clients.chembl import ChemblClient
from app.clients.ncbi import NcbiClient, extract_text_from_bioc
from app.clients.opentargets import OpenTargetsClient
from app.clients.uniprot import UniProtClient
from app.services.pipeline import build_report


def line(label: str) -> None:
    print("\n" + "=" * 60 + f"\n{label}\n" + "=" * 60)


def main() -> int:
    line("NCBI E-utilities: search 'donepezil Alzheimer'")
    with NcbiClient() as ncbi:
        pmids = ncbi.search_pmids("donepezil Alzheimer", retmax=3)
        print("PMIDs:", pmids)
        if pmids:
            bioc = ncbi.fetch_abstract_bioc(pmids[0])
            print("Abstract snippet:", extract_text_from_bioc(bioc)[:200], "...")
            print("PMID->PMCID:", ncbi.pmid_to_pmcid(pmids))

    line("Open Targets: Alzheimer disease -> targets")
    with OpenTargetsClient() as ot:
        disease = ot.resolve_disease_id("Alzheimer disease")
        print("Disease:", disease["id"], disease["name"])
        dt = ot.disease_targets(disease["id"], size=5)
        for row in dt["associatedTargets"]["rows"]:
            print(f"  {row['target']['approvedSymbol']:10} score={row['score']:.3f}")
        top = dt["associatedTargets"]["rows"][0]["target"]
        td = ot.target_drugs(top["id"])
        print(f"Known/candidate drugs for {top['approvedSymbol']}:")
        for row in (td.get("drugAndClinicalCandidates") or {}).get("rows", [])[:3]:
            moa = (row["drug"].get("mechanismsOfAction") or {}).get("rows") or [{}]
            print("  ", row["drug"]["name"], "-", row.get("maxClinicalStage"), "-", moa[0].get("mechanismOfAction"))

    line("UniProt: APP")
    with UniProtClient() as up:
        print(up.lookup_by_gene("APP"))

    line("ChEMBL: donepezil")
    with ChemblClient() as ch:
        print(ch.search_molecule("donepezil"))

    line("FULL PIPELINE: 'Alzheimer disease'")
    report = build_report("Alzheimer disease")
    print("Disease:", report.disease_name, f"({report.disease_id})")
    print("Targets:", [t.symbol for t in report.targets][:10])
    print("Existing solutions:", [c.name for c in report.existing_solutions[:5]])
    print("Repurposing candidates:")
    for c in report.repurposing_candidates[:5]:
        print(f"  {c.name}  via={c.via_targets[:3]}  conf={c.confidence}  labels={c.labels}")
    print("Summary set?:", bool(report.summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
