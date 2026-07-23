"""ClinicalTrials.gov v2 API client (free, no key).

This is what makes the tool's "prospective" purpose real: by searching trials for
the goal's conditions (e.g. achondroplasia / short stature), we surface drugs that
are ACTUALLY being investigated — including brand-new agents the other databases
don't list yet (e.g. TYRA-300/dabogratinib, recifercept, navepegritide). Because
these drugs are being trialled FOR the goal, they're inherently goal-appropriate.
"""
from __future__ import annotations

import re
from typing import Any

from .base import make_client

BASE = "https://clinicaltrials.gov/api/v2"

# intervention types worth treating as candidate "drugs"
_DRUG_TYPES = {"DRUG", "BIOLOGICAL", "GENETIC", "COMBINATION_PRODUCT"}
_SKIP_NAME_HINTS = ("placebo", "questionnaire", "standard of care", "best supportive",
                    "saline", "vehicle", "control group", "control)", "no intervention",
                    "sham", "untreated", "comparator", " alone", "exercise", "lifestyle",
                    "physical activity", "diet ", "dietary", "counseling", "counselling",
                    "education", "surgery", "surgical", "device", "training", "rehabilitation",
                    "and its metabolites", "titration of")

_PHASE_NUM = {"PHASE4": 4.0, "PHASE3": 3.0, "PHASE2": 2.0, "PHASE1": 1.0, "EARLY_PHASE1": 0.5, "NA": None}


def _clean_name(name: str) -> str:
    """Trim trial-arm verbosity to the DRUG name (keeps codes like TYRA-300 / BMN 111).

    'Infigratinib 0.25 mg/kg/day' -> 'Infigratinib'; 'Masitinib 4.5' -> 'Masitinib';
    'MSC-NTF cells transplantation by multiple ... injections' -> 'MSC-NTF cells'.
    """
    name = name.replace("®", " ").replace("™", " ")
    # cut at any procedure / delivery / description connector -> keep the part before it
    name = re.split(
        r"\s+(?:is provided|administered|administration|via|injection|injections|infusion|infusions|"
        r"transplantation|transplant|implantation|by |in addition|at \d|for the|for treatment|"
        r"subcutaneous|intravenous|intrathecal|intramuscular|oral|tablet|capsule|solution|suspension)\b",
        name, flags=re.I)[0]
    name = name.split(":")[0]
    name = re.sub(r"\s*\([^)]*group[^)]*\)", "", name, flags=re.I)  # drop "(treated group)" etc.
    # drop a trailing dose with a unit (so numeric drug codes survive)
    name = re.sub(r"\s+\d+(\.\d+)?\s*(mg|kg|mcg|g|ml|iu|units?|%)\b.*$", "", name, flags=re.I)
    # drop a trailing bare DECIMAL dose like 'Masitinib 4.5' / 'Masitinib 3.0' (codes use whole numbers)
    name = re.sub(r"\s+\d+\.\d+\s*$", "", name)
    return re.sub(r"\s+", " ", name).strip(" -,")


class ClinicalTrialsClient:
    def __init__(self) -> None:
        self._http = make_client(headers={"Accept": "application/json"})

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ClinicalTrialsClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def drugs_for_condition(self, condition: str, page_size: int = 50) -> list[dict[str, Any]]:
        """Return de-duplicated drug interventions studied for a condition.

        Each entry: {name, types, max_phase, clinical_stage, status, nct_ids, conditions}.
        """
        params = {
            "query.cond": condition,
            "pageSize": str(page_size),
            "fields": ",".join([
                "protocolSection.identificationModule.nctId",
                "protocolSection.armsInterventionsModule.interventions",
                "protocolSection.designModule.phases",
                "protocolSection.statusModule.overallStatus",
                "protocolSection.conditionsModule.conditions",
            ]),
        }
        r = self._http.get(f"{BASE}/studies", params=params)
        r.raise_for_status()
        studies = r.json().get("studies", [])

        agg: dict[str, dict[str, Any]] = {}
        for s in studies:
            ps = s.get("protocolSection", {})
            nct = ps.get("identificationModule", {}).get("nctId")
            status = ps.get("statusModule", {}).get("overallStatus")
            phases = ps.get("designModule", {}).get("phases", []) or []
            conds = ps.get("conditionsModule", {}).get("conditions", []) or []
            phase_num = max((_PHASE_NUM.get(p) or 0 for p in phases), default=0.0)
            stage = phases[-1] if phases else None
            for iv in ps.get("armsInterventionsModule", {}).get("interventions", []) or []:
                itype = (iv.get("type") or "").upper()
                name = _clean_name(iv.get("name") or "")
                if not name or itype not in _DRUG_TYPES:
                    continue
                if any(h in name.lower() for h in _SKIP_NAME_HINTS):
                    continue
                key = name.lower()
                rec = agg.setdefault(key, {
                    "name": name, "types": set(), "max_phase": 0.0,
                    "clinical_stage": None, "status": status, "nct_ids": set(), "conditions": set(),
                })
                rec["types"].add(itype)
                rec["nct_ids"].add(nct)
                rec["conditions"].update(conds[:3])
                if phase_num > rec["max_phase"]:
                    rec["max_phase"] = phase_num
                    rec["clinical_stage"] = stage
        # finalise sets -> lists
        out = []
        for rec in agg.values():
            rec["types"] = sorted(rec["types"])
            rec["nct_ids"] = sorted(x for x in rec["nct_ids"] if x)[:5]
            rec["conditions"] = sorted(rec["conditions"])[:5]
            out.append(rec)
        return out
