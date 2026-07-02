"""Orchestration: a free-text query in -> a repurposing report out.

The key idea beyond "which targets matter" is DIRECTION. The interpreter says,
per target, whether the goal wants more or less of its activity; here we read each
drug's action type (agonist vs inhibitor) and keep only drugs that push the
targets the RIGHT way. Drugs acting the wrong way (e.g. a GH-blocker when the goal
is to grow taller) are moved to an "opposite effect" list instead of polluting the
candidates.
"""
from __future__ import annotations

import json
import re
from collections import OrderedDict, defaultdict

from app.clients.clinicaltrials import ClinicalTrialsClient
from app.clients.opentargets import OpenTargetsClient
from app.clients.uniprot import UniProtClient
from app.llm.interpret import interpret_query
from app.llm.propose import _norm
from app.llm.provider import get_provider
from app.textutil import no_em_dashes
from app.models.schemas import DrugCandidate, QueryInterpretation, RepurposingReport, Target, TargetSpec

MIN_TARGET_SCORE = 0.1
MAX_TARGETS_SCAN = 30
MAX_CANDIDATES = 80

_STAGE_RANK = {
    "APPROVAL": 4.0, "PHASE_4": 4.0, "PHASE_3": 3.0, "PHASE_2": 2.0,
    "PHASE_1": 1.0, "PHASE_1_2": 1.5, "EARLY_PHASE_1": 0.5, "PRECLINICAL": 0.0,
}

_POSITIVE_ACTIONS = {
    "AGONIST", "PARTIAL AGONIST", "ACTIVATOR", "POSITIVE MODULATOR",
    "POSITIVE ALLOSTERIC MODULATOR", "OPENER", "RELEASING AGENT", "STABILISER",
}
_NEGATIVE_ACTIONS = {
    "INHIBITOR", "ANTAGONIST", "BLOCKER", "NEGATIVE MODULATOR",
    "NEGATIVE ALLOSTERIC MODULATOR", "DEGRADER", "DISRUPTING AGENT",
    "INVERSE AGONIST", "SEQUESTERING AGENT", "PROTEOLYSIS TARGETING CHIMERA",
}


def _stage_to_number(stage: str | None) -> float | None:
    return _STAGE_RANK.get(stage.upper()) if stage else None


def _action_sign(action_type: str | None) -> int:
    if not action_type:
        return 0
    a = action_type.upper()
    if a in _POSITIVE_ACTIONS:
        return 1
    if a in _NEGATIVE_ACTIONS:
        return -1
    return 0


def _first_moa(drug: dict) -> tuple[str | None, str | None]:
    rows = (drug.get("mechanismsOfAction") or {}).get("rows") or []
    if not rows:
        return None, None
    return rows[0].get("mechanismOfAction"), rows[0].get("actionType")


# --------------------------------------------------------------------------
def build_report(query: str) -> RepurposingReport:
    provider = get_provider()
    interp = interpret_query(query, provider)
    report = RepurposingReport(query=query, interpretation=interp)

    with OpenTargetsClient() as ot:
        targets = _resolve_targets(interp, report, ot)
        if not targets:
            report.notes.append("No relevant biological targets could be resolved for this query.")
            return report
        report.targets = targets
        raw = _aggregate_drugs(targets[:MAX_TARGETS_SCAN], ot)

    existing, repurposing, opposite = _build_candidates(raw, interp, report)

    # Real investigational drugs from active clinical trials for the goal's conditions
    # (this is what reliably surfaces brand-new agents like TYRA-300 that databases lack)
    trial_cands = _trial_candidates(interp, report, {c.name for c in existing + repurposing + opposite})
    # AI-proposed PROSPECTIVE candidates beyond the databases (novel/selective agents)
    known_after_trials = existing + repurposing + opposite + trial_cands
    proposed = _propose_prospective(interp, report, known_after_trials, provider)
    repurposing = trial_cands + proposed + repurposing

    # Collapse brand / synonym / formulation variants to ONE entry per active ingredient,
    # drop non-drugs, and promote AI-confirmed established treatments into Existing.
    goal_text = (interp.goal_label if interp else None) or report.disease_name or query
    existing, repurposing = _dedupe_by_ingredient(existing, repurposing, goal_text, provider)
    _assign_labels(existing)
    _assign_labels(repurposing)

    report.existing_solutions = sorted(existing, key=lambda c: (c.effectiveness_score, c.safety_score), reverse=True)[:MAX_CANDIDATES]
    report.repurposing_candidates = sorted(repurposing, key=lambda c: (c.confidence, c.safety_score), reverse=True)[:MAX_CANDIDATES]
    report.opposite_effect = sorted(opposite, key=lambda c: c.effectiveness_score, reverse=True)[:30]

    _enrich_targets(report)
    report.summary = _maybe_summarize(report, provider)
    return report


# ---- resolve targets -----------------------------------------------------
def _resolve_targets(interp: QueryInterpretation, report: RepurposingReport, ot: OpenTargetsClient) -> list[Target]:
    if interp.mode == "disease":
        for term in (interp.disease_terms or [interp.original]):
            disease = ot.resolve_disease_id(term)
            if not disease:
                continue
            report.disease_id = disease["id"]
            report.disease_name = disease["name"]
            dt = ot.disease_targets(disease["id"], size=MAX_TARGETS_SCAN)
            targets = []
            for row in dt.get("associatedTargets", {}).get("rows", []):
                if row["score"] < MIN_TARGET_SCORE:
                    continue
                t = row["target"]
                targets.append(Target(ensembl_id=t["id"], symbol=t["approvedSymbol"],
                                      name=t.get("approvedName"), association_score=row["score"]))
            return targets
        return []

    # targets mode: explicit specs with directions
    targets = []
    for spec in interp.target_specs:
        try:
            hit = ot.resolve_target(spec.symbol)
        except Exception:
            continue  # skip a target the API failed on rather than aborting the whole query
        if hit:
            targets.append(Target(ensembl_id=hit["id"], symbol=spec.symbol,
                                  name=hit.get("name"), association_score=spec.weight))
    if interp.source == "heuristic":
        report.notes.append("Targets + directions inferred without a live AI key (heuristic). "
                            "Add a Gemini key for full natural-language understanding of any query.")
    return targets


# ---- aggregate drugs -----------------------------------------------------
def _aggregate_drugs(targets: list[Target], ot: OpenTargetsClient) -> dict[str, dict]:
    drugs: dict[str, dict] = {}
    via: dict[str, set[str]] = defaultdict(set)
    target_relevance: dict[str, float] = defaultdict(float)
    known_for: dict[str, set[str]] = defaultdict(set)
    indicated_ids: dict[str, set[str]] = defaultdict(set)

    for t in targets:
        try:
            td = ot.target_drugs(t.ensembl_id)
        except Exception:
            continue  # skip a target whose drug fetch failed; keep the rest
        for row in (td.get("drugAndClinicalCandidates") or {}).get("rows", []) or []:
            drug = row.get("drug") or {}
            name = drug.get("name")
            if not name:
                continue
            stage = row.get("maxClinicalStage") or drug.get("maximumClinicalStage")
            ae = drug.get("adverseEvents") or {}
            ae_rows = ae.get("rows") or []
            warnings = drug.get("drugWarnings") or []
            moa, action = _first_moa(drug)
            d = drugs.setdefault(name, {
                "chembl_id": drug.get("id"), "drug_type": drug.get("drugType"),
                "max_phase": None, "clinical_stage": None, "moa": moa, "action_type": action,
                "adverse_event_types": ae.get("count", 0) or 0,
                "total_adverse_reports": sum(r.get("count", 0) for r in ae_rows),
                "top_adverse_events": [r.get("name") for r in ae_rows[:5] if r.get("name")],
                "has_black_box": any(w.get("warningType") == "Black Box Warning" for w in warnings),
                "is_withdrawn": any(w.get("warningType") == "Withdrawn" for w in warnings),
                "warnings": sorted({w.get("warningType") for w in warnings if w.get("warningType")}),
            })
            phase = _stage_to_number(stage)
            if phase is not None and (d["max_phase"] is None or phase > d["max_phase"]):
                d["max_phase"], d["clinical_stage"] = phase, stage
            via[name].add(t.symbol)
            target_relevance[name] = max(target_relevance[name], t.association_score)
            for item in (drug.get("indications") or {}).get("rows", []):
                dis = item.get("disease")
                if dis:
                    if dis.get("name"):
                        known_for[name].add(dis["name"])
                    if dis.get("id"):
                        indicated_ids[name].add(dis["id"])
            for item in row.get("diseases") or []:
                dis = item.get("disease")
                if dis and dis.get("name"):
                    known_for[name].add(dis["name"])

    for name, d in drugs.items():
        d["via_targets"] = sorted(via[name])
        d["target_relevance"] = target_relevance[name]
        d["known_for"] = sorted(known_for[name])
        d["indicated_ids"] = indicated_ids[name]
    return drugs


# ---- build / classify / score --------------------------------------------
def _build_candidates(raw, interp: QueryInterpretation, report: RepurposingReport):
    specs = {s.symbol: s for s in interp.target_specs}
    goal_mode = bool(specs)
    goal_keywords = interp.goal_keywords
    disease_id = report.disease_id

    existing, repurposing, opposite = [], [], []
    for name, d in raw.items():
        is_approved = (d["max_phase"] or 0) >= 4
        cand = DrugCandidate(
            name=name, chembl_id=d["chembl_id"], drug_type=d["drug_type"],
            is_approved=is_approved, clinical_stage=d["clinical_stage"], max_phase=d["max_phase"],
            mechanism_of_action=d["moa"], action_type=d["action_type"],
            via_targets=d["via_targets"], known_for=d["known_for"][:10],
            adverse_event_types=d["adverse_event_types"], total_adverse_reports=d["total_adverse_reports"],
            top_adverse_events=d["top_adverse_events"], has_black_box=d["has_black_box"],
            is_withdrawn=d["is_withdrawn"], warnings=d["warnings"],
        )

        # directional assessment (goal mode only) — needed before classifying
        if goal_mode:
            _direction_and_fit(cand, specs)

        # classify existing vs repurposing
        if goal_mode:
            text = " ".join(cand.known_for).lower()
            keyword_hit = any(k in text for k in goal_keywords)
            is_oncology = any(k in text for k in _ONCOLOGY_TERMS)
            # "established for the goal" = indicated for it, OR an approved drug that pushes a
            # relevant target the right way and is NOT primarily a cytotoxic/oncology agent
            # (oncology drugs hitting these targets are the classic repurposing candidates).
            established = is_approved and cand.direction == "helps" and cand.goal_fit >= 0.6 and not is_oncology
            is_existing = keyword_hit or established
        else:
            is_existing = bool(is_approved and disease_id and disease_id in d["indicated_ids"])

        _score(cand, d["target_relevance"], is_existing, goal_mode)

        # route
        if goal_mode and cand.direction == "opposite":
            opposite.append(cand)
            continue
        if not is_existing and cand.is_withdrawn:
            continue  # failed drug, drop from repurposing
        (existing if is_existing else repurposing).append(cand)

    return existing, repurposing, opposite


def _direction_and_fit(cand: DrugCandidate, specs: dict[str, TargetSpec]) -> None:
    sign = _action_sign(cand.action_type)
    best_help = best_harm = 0.0
    n_help = 0
    for sym in cand.via_targets:
        spec = specs.get(sym)
        if not spec or sign == 0:
            continue
        desired = 1 if spec.direction == "up" else -1
        if sign == desired:
            best_help = max(best_help, spec.weight)
            n_help += 1
        else:
            best_harm = max(best_harm, spec.weight)

    cand.goal_fit = round(max(0.0, min(1.0, best_help + 0.1 * max(0, n_help - 1) - 0.5 * best_harm)), 3)
    if best_help > 0 and best_help >= best_harm:
        cand.direction = "helps"
    elif best_harm > 0 and best_help == 0:
        cand.direction = "opposite"
    else:
        cand.direction = "unclear"


_ONCOLOGY_TERMS = (
    "cancer", "carcinoma", "neoplasm", "tumour", "tumor", "sarcoma", "leukemia",
    "leukaemia", "lymphoma", "malignan", "glioma", "melanoma", "adenocarcinoma",
    "myeloma", "blastoma",
)


def _score(cand: DrugCandidate, target_relevance: float, is_existing: bool, goal_mode: bool) -> None:
    phase_norm = (cand.max_phase or 0) / 4.0
    if goal_mode:
        establishment = 1.0 if is_existing else 0.45
        cand.effectiveness_score = round(0.5 * cand.goal_fit + 0.2 * phase_norm + 0.3 * establishment, 3)
        cand.confidence = round(0.6 * cand.effectiveness_score + 0.4 * cand.goal_fit, 3)
    else:
        coverage = min(len(cand.via_targets) / 3.0, 1.0)
        cand.effectiveness_score = round(0.6 * phase_norm + 0.25 * coverage + 0.15 * target_relevance, 3)
        cand.confidence = round(0.7 * cand.effectiveness_score + 0.3 * target_relevance, 3)
    cand.safety_score = _safety(cand)


def _safety(cand: DrugCandidate) -> float:
    """Safety = how well-established and benign the profile is — NOT how few FAERS
    reports it has (report volume mostly tracks how widely a drug is used). So we
    score clinical maturity, serious warnings and drug-class risk instead."""
    phase = cand.max_phase or 0
    if phase >= 4:        # approved: established, characterised safety profile
        s = 0.78
    elif phase >= 3:
        s = 0.6
    elif phase >= 2:
        s = 0.45
    elif phase >= 1:
        s = 0.32
    else:                 # preclinical/experimental: risks largely UNKNOWN -> low, not high
        s = 0.2

    if cand.has_black_box:
        s -= 0.30
    if cand.is_withdrawn:
        s -= 0.55
    s -= 0.04 * max(0, len(cand.warnings) - (1 if cand.has_black_box else 0))

    text = " ".join(cand.known_for).lower()
    if any(k in text for k in _ONCOLOGY_TERMS):
        s -= 0.20          # oncology/cytotoxic class is inherently harsher
    dt = (cand.drug_type or "").lower()
    if dt == "small molecule":
        s -= 0.05
    elif dt in ("protein", "enzyme", "oligosaccharide", "antibody"):
        s += 0.03          # replacement biologics tend to be better tolerated

    # breadth of distinct significant adverse events (capped; not raw volume)
    s -= min(cand.adverse_event_types, 40) / 40 * 0.18
    return round(max(0.05, min(1.0, s)), 3)


def _assign_labels(cands: list[DrugCandidate]) -> None:
    for c in cands:
        if c.direction == "helps":
            c.labels.append("Right direction")
        if c.is_approved:
            c.labels.append("Approved")
        elif (c.max_phase or 0) <= 0:
            c.labels.append("Experimental")
        if c.has_black_box:
            c.labels.append("Black-box warning")

    if len(cands) < 2:
        return
    # tie-break "Most effective" toward the safer / more-established drug
    by_eff = sorted(cands, key=lambda c: (c.effectiveness_score, c.safety_score), reverse=True)
    by_safe = sorted(cands, key=lambda c: c.safety_score, reverse=True)
    by_eff[0].labels.insert(0, "Most effective")
    if by_safe[0].safety_score >= 0.6:
        by_safe[0].labels.insert(0, "Safest")
    if by_safe[-1].safety_score < 0.5:
        by_safe[-1].labels.append("Most side effects")
    for c in cands:
        if c.effectiveness_score >= 0.7 and c.safety_score <= 0.3 and "Most effective" not in c.labels:
            c.labels.append("Effective but harsh")
        elif c.safety_score >= 0.7 and "Safest" not in c.labels:
            c.labels.append("Low side effects")


# ---- enrichment + narrative ----------------------------------------------
def _trial_candidates(interp: QueryInterpretation, report: RepurposingReport, known_names: set[str]) -> list[DrugCandidate]:
    # which conditions to search trials for
    if interp and interp.goal_keywords:
        conditions = interp.goal_keywords[:6]
    elif report.disease_name:
        conditions = [report.disease_name]
    elif interp and interp.goal_label:
        conditions = [interp.goal_label]
    else:
        conditions = [report.query]

    # terms used to confirm a trial is actually ABOUT the goal (kills fuzzy-match noise
    # like ClinicalTrials.gov expanding "growth" into oncology "growth factor" trials)
    relevance = [t.lower() for t in (interp.goal_keywords if (interp and interp.goal_keywords) else [])]
    if report.disease_name:
        relevance.append(report.disease_name.lower())
    if not relevance and interp and interp.goal_label:
        relevance.append(interp.goal_label.lower())

    def _relevant(conds: list[str]) -> bool:
        if not relevance:
            return True
        blob = " | ".join(conds).lower()
        return any(term in blob for term in relevance)

    known_norm = {_norm(n) for n in known_names}
    agg: dict[str, dict] = {}
    with ClinicalTrialsClient() as ct:
        for cond in conditions:
            try:
                drugs = ct.drugs_for_condition(cond, page_size=50)
            except Exception:
                continue
            for d in drugs:
                key = _norm(d["name"])
                if not key or key in known_norm or not _relevant(d["conditions"]):
                    continue
                rec = agg.get(key)
                if rec is None:
                    agg[key] = dict(d)
                else:
                    rec["nct_ids"] = sorted(set(rec["nct_ids"]) | set(d["nct_ids"]))[:5]
                    rec["conditions"] = sorted(set(rec["conditions"]) | set(d["conditions"]))[:5]
                    rec["max_phase"] = max(rec["max_phase"], d["max_phase"])

    out: list[DrugCandidate] = []
    for d in agg.values():
        phase = d["max_phase"]
        eff = round(0.4 + 0.12 * phase, 3)            # more advanced trials -> higher prospect
        safe = round(0.3 + 0.10 * phase, 3)           # investigational -> moderate, grows with phase
        out.append(DrugCandidate(
            name=d["name"],
            clinical_stage=d["clinical_stage"] or "In trials",
            max_phase=phase or None,
            mechanism_of_action=f"In clinical trials for {', '.join(d['conditions'][:3]) or 'this area'}",
            direction="helps",
            effectiveness_score=eff, safety_score=safe, confidence=eff, goal_fit=eff,
            known_for=d["conditions"],
            source="clinical_trials", prospective=True,
            trial_ids=d["nct_ids"],
            labels=["In clinical trials", "Prospective"],
        ))
    return out


def _propose_prospective(interp: QueryInterpretation, report: RepurposingReport, known, provider) -> list[DrugCandidate]:
    from app.llm.propose import propose_candidates

    goal = (interp.goal_label if interp else None) or report.disease_name or report.query

    # example database drugs per target (grounding so the model proposes selective
    # versions of validated targets rather than inventing drug-target links)
    examples: dict[str, list[str]] = defaultdict(list)
    for c in known:
        if c.direction != "opposite":
            for sym in c.via_targets:
                if len(examples[sym]) < 5 and c.name not in examples[sym]:
                    examples[sym].append(c.name)

    if interp and interp.target_specs:
        specs = sorted(interp.target_specs, key=lambda s: s.weight, reverse=True)[:7]
        targets_info = [{"symbol": s.symbol, "direction": s.direction, "weight": s.weight,
                         "examples": examples.get(s.symbol, [])} for s in specs]
    else:
        targets_info = [{"symbol": t.symbol, "direction": None, "weight": t.association_score,
                         "examples": examples.get(t.symbol, [])} for t in report.targets[:7]]
    known_names = {c.name for c in known}
    try:
        return propose_candidates(goal, targets_info, known_names, provider)
    except Exception:
        return []


_CANON_SYSTEM = (
    "You normalize a list of intervention names from drug databases and clinical trials. For EACH "
    "input name return an object {family, display, is_drug}.\n"
    "- is_drug = false for anything that is NOT a specific drug/compound/biologic: e.g. 'physical "
    "exercise', 'lifestyle intervention', 'diet', 'placebo', 'surgery', 'counseling', 'education', a "
    "device, or a vague phrase like 'targeted agent'. Supplements (e.g. vitamin D) ARE drugs (true).\n"
    "- family = a short lowercase key that GROUPS the same drug with its minor variants: salts, esters, "
    "brands, formulations, PEGylated/long-acting versions, and closely-related analogs that do the same "
    "job. Examples: all insulins (insulin lispro/aspart/glargine/degludec/glulisine/detemir, 'insulin "
    "beef pork') -> 'insulin'; all growth-hormone forms (somatropin/somatrem/somatrogon/lonapegsomatropin/"
    "somapacitan) -> 'somatropin'; 'infigratinib phosphate' & 'infigratinib' -> 'infigratinib'. Do NOT "
    "group genuinely different drugs that merely share a target.\n"
    "- For a verbose trial-arm description, extract the CORE drug for family/display (e.g. 'lifestyle "
    "intervention and semaglutide injectable' -> family 'semaglutide'). If no real drug is present, is_drug=false.\n"
    "- display = a clean human-readable family name (e.g. 'Insulin', 'Somatropin', 'Infigratinib').\n"
    "- established = true ONLY if the drug is an APPROVED or standard, established treatment used "
    "SPECIFICALLY for the stated GOAL (e.g. for hair regrowth: minoxidil/finasteride/dutasteride are "
    "established=true). An investigational drug, or one approved only for a DIFFERENT disease, is false.\n"
    "Respond with STRICT JSON only."
)


def _canonicalize(names: list[str], goal: str, provider) -> dict[str, dict]:
    """name -> {family, display, is_drug, established} via the live model.

    This one call drives grouping, non-drug filtering and 'established for goal' promotion, so a
    silent failure degrades everything. We retry once rather than losing it all.
    """
    uniq = sorted({n for n in names if n})[:90]
    if not provider.live or not uniq:
        return {}
    prompt = (f'GOAL: {goal}\n\nReturn JSON mapping each EXACT input name to '
              '{"family":"..","display":"..","is_drug":true|false,"established":true|false}.\n'
              "Names:\n" + json.dumps(uniq))
    for _ in range(2):
        try:
            raw = provider.complete(_CANON_SYSTEM, prompt, temperature=0.0)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(match.group(0) if match else raw)
            out: dict[str, dict] = {}
            for n, v in data.items():
                if isinstance(v, dict):
                    out[n] = {
                        "family": str(v.get("family", "")).strip().lower(),
                        "display": (v.get("display") or "").strip(),
                        "is_drug": bool(v.get("is_drug", True)),
                        "established": bool(v.get("established", False)),
                    }
            if out:
                return out
        except Exception:
            continue
    return {}


def _dedupe_by_ingredient(existing: list[DrugCandidate], repurposing: list[DrugCandidate], goal: str, provider):
    """Group same-drug/family variants into ONE card (e.g. all insulins -> Insulin), drop non-drug
    interventions (physical exercise, lifestyle...), and give each card a clean display name. Merged
    variant names are kept on the card for the detail popup. Database drugs are always treated as
    real drugs (never dropped as 'not a drug')."""
    info = _canonicalize([c.name for c in existing + repurposing], goal, provider)

    def famkey(c: DrugCandidate) -> str:
        d = info.get(c.name)
        return (d["family"] if d and d.get("family") else "") or _norm(c.name)

    def is_drug(c: DrugCandidate) -> bool:
        if c.source == "database":
            return True
        d = info.get(c.name)
        return d.get("is_drug", True) if d else True

    existing = [c for c in existing if is_drug(c)]
    repurposing = [c for c in repurposing if is_drug(c)]

    groups: "OrderedDict[str, list[tuple[str, DrugCandidate]]]" = OrderedDict()
    for tag, c in [("e", c) for c in existing] + [("r", c) for c in repurposing]:
        groups.setdefault(famkey(c), []).append((tag, c))

    new_existing, new_repurposing = [], []
    for fam, members in groups.items():
        cands = [c for _, c in members]
        rep = max(cands, key=lambda c: (1 if c.source == "database" else 0, c.max_phase or 0, c.effectiveness_score))
        names = sorted({c.name for c in cands})
        for c in cands:
            if c is not rep:
                _merge_into(rep, c)
        disp = next((info[c.name]["display"] for c in cands if info.get(c.name, {}).get("display")), None)
        if disp:
            rep.name = disp
        if len(names) > 1:
            rep.variants = names

        # a group is "established" if any member was already Existing, OR the AI says this drug is
        # an approved/standard treatment for THIS goal (promotes e.g. minoxidil out of repurposing)
        ai_established = any(info.get(c.name, {}).get("established") for c in cands)
        goes_existing = any(t == "e" for t, _ in members) or ai_established
        if goes_existing and rep.prospective:
            # promoted: drop the investigational framing
            rep.prospective = False
            rep.labels = [l for l in rep.labels if l not in ("In clinical trials", "Prospective")]
            if rep.source == "clinical_trials":
                rep.source = "established"
        (new_existing if goes_existing else new_repurposing).append(rep)
    return new_existing, new_repurposing


def _merge_into(rep: DrugCandidate, c: DrugCandidate) -> None:
    rep.trial_ids = sorted(set(rep.trial_ids) | set(c.trial_ids))[:6]
    rep.known_for = sorted(set(rep.known_for) | set(c.known_for))[:10]
    rep.via_targets = sorted(set(rep.via_targets) | set(c.via_targets))
    if (c.max_phase or 0) > (rep.max_phase or 0):
        rep.max_phase, rep.clinical_stage = c.max_phase, c.clinical_stage


def _enrich_targets(report: RepurposingReport, top: int = 6) -> None:
    with UniProtClient() as up:
        for t in report.targets[:top]:
            try:
                info = up.lookup_by_gene(t.symbol)
            except Exception:
                info = None
            if info:
                t.uniprot_id = info.get("uniprot_id")
                t.protein_function = info.get("function")


def _maybe_summarize(report: RepurposingReport, provider) -> str:
    if not provider.live:
        return ""
    subject = report.disease_name or (report.interpretation.goal_label if report.interpretation else None) or report.query
    targets = ", ".join(t.symbol for t in report.targets[:12])
    rep = "\n".join(
        f"- {c.name} (via {', '.join(c.via_targets)}; {c.action_type}; stage {c.clinical_stage}; labels {', '.join(c.labels)})"
        for c in report.repurposing_candidates[:10]
    )
    system = (
        "You are a friendly science communicator writing for an everyday person with NO medical or "
        "biology background. Write in plain, clear English, like explaining to a smart 15-year-old. "
        "Avoid jargon; if you must mention a gene/protein, immediately explain what it does in simple "
        "words (e.g. 'FGFR3, a protein that acts like a brake on bone growth'). Use short paragraphs. "
        "NEVER use em dashes or en dashes; use commas, periods, or 'and'. "
        "These are research ideas to investigate, not proven treatments or medical advice, say so plainly."
    )
    prompt = (
        f"Topic the user searched: {subject}\n"
        f"Biology involved (technical target names, translate these for the reader): {targets}\n\n"
        f"Candidate drugs the tool found (technical):\n{rep}\n\n"
        "Write a short, easy-to-read summary with these three clearly headed parts:\n"
        "**What's happening in the body**: explain the relevant biology in simple everyday language.\n"
        "**Drugs that might help, and why**: pick 3-5 of the candidates and explain in ONE simple "
        "sentence each what the drug does and why it might help (no jargon).\n"
        "**How confident should you be**: an honest, plain-English note on how speculative this is and "
        "that it needs real research. Keep the whole thing concise and genuinely understandable."
    )
    try:
        return no_em_dashes(provider.complete(system, prompt))
    except Exception:
        return ""
