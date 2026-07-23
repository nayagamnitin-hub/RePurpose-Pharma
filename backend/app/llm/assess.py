"""LLM clinical-assessment + augmentation of the candidate list.

The structured heuristics are noisy: FAERS report counts mostly track how WIDELY a
drug is used (so a popular drug looks 'less safe'), clinical phase is a crude
effectiveness proxy, and database/trial coverage misses well-known agents. This
final step uses the live model's clinical knowledge to, for the given goal:
  - recalibrate effectiveness & safety per drug on REAL profiles (so labels are
    right, e.g. dutasteride has a broader side-effect profile than finasteride),
  - classify established vs investigational for THIS goal,
  - drop off-goal / wrong-direction entries (trial co-interventions),
  - ADD important real drugs the databases missed (e.g. minoxidil, AHK-Cu).

Conservative and reversible: if the model call fails we keep the heuristic result.
"""
from __future__ import annotations

import json
import re

from app.models.schemas import DrugCandidate
from app.textutil import no_em_dashes

from .propose import _norm
from .provider import LLMProvider

_SYS = (
    "You are a clinical pharmacology expert curating DRUG options for a stated GOAL. Judge each drug on "
    "REAL medical knowledge, NOT popularity: a widely used drug is NOT less safe just because it has more "
    "reports. For each drug provide: established (set true ONLY if the drug is CURRENTLY APPROVED or routinely "
    "PRESCRIBED in standard clinical practice specifically for THIS goal; investigational, experimental, "
    "research-stage, or off-label-only compounds, even promising ones like curcumin analogs or peptides in "
    "trials, are established=FALSE and belong in repurposing; when unsure, choose false), relevant (false if "
    "it is off-goal, a trial co-intervention, or acts in the "
    "WRONG direction for the goal), effectiveness (0.0-1.0 for this goal), safety (0.0-1.0, higher = safer, "
    "milder or fewer side effects on its real profile), side_effects (a few, comma separated), why (one short "
    "clause). ALSO add important real drugs/compounds for this goal that are MISSING from the list, including "
    "standard treatments AND notable investigational/off-label agents, topical treatments, and peptides "
    "(e.g. copper peptides) where genuinely relevant; mark those added=true. "
    "STRICT: added items must be actual CHEMICAL or BIOLOGIC SUBSTANCES (small molecules, biologics, "
    "peptides, hormones). Do NOT add devices, laser or light therapy, procedures, surgery, diet, exercise, "
    "or lifestyle interventions. No em dashes. Respond with STRICT JSON only."
)

# names that are not drugs (guard against the model adding devices/procedures anyway)
_NON_DRUG_RE = re.compile(
    r"(?i)\b(laser|light therapy|phototherapy|device|procedure|surgery|surgical|therapy session|"
    r"exercise|diet|lifestyle|counsel|education|transplant|microneedl|dermaroll|massage|nutrition)\b"
)


def assess_and_augment(goal, existing, repurposing, provider: LLMProvider, max_assess: int = 45, max_added: int = 12):
    if not provider.live or not goal:
        return existing, repurposing

    tagged = [("e", c) for c in existing] + [("r", c) for c in repurposing[:max_assess]]
    listing = [{
        "name": c.name,
        "currently": ("established" if tag == "e" else "repurposing/experimental"),
        "targets": c.via_targets[:4],
        "stage": c.clinical_stage,
        "known_for": c.known_for[:3],
    } for tag, c in tagged]

    prompt = (
        f"GOAL: {goal}\n\nCandidate drugs to judge (and add any important missing ones):\n"
        + json.dumps(listing)
        + '\n\nReturn JSON: {"drugs":[{"name":"..","established":true|false,"relevant":true|false,'
          '"effectiveness":0.0,"safety":0.0,"side_effects":"..","why":"..","added":true|false}]}'
    )

    data = None
    for _ in range(2):
        try:
            raw = provider.complete(_SYS, prompt, temperature=0.0)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed = json.loads(m.group(0) if m else raw)
            if parsed.get("drugs"):
                data = parsed
                break
        except Exception:
            data = None
    if not data:
        return existing, repurposing

    judg: dict[str, dict] = {}
    for d in data["drugs"]:
        nm = (d.get("name") or "").strip()
        if nm:
            judg[_norm(nm)] = d

    new_existing, new_repurposing, used = [], [], set()

    def _apply(c: DrugCandidate, d: dict) -> None:
        eff = _clamp(d.get("effectiveness"), c.effectiveness_score)
        safe = _clamp(d.get("safety"), c.safety_score)
        c.effectiveness_score = round(eff, 3)
        c.safety_score = round(safe, 3)
        c.confidence = round(0.7 * eff + 0.3 * (c.goal_fit or eff), 3)
        se = d.get("side_effects")
        if se:
            c.top_adverse_events = [s.strip() for s in re.split(r"[;,]", se) if s.strip()][:5]

    for tag, c in tagged:
        used.add(_norm(c.name))
        d = judg.get(_norm(c.name))
        if d is not None:
            if d.get("relevant") is False:
                continue  # drop off-goal / wrong-direction
            _apply(c, d)
            (new_existing if d.get("established") else new_repurposing).append(c)
        else:
            (new_existing if tag == "e" else new_repurposing).append(c)

    # keep the un-assessed repurposing tail as-is
    for c in repurposing[max_assess:]:
        if _norm(c.name) not in used:
            used.add(_norm(c.name))
            new_repurposing.append(c)

    # drugs the model added
    added = 0
    for d in data["drugs"]:
        if not d.get("added") or added >= max_added:
            continue
        nm = (d.get("name") or "").strip()
        if not nm or _norm(nm) in used or d.get("relevant") is False or _NON_DRUG_RE.search(nm):
            continue
        used.add(_norm(nm))
        added += 1
        eff = _clamp(d.get("effectiveness"), 0.6)
        safe = _clamp(d.get("safety"), 0.6)
        est = bool(d.get("established"))
        cand = DrugCandidate(
            name=nm, direction="helps",
            mechanism_of_action=no_em_dashes(d.get("why") or "") or None,
            effectiveness_score=round(eff, 3), safety_score=round(safe, 3),
            confidence=round(eff, 3), goal_fit=round(eff, 3),
            top_adverse_events=[s.strip() for s in re.split(r"[;,]", d.get("side_effects") or "") if s.strip()][:5],
            source=("established" if est else "ai"),
            prospective=(not est),
            is_approved=est,
            max_phase=(4.0 if est else None),
            clinical_stage=("Approved / established" if est else "Investigational"),
            labels=(["Established"] if est else ["AI-proposed", "Prospective"]),
        )
        (new_existing if est else new_repurposing).append(cand)

    # Cross-check EVERY drug in Established: is it truly prescribed as standard care for THIS goal?
    new_existing, moved = _verify_established(new_existing, goal, provider)
    return new_existing, moved + new_repurposing


_VERIFY_EST_SYS = (
    "You are a strict clinical fact-checker. For each drug decide if it is CURRENTLY an approved or routinely "
    "PRESCRIBED, standard treatment used specifically for the stated goal. Investigational, experimental, "
    "research-stage, or only-in-trials agents are NOT standard (false), even if promising. If unsure, false. "
    "Respond with STRICT JSON only."
)


def _verify_established(existing, goal, provider):
    """Move any 'established' drug that is not actually standard-of-care for this goal into repurposing."""
    if not provider.live or not existing:
        return existing, []
    names = [c.name for c in existing]
    prompt = (f"GOAL: {goal}\n\nFor each drug, is it currently an APPROVED or routinely PRESCRIBED standard "
              f'treatment specifically for this goal? Return JSON {{"results":[{{"name":"..","standard":true|false}}]}}.\n'
              f"Drugs:\n{json.dumps(names)}")
    verdict: dict[str, bool] = {}
    for _ in range(2):
        try:
            raw = provider.complete(_VERIFY_EST_SYS, prompt, temperature=0.0)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            for r in json.loads(m.group(0) if m else raw).get("results", []):
                nm = (r.get("name") or "").strip()
                if nm:
                    verdict[_norm(nm)] = bool(r.get("standard"))
            if verdict:
                break
        except Exception:
            pass
    if not verdict:
        return existing, []

    keep, moved = [], []
    for c in existing:
        if verdict.get(_norm(c.name)) is False:  # explicitly NOT standard for this goal -> repurposing
            c.labels = [l for l in c.labels if l != "Established"]
            c.prospective = True
            if c.source in ("established", "ai"):
                c.is_approved = False
                c.max_phase = None
                c.labels = [l for l in c.labels if l != "Approved"]
                if c.clinical_stage in ("Approved / established", None):
                    c.clinical_stage = "Investigational"
            moved.append(c)
        else:
            keep.append(c)
    return keep, moved


def _clamp(v, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default
