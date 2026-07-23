"""AI-proposed PROSPECTIVE repurposing candidates.

The databases only know drugs already linked to a target. The whole point of this
tool is to go further: given the goal and the VALIDATED target biology, have the
model propose drugs that mechanistically could work — including new, selective or
investigational agents the databases don't list yet (e.g. a selective FGFR3
inhibitor like TYRA-300 for height). The model is grounded on the confirmed targets
and the real database drugs that hit them, and asked for more-selective / next-gen
versions — which both improves accuracy and follows the "selective beats broad"
mechanistic logic.

These are clearly flagged prospective / AI-proposed so the user treats them as
hypotheses, not established facts.
"""
from __future__ import annotations

import json
import re

from app.models.schemas import DrugCandidate
from app.textutil import no_em_dashes

from .provider import LLMProvider

_SYSTEM = (
    "You are a drug-repurposing scientist proposing PROSPECTIVE candidates for a hypothesis tool. "
    "CRITICAL ACCURACY RULES: only name REAL drugs/compounds that actually exist, and only pair a "
    "drug with a target if that drug genuinely acts on that target — if unsure, OMIT it. Never invent "
    "drug-target relationships. Prefer drugs that are MORE SELECTIVE or next-generation versions for "
    "the VALIDATED targets given (a target-selective agent is usually more effective and safer for a "
    "specific purpose than a broad/multi-target one). Respond with STRICT JSON only."
)


def propose_candidates(
    goal: str,
    targets_info: list[dict],
    known_names: set[str],
    provider: LLMProvider,
    max_n: int = 8,
) -> list[DrugCandidate]:
    if not provider.live or not goal:
        return []

    tgt_lines = []
    for t in targets_info:
        d = t.get("direction")
        arrow = ("DECREASE its activity (need inhibitor/antagonist)" if d == "down"
                 else "INCREASE its activity (need agonist)" if d == "up" else "modulate")
        ex = t.get("examples") or []
        ex_txt = f" — existing database drugs hitting it: {', '.join(ex[:5])}" if ex else ""
        tgt_lines.append(f"- {t['symbol']}: {arrow}{ex_txt}")
    targets_txt = "\n".join(tgt_lines) or "(infer the relevant targets yourself)"

    allowed = ", ".join(t["symbol"] for t in targets_info)
    prompt = f"""Goal: {goal}

VALIDATED targets (confirmed relevant), the direction to push each, and example database
drugs that already act on them:
{targets_txt}

HARD CONSTRAINT: every drug you propose MUST genuinely act on ONE of these exact genes: {allowed}.
Do NOT propose a drug that works through any other target or via vague "indirect" effects — if you
cannot name a real drug that directly acts on one of these exact genes, propose fewer (or none).

Propose up to {max_n} additional REAL drugs to repurpose for this goal. Focus on:
1. MORE SELECTIVE or next-generation agents for the validated targets above. If broad multi-target
   drugs already hit a target, a newer agent SELECTIVE for that single gene would likely be more
   effective and safer for this purpose — name it specifically (recall real generic names or clinical
   development codes, e.g. selective FGFR3 inhibitors in trials for skeletal growth disorders).
2. Real investigational/clinical-stage drugs the databases may not list yet.

Do NOT repeat drugs already listed. Do NOT invent drugs or drug-target links — omit anything uncertain.

Return JSON exactly:
{{"candidates": [
  {{"name": "...", "targets": ["FGFR3"], "action": "inhibitor|agonist|antagonist|...",
    "selectivity": "e.g. selective FGFR3 vs pan-FGFR",
    "status": "approved | phase 3 | phase 2 | phase 1 | preclinical | investigational",
    "effectiveness": 0.0-1.0, "safety": 0.0-1.0,
    "rationale": "one or two sentences: why it could work + selectivity reasoning"}}
]}}"""

    dir_map = {t["symbol"]: t.get("direction") for t in targets_info}
    try:
        raw = provider.complete(_SYSTEM, prompt, temperature=0.0)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0) if match else raw)
    except Exception:
        return []

    known_norm = {_norm(n) for n in known_names}
    valid_symbols = {s.lower(): s for s in dir_map}  # only validated targets are allowed
    out: list[DrugCandidate] = []
    for c in data.get("candidates", [])[:max_n]:
        name = (c.get("name") or "").strip()
        # dedupe on a normalised form so "IN figratinib" / "infigratinib phosphate" don't slip past
        if not name or _norm(name) in known_norm:
            continue
        # HARD GUARDRAIL: the drug must hit one of our validated targets (no off-target drift)
        targets = [valid_symbols[t.lower()] for t in c.get("targets", []) if t and t.lower() in valid_symbols]
        if not targets:
            continue
        known_norm.add(_norm(name))
        eff = _clamp(c.get("effectiveness"), 0.6)
        safe = _clamp(c.get("safety"), 0.5)
        status = (c.get("status") or "investigational").strip()
        moa = f"{c.get('selectivity') or ''} {c.get('action') or ''}".strip()
        if targets:
            moa = f"{moa} of {', '.join(targets)}".strip()
        out.append(DrugCandidate(
            name=name,
            clinical_stage=status,
            mechanism_of_action=moa or None,
            action_type=(c.get("action") or "").upper() or None,
            direction="helps",
            via_targets=targets,
            effectiveness_score=round(eff, 3),
            safety_score=round(safe, 3),
            confidence=round(eff, 3),
            goal_fit=round(eff, 3),
            source="ai",
            prospective=True,
            rationale=no_em_dashes(c.get("rationale") or "") or None,
            labels=["AI-proposed", "Prospective"],
        ))
    return _verify(out, dir_map, provider)


_VERIFY_SYSTEM = (
    "You are a strict pharmacology fact-checker. Be conservative: mark something false if it is wrong "
    "OR you are not confident. Respond with STRICT JSON only."
)


def _verify(cands: list[DrugCandidate], dir_map: dict, provider: LLMProvider) -> list[DrugCandidate]:
    """Drop AI-proposed candidates that (a) don't really act on the target, or (b) push it the WRONG
    way for the goal. Catches hallucinated pairings AND direction errors (e.g. proposing the GH
    antagonist pegvisomant/SOMAVERT as a height-increasing agonist)."""
    items = []
    for c in cands:
        t = c.via_targets[0] if c.via_targets else ""
        need = dir_map.get(t)
        need_txt = ("increase/agonise it" if need == "up"
                    else "decrease/inhibit it" if need == "down" else "modulate it")
        items.append({"name": c.name, "target": t, "needed_effect": need_txt})
    if not items:
        return cands
    prompt = (
        "For each item, the goal requires the target's activity to be made: needed_effect. "
        "Set valid=true ONLY if BOTH are true: (1) the drug really acts on that target, and "
        "(2) its action pushes the target in the needed direction (an agonist increases, an "
        "inhibitor/antagonist decreases). Otherwise valid=false.\n"
        'Return JSON: {"results":[{"name":"...","target":"...","valid":true|false}]}\n\n'
        f"Items:\n{json.dumps(items)}"
    )
    try:
        raw = provider.complete(_VERIFY_SYSTEM, prompt, temperature=0.0)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        results = json.loads(match.group(0) if match else raw).get("results", [])
        valid = {(r.get("name", "").lower(), r.get("target", "")) for r in results if r.get("valid")}
    except Exception:
        return cands
    return [c for c in cands if (c.name.lower(), c.via_targets[0] if c.via_targets else "") in valid]


_SALT_WORDS = ("hydrochloride", "phosphate", "citrate", "sulfate", "sulphate", "calcium",
               "sodium", "hydrobromide", "mesylate", "esylate", "rinfabate", "valerate", "acetate")


def _norm(name: str) -> str:
    """Normalise a drug name for duplicate detection: lowercase, strip salt words,
    drop non-alphanumerics. So 'IN figratinib' and 'Infigratinib phosphate' -> 'infigratinib'."""
    n = name.lower()
    for w in _SALT_WORDS:
        n = n.replace(w, "")
    return "".join(ch for ch in n if ch.isalnum())


def _clamp(v, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default
