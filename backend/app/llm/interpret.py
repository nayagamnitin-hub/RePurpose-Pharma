"""Turn a free-text query into something the databases can act on — WITH direction.

A goal like "grow taller" implies not just *which* targets matter but *which way*
to push them: boost the GH/IGF axis, but suppress FGFR3 and estrogen-driven
growth-plate fusion. Encoding direction lets the pipeline keep agonists and drop
antagonists (or vice-versa), instead of blindly listing anything that touches a
relevant target.

With a live LLM this works for any query. Without one, a small heuristic map
covers common demo goals.
"""
from __future__ import annotations

import json
import re

from app.models.schemas import QueryInterpretation, TargetSpec

from .provider import LLMProvider


# --- heuristic fallback (symbol, direction, weight, role) -------------------
def _spec(sym, direction, weight, role):
    return TargetSpec(symbol=sym, direction=direction, weight=weight, role=role)


_HEURISTICS: list[dict] = [
    {
        "keywords": ("taller", "grow taller", "get taller", "increase height", "height",
                     "growth plate", "growth spurt", "short stature", "stature"),
        "goal_label": "increased height / linear growth",
        "rationale": "Linear growth is driven by the GH/IGF-1 axis acting on growth-plate "
                     "chondrocytes. To grow taller you want to BOOST GH/IGF-1 and CNP (NPR2) "
                     "signalling, while SUPPRESSING FGFR3 (which restrains chondrocytes) and "
                     "estrogen (which fuses the growth plate).",
        "specs": [
            _spec("GHR", "up", 1.0, "GH receptor — primary growth driver"),
            _spec("IGF1", "up", 1.0, "IGF-1 — main growth mediator"),
            _spec("IGF1R", "up", 0.9, "IGF-1 receptor"),
            _spec("GH1", "up", 0.9, "growth hormone"),
            _spec("NPR2", "up", 0.7, "CNP/NPR2 growth-plate signalling"),
            _spec("CTNNB1", "up", 0.4, "WNT/β-catenin bone formation"),
            _spec("AR", "up", 0.3, "androgens raise growth velocity (but hasten fusion)"),
            _spec("FGFR3", "down", 0.8, "FGFR3 restrains chondrocyte proliferation"),
            _spec("CYP19A1", "down", 0.6, "aromatase — makes estrogen that fuses the plate"),
            _spec("ESR1", "down", 0.6, "estrogen receptor — growth-plate fusion"),
        ],
        "goal_keywords": ["short stature", "idiopathic short stature", "growth failure",
                          "growth hormone deficiency", "growth retardation", "turner",
                          "noonan", "prader-willi", "small for gestational age",
                          "achondroplasia", "dwarfism", "skeletal dysplasia"],
    },
    {
        "keywords": ("hair loss", "hair regrowth", "hair growth", "regrow hair", "balding",
                     "bald", "alopecia", "thinning hair", "hair thinning"),
        "goal_label": "hair regrowth",
        "rationale": "Androgenetic alopecia is driven by DHT (5α-reductase → AR); WNT/β-catenin "
                     "and prostaglandin signalling drive the follicle growth phase.",
        "specs": [
            _spec("AR", "down", 1.0, "androgen receptor — DHT miniaturises follicles"),
            _spec("SRD5A2", "down", 0.9, "5α-reductase makes DHT"),
            _spec("WNT10B", "up", 0.6, "WNT drives anagen"),
            _spec("PTGFR", "up", 0.5, "prostaglandin F — hair growth"),
        ],
        "goal_keywords": ["alopecia", "hair loss", "androgenetic"],
    },
    {
        "keywords": ("lose weight", "weight loss", "obesity", "fat loss", "appetite"),
        "goal_label": "weight loss",
        "rationale": "Incretin (GLP1R/GIPR) and melanocortin (MC4R) agonism reduce appetite; "
                     "leptin signalling (LEPR) controls satiety.",
        "specs": [
            _spec("GLP1R", "up", 1.0, "GLP-1 receptor agonism curbs appetite"),
            _spec("GIPR", "up", 0.7, "GIP receptor"),
            _spec("MC4R", "up", 0.7, "melanocortin-4 satiety"),
            _spec("GCGR", "up", 0.5, "glucagon receptor — energy expenditure"),
        ],
        "goal_keywords": ["obesity", "weight", "overweight"],
    },
    {
        "keywords": ("muscle", "muscular", "build muscle", "hypertrophy", "stronger"),
        "goal_label": "muscle growth",
        "rationale": "Myostatin (MSTN via ACVR2B) limits muscle; androgens and IGF-1 promote "
                     "hypertrophy.",
        "specs": [
            _spec("MSTN", "down", 1.0, "myostatin caps muscle growth"),
            _spec("ACVR2B", "down", 0.8, "activin receptor relays myostatin"),
            _spec("AR", "up", 0.7, "androgens drive hypertrophy"),
            _spec("IGF1", "up", 0.6, "IGF-1 anabolic signalling"),
        ],
        "goal_keywords": ["muscular dystrophy", "muscle wasting", "cachexia", "sarcopenia"],
    },
]

_QUESTION_HINTS = ("how", "why", "what", "can you", "should", "best way", "ways to", "?")


def _looks_like_question(q: str) -> bool:
    ql = q.lower().strip()
    return any(ql.startswith(h) or h in ql for h in _QUESTION_HINTS) or len(q.split()) > 5


_SYSTEM = (
    "You are a biomedical query router. Given a user's plain-language query, decide how to "
    "search drug-target databases AND which direction to push each target. Respond with "
    "STRICT JSON only, no prose."
)

_PROMPT_TEMPLATE = """User query: {query}

Return JSON with this exact shape:
{{
  "mode": "disease" | "targets",
  "disease_terms": [ up to 3 standard disease names, or [] ],
  "goal_label": "short phrase naming the desired effect, e.g. 'increased height'",
  "target_specs": [
    {{"symbol": "FGFR3", "direction": "up"|"down", "weight": 0.0-1.0, "role": "one phrase"}}
  ],
  "goal_keywords": [ disease/indication terms that would mean a drug is ALREADY established for this goal ]
}}

Rules:
- "direction" = which way to push the target to ACHIEVE THE GOAL. "up" means you want MORE of
  that target's activity (so an AGONIST helps); "down" means LESS (so an INHIBITOR/ANTAGONIST helps).
- If the query names a disease, use mode "disease" and fill disease_terms (target_specs may be []).
- If it's a goal/phenotype, use mode "targets" and be THOROUGH: list 8-15 relevant human targets.
  Include BOTH the signalling ligands AND their receptors (e.g. for a hormone axis include both the
  hormone gene and its receptor — GH1 AND GHR, IGF1 AND IGF1R), plus upstream/downstream modulators
  (e.g. enzymes, nuclear receptors) that influence the phenotype.
- Drugs usually act on receptors/enzymes, so always include the receptor a drug would target.
- Use official HGNC gene symbols. Only include mechanistically relevant targets."""


def interpret_query(query: str, provider: LLMProvider) -> QueryInterpretation:
    curated = _match_heuristic(query)
    if provider.live:
        try:
            llm = _interpret_with_llm(query, provider)
            # combine LLM generality with curated completeness when we have domain knowledge
            return _merge(llm, curated) if (curated and llm.mode == "targets") else llm
        except Exception:
            pass
    return _build_from_curated(query, curated) if curated else _literal(query)


def _merge(llm: QueryInterpretation, curated: dict) -> QueryInterpretation:
    by_symbol = {s.symbol: s for s in llm.target_specs}
    # Curated specs are hand-verified, so they OVERRIDE the LLM for shared targets (correct
    # direction matters: e.g. for hair regrowth AR must be "down"/blocked, so an AR-agonist
    # SARM is correctly rejected) and add any targets the LLM missed.
    for spec in curated["specs"]:
        by_symbol[spec.symbol] = spec
    llm.target_specs = list(by_symbol.values())
    llm.goal_keywords = sorted(set(llm.goal_keywords) | set(curated["goal_keywords"]))
    llm.goal_label = llm.goal_label or curated["goal_label"]
    return llm


def _interpret_with_llm(query: str, provider: LLMProvider) -> QueryInterpretation:
    raw = provider.complete(_SYSTEM, _PROMPT_TEMPLATE.format(query=query), temperature=0.0)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    data = json.loads(match.group(0) if match else raw)
    mode = "targets" if data.get("mode") == "targets" else "disease"
    specs = [
        TargetSpec(
            symbol=s["symbol"],
            direction="down" if str(s.get("direction", "up")).lower() == "down" else "up",
            weight=float(s.get("weight", 0.5)),
            role=s.get("role"),
        )
        for s in data.get("target_specs", []) if s.get("symbol")
    ][:15]
    return QueryInterpretation(
        original=query,
        mode=mode,
        disease_terms=[d for d in data.get("disease_terms", []) if d][:3],
        target_specs=specs,
        goal_keywords=[k.lower() for k in data.get("goal_keywords", []) if k],
        goal_label=data.get("goal_label"),
        rationale=data.get("rationale"),
        source="llm",
    )


def _match_heuristic(query: str) -> dict | None:
    ql = query.lower()
    for h in _HEURISTICS:
        # word-boundary match so e.g. "grow" can't match "reGROW", "tall" can't match "instTALL"
        if any(re.search(r"\b" + re.escape(k) + r"\b", ql) for k in h["keywords"]):
            return h
    return None


def _build_from_curated(query: str, h: dict) -> QueryInterpretation:
    return QueryInterpretation(
        original=query, mode="targets", target_specs=h["specs"],
        goal_keywords=h["goal_keywords"], goal_label=h["goal_label"],
        rationale=h["rationale"], source="heuristic",
    )


def _literal(query: str) -> QueryInterpretation:
    return QueryInterpretation(
        original=query, mode="disease", disease_terms=[query],
        rationale=("No live AI key, so this was searched literally."
                   if _looks_like_question(query) else None),
        source="literal",
    )
