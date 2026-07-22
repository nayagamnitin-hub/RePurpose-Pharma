"""Pydantic response models — the contract the frontend consumes."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Target(BaseModel):
    ensembl_id: str
    symbol: str
    name: str | None = None
    association_score: float = 0.0
    # filled from UniProt when available
    uniprot_id: str | None = None
    protein_function: str | None = None


class DrugCandidate(BaseModel):
    name: str
    chembl_id: str | None = None
    drug_type: str | None = None
    is_approved: bool | None = None
    clinical_stage: str | None = None          # e.g. "APPROVAL", "PHASE_3"
    max_phase: float | None = None             # numeric: 4=approved .. 0=preclinical
    mechanism_of_action: str | None = None
    action_type: str | None = None             # AGONIST / INHIBITOR / ANTAGONIST ...
    # whether the drug pushes the targets toward the user's goal
    direction: str = "unclear"                 # "helps" | "opposite" | "unclear"
    goal_fit: float = 0.0                      # 0-1 directional fit with the goal

    # provenance: "database" (Open Targets) | "clinical_trials" | "ai" (AI-proposed)
    source: str = "database"
    prospective: bool = False                  # not yet an established/approved use
    rationale: str | None = None               # AI mechanistic reasoning (for AI leads)
    trial_ids: list[str] = []                  # ClinicalTrials.gov NCT ids (for trial-backed leads)
    variants: list[str] = []                   # merged near-identical forms (e.g. insulin lispro/aspart)

    # the disease target(s) this drug connects through
    via_targets: list[str] = []
    # diseases this drug is already approved/studied for
    known_for: list[str] = []

    # safety signals (from Open Targets / FAERS)
    adverse_event_types: int = 0               # # of significant adverse-event types
    total_adverse_reports: int = 0             # summed report counts
    top_adverse_events: list[str] = []
    has_black_box: bool = False
    is_withdrawn: bool = False
    warnings: list[str] = []

    # derived 0-1 scores used for ranking and labels
    effectiveness_score: float = 0.0
    safety_score: float = 0.0
    confidence: float = 0.0
    # human-readable badges: "Most effective", "Safest", etc.
    labels: list[str] = []

    # PubMed studies for this drug (filled on demand)
    pmids: list[str] = []


class LiteratureRef(BaseModel):
    pmid: str
    pmcid: str | None = None
    title: str | None = None
    snippet: str | None = None


class TargetSpec(BaseModel):
    """A target plus the direction we want to push it for the user's goal."""
    symbol: str
    direction: str = "up"                      # "up" = want more activity, "down" = less
    weight: float = 0.5                        # importance of this target to the goal (0-1)
    role: str | None = None                    # short biology note


class QueryInterpretation(BaseModel):
    """How a free-text query was understood before hitting the databases."""
    original: str
    mode: str = "disease"                      # "disease" | "targets"
    disease_terms: list[str] = []              # disease names to resolve
    target_specs: list[TargetSpec] = []        # targets + desired direction (goal mode)
    goal_keywords: list[str] = []              # indication terms that mean "established for this goal"
    goal_label: str | None = None              # e.g. "increased height"
    rationale: str | None = None
    source: str = "literal"                    # "llm" | "heuristic" | "literal"

    @property
    def target_symbols(self) -> list[str]:
        return [s.symbol for s in self.target_specs]


class RepurposingReport(BaseModel):
    query: str
    interpretation: QueryInterpretation | None = None
    disease_id: str | None = None
    disease_name: str | None = None

    targets: list[Target] = []
    existing_solutions: list[DrugCandidate] = Field(default_factory=list)
    repurposing_candidates: list[DrugCandidate] = Field(default_factory=list)
    # drugs that act in the OPPOSITE direction to the goal (filtered out, kept for transparency)
    opposite_effect: list[DrugCandidate] = Field(default_factory=list)

    summary: str | None = None
    # short note on whether the condition has a cure vs only slowing/managing it
    overview_note: str | None = None
    notes: list[str] = []
