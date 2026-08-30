"""Ask-the-AI about a drug (or compare drugs), grounded in literature + mechanism.

Evidence comes from PubMed snippets when available; the model cites PMIDs for those.
When literature is thin (common for prospective/repurposing uses), the model may also
reason from established pharmacology — but must clearly label that as mechanistic
reasoning rather than cited evidence, and never fabricate statistics or PMIDs.
"""
from __future__ import annotations

from app.clients.ncbi import NcbiClient, extract_text_from_bioc
from app.models.schemas import LiteratureRef
from app.textutil import no_em_dashes

from .provider import PLACEHOLDER_BANNER, LLMProvider

_SYSTEM = (
    "You are a careful biomedical research assistant for a drug-REPURPOSING hypothesis tool. "
    "Use the provided study snippets as primary evidence and cite their PMIDs like (PMID 12345678) "
    "whenever you rely on them. When the snippets don't cover the question (common for novel or "
    "repurposing uses), you MAY reason from well-established pharmacology and mechanism — but clearly "
    "prefix that with 'Mechanistic reasoning:' so it is not mistaken for cited evidence. Always "
    "distinguish established clinical evidence from hypothesis, and flag where data is missing. "
    "Never fabricate specific numbers, trials, or PMIDs. This is research hypothesis generation, "
    "not medical advice. NEVER use em dashes or en dashes; use commas, periods, or 'and'."
)


def fetch_studies(term: str, retmax: int = 8, with_snippets: int = 4) -> list[LiteratureRef]:
    """PubMed studies for a term (keep the term clean — drug + topic, not the question)."""
    refs: list[LiteratureRef] = []
    with NcbiClient() as ncbi:
        try:
            pmids = ncbi.search_pmids(term, retmax=retmax)
        except Exception:
            return refs
        try:
            titles = ncbi.esummary_titles(pmids)  # clean title for every study, one call
        except Exception:
            titles = {}
        for i, pmid in enumerate(pmids):
            snippet = None
            if i < with_snippets:  # snippets are for the AI's evidence, not for display
                try:
                    snippet = extract_text_from_bioc(ncbi.fetch_abstract_bioc(pmid))[:600]
                except Exception:
                    pass
            refs.append(LiteratureRef(pmid=pmid, title=titles.get(pmid), snippet=snippet))
    return refs


def answer_question(
    question: str,
    subjects: list[str],
    studies: list[LiteratureRef],
    provider: LLMProvider,
    context: str = "",
) -> str:
    evidence = "\n\n".join(f"PMID {r.pmid}: {r.snippet}" for r in studies if r.snippet)
    pmid_list = ", ".join(r.pmid for r in studies[:8])

    if not provider.live:
        return (
            f"{PLACEHOLDER_BANNER}\n\n**Q:** {question}\n**About:** {', '.join(subjects)}\n\n"
            f"A live model would synthesize a cited, mechanism-aware answer. "
            f"Retrieved studies: PMID {pmid_list or 'none found'}."
        )

    prompt = (
        f"Question: {question}\n\n"
        f"Drug(s): {', '.join(subjects)}\n"
        + (f"Context: {context}\n" if context else "")
        + "\nStudy snippets (may be empty):\n"
        + (evidence or "(no relevant abstracts were retrieved)")
        + (f"\n\nOther PMIDs found (titles not fetched): {pmid_list}" if pmid_list and not evidence else "")
        + "\n\nAnswer the question. Cite PMIDs for anything drawn from the snippets. If the snippets "
          "don't cover it, give your best mechanistic answer labelled 'Mechanistic reasoning:' and state "
          "what evidence would be needed. If comparing drugs, contrast them point by point (efficacy, "
          "selectivity, safety, evidence maturity) and note where one is only a prospective hypothesis."
    )
    return no_em_dashes(provider.complete(_SYSTEM, prompt))
