"""Clickable demo UI for the AI Drug Repurposing Research Assistant.

Run from the backend/ directory:
    streamlit run streamlit_app.py
Calls the pipeline in-process (no separate API server needed).
"""
from __future__ import annotations

import streamlit as st

from app.llm.provider import get_provider
from app.llm.qa import answer_question, fetch_studies
from app.models.schemas import DrugCandidate
from app.services.pipeline import build_report

st.set_page_config(page_title="AI Drug Repurposing Assistant", page_icon="🧬", layout="wide")

LABEL_ICON = {
    "Most effective": "💪", "Safest": "🛡️", "Low side effects": "✅",
    "Most side effects": "⚠️", "Effective but harsh": "💥",
    "Approved": "🟢", "Experimental": "🧪", "Right direction": "🎯",
    "Black-box warning": "⛔", "Withdrawn": "🚫",
    "AI-proposed": "🔮", "Prospective": "🔭", "In clinical trials": "🧫",
}

EXAMPLES = [
    "Alzheimer disease", "Parkinson disease",
    "how to grow taller in puberty", "how to lose weight",
]


@st.cache_data(show_spinner=False)
def run_analysis(query: str):
    return build_report(query)


@st.cache_data(show_spinner=False)
def load_studies(drug: str, topic: str = ""):
    """Studies for a drug. Tries 'drug + topic'; if that's empty (e.g. the topic was a
    long free-text goal), falls back to the drug name alone so studies always load."""
    topic = (topic or "").strip()
    refs = fetch_studies(f"{drug} {topic}".strip(), retmax=8, with_snippets=4)
    if not refs and topic:
        refs = fetch_studies(drug.strip(), retmax=8, with_snippets=4)
    return refs


def badges(labels: list[str]) -> str:
    return "  ".join(f"{LABEL_ICON.get(l, '🏷️')} {l}" for l in labels)


def render_drug(c: DrugCandidate, context: str, study_topic: str, provider) -> None:
    flag = "🔭 " if c.prospective else ""
    headline = c.labels[0] if c.labels else (c.clinical_stage or "")
    title = f"{flag}**{c.name}**  ·  {LABEL_ICON.get(headline, '')} {headline}  ·  confidence {int(c.confidence * 100)}%"
    with st.expander(title):
        if c.source == "clinical_trials":
            st.info("🧫 **In clinical trials** — a real investigational drug currently being studied for "
                    "this area (from ClinicalTrials.gov), not yet an approved/established treatment.")
        elif c.prospective:
            st.info("🔭 **Prospective / AI-proposed** — a mechanistic hypothesis, not yet established in the "
                    "drug databases for this use. Scores are AI estimates.")
        if c.rationale:
            st.markdown(f"**Why it could work:** {c.rationale}")
        if c.trial_ids:
            links = "  ·  ".join(f"[{nct}](https://clinicaltrials.gov/study/{nct})" for nct in c.trial_ids)
            st.markdown(f"**Clinical trials:** {links}")
        if c.labels:
            st.markdown(badges(c.labels))
        dir_icon = {"helps": "✅ pushes targets the right way", "opposite": "⛔ acts in the opposite direction",
                    "unclear": "❔ direction unclear"}.get(c.direction, "")
        st.caption(f"Action: {c.action_type or 'n/a'} — {dir_icon}")
        st.caption(f"Mechanism: {c.mechanism_of_action or 'n/a'}  ·  Type: {c.drug_type or 'n/a'}  ·  Stage: {c.clinical_stage or 'n/a'}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Effectiveness", f"{int(c.effectiveness_score * 100)}%")
        m2.metric("Safety", f"{int(c.safety_score * 100)}%")
        m3.metric("Confidence", f"{int(c.confidence * 100)}%")

        st.write(f"**Acts via targets:** {', '.join(c.via_targets) or 'n/a'}")
        st.write(f"**Already studied/used for:** {', '.join(c.known_for[:6]) or 'n/a'}")
        if c.top_adverse_events:
            st.write(f"**Notable adverse events** ({c.adverse_event_types} types, {c.total_adverse_reports} reports): "
                     + ", ".join(c.top_adverse_events))
        if c.warnings:
            st.warning("Warnings: " + ", ".join(c.warnings))
        if c.chembl_id:
            st.markdown(f"[ChEMBL record](https://www.ebi.ac.uk/chembl/explore/compound/{c.chembl_id})")

        # --- studies (note: button key must differ from the storage key) ---
        if st.button("📚 Load studies", key=f"btn_studies_{c.name}"):
            st.session_state[f"studies_{c.name}"] = load_studies(c.name, study_topic)
        studies = st.session_state.get(f"studies_{c.name}")
        if studies is not None:
            if studies:
                for ref in studies:
                    st.markdown(f"- [PMID {ref.pmid}](https://pubmed.ncbi.nlm.nih.gov/{ref.pmid}/)"
                                + (f" — {ref.snippet[:180]}…" if ref.snippet else ""))
            else:
                st.caption("No PubMed studies found for this drug.")

        # --- ask the AI ---
        st.markdown("**🤖 Ask the AI about this drug**")
        q = st.text_input("Your question", key=f"q_{c.name}",
                          placeholder=f"e.g. Is there evidence {c.name} helps with {context}?",
                          label_visibility="collapsed")
        if st.button("Ask", key=f"ask_{c.name}") and q:
            with st.spinner("Reading the literature…"):
                # keep the PubMed search clean (drug + short topic) — NOT the raw question
                qa_studies = load_studies(c.name, study_topic)
                drug_ctx = f"{c.name}: {c.mechanism_of_action or ''}; goal: {context}"
                ans = answer_question(q, [c.name], qa_studies, provider, context=drug_ctx)
                st.session_state[f"ans_{c.name}"] = ans
        if st.session_state.get(f"ans_{c.name}"):
            st.info(st.session_state[f"ans_{c.name}"])


# ===========================================================================
st.title("🧬 AI Drug Repurposing Research Assistant")
st.caption("Search a disease **or a plain-language goal** to explore existing drugs that might be repurposed. "
           "**Research / hypothesis-generation tool only — not medical advice.**")

provider = get_provider()
ai_state = "🟢 live (" + provider.kind + ")" if provider.live else f"⚪ placeholder ({provider.kind})"

with st.sidebar:
    st.header("Search")
    query = st.text_input("Disease, target, or goal", value=st.session_state.get("query", "Alzheimer disease"))
    st.caption("Examples:")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            query = ex
            st.session_state["query"] = ex
            st.session_state["go"] = True
    go = st.button("🔍 Analyze", type="primary", use_container_width=True)
    per_column = st.slider("Max drugs shown per column", 3, 50, 12,
                          help="Targets are scanned automatically; this only limits how many drugs are listed.")
    st.divider()
    st.caption(f"AI: {ai_state}")
    if not provider.live:
        st.caption("Add a valid `AIza…` Gemini key to `.env` to enable real AI answers.")

if go or st.session_state.pop("go", False):
    st.session_state["query"] = query
    if not query or len(query) < 2:
        st.warning("Enter a disease or goal.")
        st.stop()
    try:
        with st.spinner(f"Analyzing “{query}” across Open Targets, UniProt & PubMed…"):
            st.session_state["report"] = run_analysis(query)
    except Exception as e:
        st.error(f"Something went wrong while analyzing (often a brief upstream API hiccup). "
                 f"Please click **Analyze** again.\n\nDetails: {type(e).__name__}: {e}")
        st.stop()

report = st.session_state.get("report")
if not report:
    st.info("← Enter a disease or goal in the sidebar and click **Analyze**.")
    st.stop()

# ---- interpretation banner ----
interp = report.interpretation
if interp and interp.mode == "targets":
    st.success(f"Understood **“{report.query}”** as a goal → scanning targets: **{', '.join(interp.target_symbols)}**")
    if interp.rationale:
        st.caption(interp.rationale)
elif report.disease_name:
    st.success(f"**{report.disease_name}**  ·  `{report.disease_id}`")
for note in report.notes:
    st.caption("ℹ️ " + note)

if not report.targets:
    st.error("Couldn't resolve any biological targets for this query. Try rephrasing.")
    st.stop()

# ---- AI summary ----
if report.summary:
    st.subheader("🤖 AI research summary")
    st.markdown(report.summary)

context = report.disease_name or report.query
# a SHORT, clean topic for PubMed searches (the raw free-text query returns no hits)
study_topic = report.disease_name or (report.interpretation.goal_label if report.interpretation else None) or ""

# ---- two columns ----
left, right = st.columns(2)
with left:
    st.subheader(f"💊 Existing / established ({len(report.existing_solutions)})")
    st.caption("Drugs already established/tested for this goal.")
    for c in report.existing_solutions[:per_column]:
        render_drug(c, context, study_topic, provider)
with right:
    st.subheader(f"🔬 Repurposing candidates ({len(report.repurposing_candidates)})")
    st.caption("Drugs from other uses that push the same targets the right way — ranked by confidence.")
    for c in report.repurposing_candidates[:per_column]:
        render_drug(c, context, study_topic, provider)

# ---- opposite-effect transparency ----
if report.opposite_effect:
    with st.expander(f"⛔ Excluded — act in the OPPOSITE direction ({len(report.opposite_effect)})"):
        st.caption("These hit relevant targets but the wrong way for this goal, so they were filtered out.")
        for c in report.opposite_effect:
            st.markdown(f"- **{c.name}** — {c.action_type} of {', '.join(c.via_targets)}  ·  {c.mechanism_of_action or ''}")

# ---- compare ----
st.divider()
st.subheader("⚖️ Compare treatments with AI")
all_drugs = [c.name for c in report.existing_solutions + report.repurposing_candidates]
picks = st.multiselect("Pick drugs to compare", all_drugs, max_selections=4)
cq = st.text_input("Comparison question", placeholder="e.g. Which has more evidence and fewer side effects for this use?")
if st.button("Compare") and picks and cq:
    with st.spinner("Gathering evidence for each…"):
        studies = []
        for d in picks:
            studies += load_studies(d, study_topic)
        st.info(answer_question(cq, picks, studies, provider, context=f"goal: {context}"))

# ---- targets ----
st.divider()
st.subheader(f"🎯 Targets scanned ({len(report.targets)})")
st.dataframe(
    [{"Gene": t.symbol, "Relevance": round(t.association_score, 3), "Protein": t.name, "UniProt": t.uniprot_id or ""}
     for t in report.targets],
    use_container_width=True, hide_index=True,
)
with st.expander("Protein functions (top targets)"):
    for t in report.targets:
        if t.protein_function:
            st.markdown(f"**{t.symbol}** — {t.protein_function[:500]}…")
