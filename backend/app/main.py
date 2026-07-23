"""FastAPI entrypoint — serves the web UI and the JSON API.

Run from the backend/ directory:
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/  (UI)  or  /docs  (API).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.llm.provider import get_provider
from app.llm.qa import answer_question, fetch_studies
from app.models.schemas import LiteratureRef, RepurposingReport
from app.services.pipeline import build_report

app = FastAPI(
    title="AI Drug Repurposing Research Assistant",
    version="1.0.0",
    description="Hypothesis-generation tool: condition/goal -> targets -> existing + investigational drugs -> evidence.",
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.middleware("http")
async def _no_cache(request, call_next):
    """Tell browsers not to cache the UI, so an updated app.js/index.html never gets stuck
    on a stale cached version (which can break the page)."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


# ---- API -----------------------------------------------------------------
@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/repurpose", response_model=RepurposingReport)
def repurpose(q: str = Query(..., min_length=2, description="Disease, target, or plain-language goal")) -> RepurposingReport:
    report = build_report(q)
    if not report.targets:
        raise HTTPException(status_code=404, detail=f"Could not resolve any targets for '{q}'.")
    return report


def _studies(drug: str, topic: str) -> list[LiteratureRef]:
    """drug + topic, falling back to the drug name alone so studies always load."""
    topic = (topic or "").strip()
    refs = fetch_studies(f"{drug} {topic}".strip(), retmax=8, with_snippets=4)
    if not refs and topic:
        refs = fetch_studies(drug.strip(), retmax=8, with_snippets=4)
    return refs


@app.get("/api/studies")
def studies(drug: str = Query(..., min_length=1), topic: str = Query("")) -> dict[str, list[LiteratureRef]]:
    try:
        return {"studies": _studies(drug, topic)}
    except Exception:
        return {"studies": []}


class AskRequest(BaseModel):
    drug: str
    question: str
    context: str = ""
    topic: str = ""


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    provider = get_provider()
    try:
        refs = _studies(req.drug, req.topic)
        answer = answer_question(req.question, [req.drug], refs, provider, context=req.context)
    except Exception:
        answer = "Sorry, the AI could not answer just now (a brief service hiccup). Please try again."
    return {"answer": answer, "live": provider.live}


class ExplainRequest(BaseModel):
    drug: str
    label: str
    goal: str = ""
    mechanism: str = ""
    stage: str = ""
    known_for: list[str] = []
    warnings: list[str] = []
    effectiveness: int = 0
    safety: int = 0
    confidence: int = 0
    established: bool = False


_EXPLAIN_SYS = (
    "You explain, in 2 to 3 short plain-language sentences, what a specific label or score means for a "
    "specific drug in a drug-repurposing tool. Be concrete and factual. Do NOT invent exact approval years, "
    "trial names, or statistics; if you are not sure of a specific detail, speak in accurate general terms. "
    "No em dashes."
)


@app.post("/api/explain")
def explain(req: ExplainRequest) -> dict:
    from app.textutil import no_em_dashes
    provider = get_provider()
    kind = ("established/approved treatment" if req.established else "repurposing / investigational candidate")
    facts = (
        f"Drug: {req.drug}\nGoal being explored: {req.goal or 'n/a'}\nThis drug is a {kind} for the goal.\n"
        f"Mechanism: {req.mechanism or 'n/a'}\nDevelopment stage: {req.stage or 'n/a'}\n"
        f"Known uses: {', '.join(req.known_for) or 'n/a'}\nWarnings: {', '.join(req.warnings) or 'none'}\n"
        f"Scores shown to the user: effectiveness {req.effectiveness}%, safety {req.safety}%, confidence {req.confidence}%.\n\n"
        f"The user clicked the label: '{req.label}'. Explain what THIS label/score means for THIS drug:\n"
        "- 'Approved': what it is approved to treat and that approval means regulators judged trials to show benefit outweighing risks.\n"
        "- 'Black-box warning': what a black-box warning is and, if known, the general safety reason this drug carries one.\n"
        "- a Confidence score: explain that confidence reflects how strong the case is that this drug helps the goal. For an "
        "established treatment it is high because it is a proven, approved option. For a REPURPOSING candidate, be careful NOT to "
        "imply proven clinical data. Frame it as MECHANISTIC PLAUSIBILITY (it acts on a relevant target/pathway) and a hypothesis "
        "needing more research, not established proof. Tell the user they can click 'Learn More' to see the actual studies.\n"
        "- Effectiveness or Safety score: explain what it estimates and why it is around this level for this drug.\n"
        "- 'Most effective' / 'Safest' / 'Most side effects' / 'Low side effects': explain it is a comparison within this list.\n"
        "- otherwise: explain the label plainly."
    )
    try:
        return {"explanation": no_em_dashes(provider.complete(_EXPLAIN_SYS, facts, temperature=0.2))}
    except Exception:
        return {"explanation": "Couldn't load an explanation just now. Please try again."}


class EvidenceRequest(BaseModel):
    drug: str
    goal: str = ""
    established: bool = False
    mechanism: str = ""
    targets: list[str] = []


@app.post("/api/evidence")
def evidence(req: EvidenceRequest) -> dict:
    """'Learn More' on a confidence score: real PubMed studies + an evidence-grounded summary."""
    from app.textutil import no_em_dashes
    provider = get_provider()
    refs = _studies(req.drug, req.goal)
    snippets = "\n\n".join(f"PMID {r.pmid}: {r.snippet}" for r in refs if r.snippet)

    if req.established:
        task = ("This drug is an established/approved treatment for the goal. In 3 to 4 sentences say what it "
                "is used for and what it has been shown to do, citing PMIDs from the snippets. You may note it "
                "is an approved/standard option, but do NOT invent a specific approval date or trial name.")
    else:
        task = (f"This is a REPURPOSING candidate. In 3 to 4 sentences give the evidence-based rationale: it acts "
                f"on {', '.join(req.targets) or 'its target'} ({req.mechanism or 'see mechanism'}); explain that this "
                f"target or pathway is relevant to the goal and that this makes it a plausible candidate. Cite PMIDs "
                f"from the snippets where relevant. Be explicit where evidence is limited or only mechanistic (a "
                f"hypothesis), rather than overclaiming.")
    system = ("You summarize the REAL evidence for a drug relative to a goal, for a repurposing tool. Use ONLY the "
              "provided study snippets for factual claims and cite them like (PMID 12345678). If snippets are thin, "
              "say the evidence is limited instead of overclaiming. No em dashes.")
    prompt = f"Drug: {req.drug}\nGoal: {req.goal}\n\nStudy snippets:\n{snippets or '(none retrieved)'}\n\n{task}"
    try:
        summary = no_em_dashes(provider.complete(system, prompt, temperature=0.2))
    except Exception:
        summary = "Couldn't load the evidence summary just now."
    return {"summary": summary, "studies": refs}


class ChatRequest(BaseModel):
    message: str
    sessionId: str = "web"
    context: str = ""


def _study_context(message: str) -> tuple[str | None, str | None]:
    """If the user references a paper (PMCID / PMID / DOI / PubMed URL), fetch its ACTUAL text
    from NCBI (legal: PubMed abstract + PMC open-access full text). We always resolve to a PMID
    first, because a PMID always has a fetchable abstract, so the AI never has to guess."""
    from app.clients.ncbi import NcbiClient, extract_text_from_bioc

    pmc = re.search(r"PMC\d+", message, re.I)
    pmid = re.search(r"(?:PMID[:\s]*|pubmed\.ncbi\.nlm\.nih\.gov/)\s*(\d{5,9})", message, re.I)
    doi = re.search(r"\b10\.\d{4,9}/[^\s\"'<>)\]]+", message)
    if not (pmc or pmid or doi):
        return None, None

    ncbi = NcbiClient()
    try:
        pmcid = pmc.group(0).upper() if pmc else None
        target_pmid = pmid.group(1) if pmid else None
        # resolve PMCID or DOI down to a PMID so we can always get the abstract
        if not target_pmid and pmcid:
            try:
                target_pmid = ncbi.pmcid_to_pmid(pmcid)
            except Exception:
                pass
        if not target_pmid and doi:
            try:
                ids = ncbi.search_pmids(f"{doi.group(0)}[DOI]", retmax=1)
                target_pmid = ids[0] if ids else None
            except Exception:
                pass

        label = pmcid or (f"PMID {target_pmid}" if target_pmid else "the study")
        text = None
        # prefer richer PMC open-access full text if available
        if pmcid:
            try:
                text = extract_text_from_bioc(ncbi.fetch_fulltext_bioc(pmcid))
            except Exception:
                text = None
        # reliable fallback: the abstract (always available for a valid PMID)
        if not text and target_pmid:
            try:
                text = extract_text_from_bioc(ncbi.fetch_abstract_bioc(target_pmid))
            except Exception:
                text = None
        return (label, text[:3500]) if text else (None, None)
    finally:
        ncbi.close()


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    """Proxy to the custom n8n 'RePurpose Pharma Chatbot' webhook (keeps the URL server-side, no CORS)."""
    from app.config import settings
    url = settings.n8n_webhook_url
    if not url:
        return {"reply": "The custom assistant isn't connected yet. Add your n8n webhook URL as the "
                         "N8N_WEBHOOK_URL setting and I'll come to life here."}

    parts = []
    goal = (req.context or "").strip()
    if goal:
        parts.append(f"[Context: the user is exploring drug REPURPOSING for '{goal}' and is viewing those results.]")
    try:
        label, study = _study_context(req.message)
    except Exception:
        label, study = None, None
    if study:
        goal_phrase = f"the user's goal ({goal})" if goal else "other conditions or goals"
        parts.append(
            f"The user referenced a study ({label}). Its ACTUAL text from PubMed/PMC is below. Base factual "
            f"claims ONLY on this text (do not guess from the title).\n\"\"\"\n{study}\n\"\"\"\n\n"
            f"REPURPOSING ANALYSIS (required, this is a drug-repurposing tool): Do NOT simply say the study "
            f"does not mention {goal or 'the goal'} and stop. Instead: (1) briefly state what the study actually "
            f"shows (mechanism, targets, effects); (2) then REASON mechanistically about whether the drug could be "
            f"repurposed for {goal_phrase}, giving a clear 'it could plausibly help because...' OR 'it likely could "
            f"not help because...' with the biological rationale (shared targets/pathways, effects on relevant "
            f"mechanisms), even though this study does not test that use. Clearly label the repurposing reasoning "
            f"as a hypothesis for further research."
        )
    parts.append(req.message)
    message = "\n\n".join(parts)

    from app.clients.base import make_client
    for _ in range(2):  # retry once on a transient n8n/network hiccup
        try:
            with make_client() as http:
                r = http.post(url, json={"message": message, "sessionId": req.sessionId}, timeout=120.0)
                r.raise_for_status()
                data = r.json()
            raw = data.get("reply") or data.get("output") or data.get("text") or ""
            return {"reply": _strip_thinking(raw)}
        except Exception:
            pass
    return {"reply": "Sorry, I couldn't reach the assistant just now. Please try again in a moment."}


def _strip_thinking(reply: str) -> str:
    """Remove reasoning-model 'thinking' so the user only sees the final answer."""
    if not reply:
        return "(the assistant returned no text)"
    # well-formed: <think>...</think> followed by the answer
    reply = re.sub(r"(?is)<think>.*?</think>\s*", "", reply)
    if "</think>" in reply:
        reply = reply.split("</think>")[-1]
    # malformed: an unclosed <think> means the answer was cut off inside the reasoning
    if re.search(r"(?i)<think>", reply):
        return ("The assistant's answer got cut off while reasoning. Please try again or rephrase. "
                "(Tip: switching the n8n Groq model to a non-reasoning model like "
                "llama-3.3-70b-versatile gives clean, complete answers.)")
    return reply.strip() or "(the assistant returned no text)"


class DrugInfoRequest(BaseModel):
    name: str
    mechanism: str = ""
    targets: list[str] = []
    known_for: list[str] = []
    clinical_stage: str = ""
    source: str = ""
    prospective: bool = False
    goal: str = ""
    adverse_events: list[str] = []
    variants: list[str] = []


_DRUG_SYS = (
    "You explain a drug to a curious general reader. Use the structured facts and reference text "
    "provided. You MUST ALWAYS produce a useful explanation in COMPLETE sentences. NEVER say you are "
    "unfamiliar with the drug; synthesize from the facts given. Follow this template:\n"
    "- If the drug is ALREADY established/used FOR the user's goal: say it is used for [goal], what kind "
    "of drug it is, and that it works by acting on [target].\n"
    "- If it is being REPURPOSED (normally used for something else): say what it is normally used for, "
    "that it acts on [target] which is ALSO involved in [goal], so it is being explored for [goal], and "
    "note whether it is in clinical trials and the stage.\n"
    "Mention 1-2 notable side effects if provided. Write 3-5 complete sentences, plain language, name the "
    "target plainly, no heavy gene jargon, do not invent statistics. NEVER use em dashes or en dashes; "
    "use commas, periods, or 'and'. After the explanation, on a SEPARATE final line write exactly "
    "'BRAND: <single best-known trade name, or none>' (e.g. lisdexamfetamine -> Vyvanse, semaglutide -> Ozempic)."
)


def _fallback_summary(req: DrugInfoRequest, description: str) -> str:
    from app.textutil import trim_to_sentence
    if description:
        return trim_to_sentence(description, 500)
    used = ", ".join(req.known_for) or "various conditions"
    s = f"{req.name} is a drug studied for {used}."
    if req.targets:
        tgt = ", ".join(req.targets)
        s += f" It acts on {tgt}, which is also relevant to {req.goal}." if req.goal else f" It acts on {tgt}."
    if req.clinical_stage:
        s += f" It is at the {req.clinical_stage} stage of development."
    return s


@app.post("/api/drug")
def drug(req: DrugInfoRequest) -> dict:
    from app.clients.druginfo import drug_profile, wiki_image
    from app.textutil import no_em_dashes

    info = drug_profile(req.name)
    provider = get_provider()
    info["brand"] = ""
    info["summary"] = _fallback_summary(req, info.get("description") or "")

    if provider.live:
        established = (not req.prospective) and req.source not in ("clinical_trials", "ai")
        facts = (
            f"Drug: {req.name}\n"
            f"Best classified as: {'an established treatment for the goal' if established else 'a repurposing / investigational candidate'}\n"
            f"User's goal/condition: {req.goal or 'n/a'}\n"
            f"Biological target(s) it acts on: {', '.join(req.targets) or 'n/a'}\n"
            f"Mechanism note: {req.mechanism or 'n/a'}\n"
            f"Normally used / studied for: {', '.join(req.known_for) or 'n/a'}\n"
            f"Development stage: {req.clinical_stage or 'n/a'}\n"
            f"Notable side effects: {', '.join(req.adverse_events) or 'n/a'}\n"
            f"Encyclopedia reference (may be empty):\n{(info.get('description') or '')[:1000]}"
        )
        try:
            raw = provider.complete(_DRUG_SYS, facts, temperature=0.2)
            brand_m = re.search(r"(?im)^\s*brand:\s*(.+?)\s*$", raw)
            if brand_m:
                b = brand_m.group(1).strip().strip(".")
                if b and b.lower() not in ("none", "n/a", "unknown"):
                    info["brand"] = b
                raw = re.sub(r"(?im)^\s*brand:.*$", "", raw).strip()
            summary = no_em_dashes(raw)
            if summary:
                info["summary"] = summary
        except Exception:
            pass

    # add the best-known brand's product image first (most recognizable)
    if info.get("brand") and info["brand"].lower() != req.name.lower():
        bimg = wiki_image(info["brand"])
        if bimg and bimg not in info["images"]:
            info["images"].insert(0, bimg)
    return info


# ---- static UI (mounted last so /api/* takes precedence) ------------------
@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
