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
    if req.context:
        parts.append(f"[Context: the user is currently viewing results for '{req.context}'.]")
    try:
        label, study = _study_context(req.message)
    except Exception:
        label, study = None, None
    if study:
        parts.append(f"The user referenced a study ({label}). Here is its ACTUAL text retrieved from "
                     f"PubMed/PMC. Base your answer ONLY on this text, do not guess from the title:\n"
                     f'"""\n{study}\n"""')
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
