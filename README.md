# AI Drug Repurposing Research Assistant

An AI-assisted **hypothesis-generation and literature-exploration** tool. Given a
disease/condition/target, it identifies the proteins involved, finds existing
drugs that act on those proteins, gathers supporting PubMed evidence, and (once an
LLM provider is configured) writes a mechanistic narrative with a confidence note.

> This is a research aid, **not** a diagnostic or drug-discovery claim. Every
> structured fact is traceable to a source database; the LLM only summarizes.

## Architecture

```
disease name
   │
   ▼
Open Targets ── disease → ranked target proteins ──► UniProt (protein function)
   │
   ▼
Open Targets ── target → existing drugs (mechanism, clinical stage, known uses)
   │
   ▼
ranking (repurposing distance + clinical maturity + target coverage)
   │
   ▼
NCBI E-utilities + BioC ── PubMed evidence per candidate
   │
   ▼
(optional) LLM provider ── narrative summary + confidence note
```

### Data sources (all free, no key required to run)
| Source | Role |
|--------|------|
| **NCBI E-utilities** | topic → PubMed IDs |
| **NCBI BioC** | PMID/PMCID → clean article text (JSON) |
| **NCBI PMC ID Converter** | PMID → PMCID (only needed for open-access full text) |
| **NCBI Citation Exporter** | formatted citations (RIS/MEDLINE) |
| **Open Targets Platform** (GraphQL) | disease → targets, target → drugs (the structured backbone) |
| **UniProt** | protein function / names |
| **ChEMBL** | drug details (clinical phase, type) |
| **ClinicalTrials.gov** (v2 API) | real investigational drugs in active trials for the goal (surfaces brand-new agents) |

An optional NCBI API key raises the rate limit from 3 → 10 req/s.

## Project layout
```
backend/
  app/
    config.py            # env-driven settings
    clients/             # one module per external API
      base.py  ncbi.py  opentargets.py  uniprot.py  chembl.py
    llm/provider.py      # provider-agnostic LLM interface (Null/Ollama; pluggable)
    models/schemas.py    # API response contract
    services/pipeline.py # orchestration: disease in → report out
    main.py              # FastAPI app
  scripts/smoke_test.py  # live end-to-end test of every client
  requirements.txt
  .env.example
```

## Setup
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env   # then edit if you have an NCBI key / LLM provider
```

## Run

### Web UI (the site)
```powershell
$env:PYTHONUTF8='1'; .\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```
Open <http://127.0.0.1:8000/> — the custom front-end (hero, search, established vs.
repurposing/in-development sections, per-drug studies + Ask-AI). API docs at `/docs`.

Key API routes: `GET /api/repurpose?q=…`, `GET /api/studies?drug=…&topic=…`, `POST /api/ask`.

### Legacy UI (Streamlit, kept as a backup)
```powershell
$env:PYTHONUTF8='1'; .\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```
Opens at <http://localhost:8501>.

### Smoke-test everything against the live APIs
```powershell
$env:PYTHONUTF8='1'; .\.venv\Scripts\python.exe -m scripts.smoke_test
```

## LLM provider

The pipeline runs fully without an LLM (structured report only). To enable the
narrative summary, set in `.env`:
- `LLM_PROVIDER=ollama` + `LLM_MODEL=llama3.1` for a free local model, or
- wire another provider in `app/llm/provider.py` (the interface is one method).

## Roadmap (from the project plan)
- [x] Frontend UI (Streamlit demo) — `streamlit_app.py`
- [ ] Interactive pathway / PPI visualization (Reactome / STRING)
- [ ] ClinicalTrials.gov integration
- [ ] Molecular-similarity analysis (PubChem / RDKit)
- [ ] Protein structure view (RCSB PDB / AlphaFold)
