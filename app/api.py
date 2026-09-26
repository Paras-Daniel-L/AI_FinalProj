"""
RAG Chatbot API  –  Sagot AI

Run from the project root with:
    uvicorn main:app --reload --port 8000
http://localhost:8000

Requirements:
    pip install fastapi uvicorn python-multipart
    (plus all deps from the original RAG project)
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_chroma import Chroma

from .classifier import NO_SOURCES_LABEL, get_year_filter, label_from_docs
from .database import add_to_chroma, clear_database, load_documents, split_documents
from .embeddings import get_embedding_function
from .greetings import is_greeting
from .language import LanguageResult, detect_language
from .llm import rag_answer
from .prompts import get_greeting_message, get_no_answer_message
from .retrieval import retrieve_docs
from .schemas import QueryRequest, QueryResponse, StatusResponse

load_dotenv()

app = FastAPI(title="Sagot AI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_PATH = "chroma"
DATA_PATH   = "data"

# api.py lives in app/, so the project root (and static/) is one level up.
STATIC_DIR = Path(__file__).parent.parent / "static"

# Serves /static/css/style.css and /static/js/app.js referenced by index.html.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Helpers ────────────────────────────────────────────────────────────────────

def _greeting_response(lang: LanguageResult) -> QueryResponse:
    """
    A bare greeting ("hi", "kumusta ka") with nothing else attached — see
    app/greetings.py's is_greeting() for exactly what qualifies. Returns a
    fixed, localized greeting string; NO LLM call happens on this path.

    Sagot AI has no general-knowledge/open-domain conversation mode
    anymore (see prompts.SYSTEM_PROMPT) — this exists only so a plain
    "hello" doesn't come back as a no-answer refusal.
    Routing and retrieval never run for this path.
    """
    return QueryResponse(
        answer=get_greeting_message(lang.label),
        sources=[],
        classification="Greeting",
        mode="greeting",
        language=lang.label,
    )


def _no_answer_response(lang: LanguageResult) -> QueryResponse:
    """
    A tax query for which the knowledge base has no supporting evidence —
    either the DB is empty, retrieval found nothing, or nothing survived
    the coarse relevance filter. No generation call is made on this path
    at all (no open-domain fallback exists to fall through to); the
    localized refusal is returned directly.
    """
    return QueryResponse(
        answer=get_no_answer_message(lang.label),
        sources=[],
        classification=NO_SOURCES_LABEL,
        mode="no_answer",
        language=lang.label,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the chatbot UI shell (CSS/JS load separately via the /static mount)."""
    ui_path = STATIC_DIR / "index.html"
    if not ui_path.exists():
        return HTMLResponse(
            "<h2>UI file not found. Expected static/index.html at the project root.</h2>",
            status_code=404
        )
    return HTMLResponse(ui_path.read_text(encoding="utf-8"))


@app.get("/status", response_model=StatusResponse)
def get_status():
    """Return the number of documents currently indexed in ChromaDB."""
    try:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
        doc_count = len(db.get(include=[])["ids"])
    except Exception:
        doc_count = 0
    return StatusResponse(document_count=doc_count, chroma_path=CHROMA_PATH, data_path=DATA_PATH)


@app.post("/query", response_model=QueryResponse)
def query_endpoint(body: QueryRequest):
    """
    Grounded BIR-tax RAG endpoint. No general-knowledge fallback exists:
    every non-greeting query is routed (explicit-year filter, see
    classifier.py), retrieved, labeled from the retrieved documents'
    own metadata, then answered strictly from those documents or refused
    via the canned no-answer response.
    """
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Step 0: Detect the user's language once, from the untouched
        # original query, before anything else runs. This value — not a
        # re-derivation from a possibly-mutated prompt later — is what
        # travels through retrieval into generation. See app/language.py.
        lang = detect_language(body.query)

        # Step 1: Bare greeting ("hi", "kumusta ka") -> fixed reply, no LLM
        # call, no retrieval. Anything with real content attached to a
        # greeting-looking prefix does NOT match here and falls through to
        # the normal pipeline below.
        if is_greeting(body.query):
            return _greeting_response(lang)

        # Step 2: Route (year filter from an explicit year/citation in the
        # query, or None for whole-corpus search). Rule-based; the
        # classification LABEL is derived later, from what retrieval found.
        year_filter = get_year_filter(body.query)

        # Step 3: Knowledge query -> try RAG. An empty/unreachable DB means
        # there is no evidence to ground an answer in, so this returns a
        # no-answer response — there is no open-domain fallback to drop to.
        try:
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
            db_empty = len(db.get(include=[])["ids"]) == 0
        except Exception:
            db_empty, db = True, None

        if db_empty or db is None:
            return _no_answer_response(lang)

        # Step 4: Retrieve documents (RRF-fused, then coarse-relevance
        # filtered — see retrieve_docs()/filter_by_relevance() in
        # retrieval.py). Empty here means either nothing was retrieved or
        # nothing cleared the relevance bar.
        combined = retrieve_docs(body.query, year_filter, db)

        if not combined:
            return _no_answer_response(lang)

        # Step 4b: Label the query from the retrieved evidence — which
        # corpus categories (year folders / faq) the chunks came from.
        label = label_from_docs(combined)

        # Step 5: RAG answer with retrieved context, in the user's language.
        # rag_answer() itself may return the canned no-answer message (the
        # generation-layer grounding check caught something the coarse
        # retrieval-layer filter above let through) — reflect that in
        # `mode`/`sources` rather than claiming these sources were used.
        context_text = "\n\n---\n\n".join(doc.page_content for doc in combined)
        answer = rag_answer(body.query, body.history, context_text, language=lang)

        if answer == get_no_answer_message(lang.label):
            return _no_answer_response(lang)

        sources = [doc.metadata.get("id", "unknown") for doc in combined]
        return QueryResponse(
            answer=answer,
            sources=sources,
            classification=label,
            mode="rag",
            language=lang.label,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), category: str = Form(default="")):
    """
    Upload a PDF and index it into ChromaDB.

    `category` is optional and maps to the same folder-based `year` tag
    load_documents() uses for everything else (e.g. "2022", "2026", "faq").
    Pass one of the existing YEAR_FOLDERS values if this document should
    stay reachable through the classifier's year_filter; leave it blank
    to drop the file straight into data/ root, tagged "uncategorized" by
    load_documents() — still fully searchable, just not classifier-
    filterable by year.

    Filenames and categories come from the client, so both are reduced to
    a bare basename via Path(...).name before touching the filesystem —
    otherwise a filename like "../../x.pdf" or a category like "../.env"
    could write outside DATA_PATH.
    """
    safe_name = Path(file.filename or "").name
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_category = Path(category or "").name
    dest_dir = Path(DATA_PATH) / safe_category if safe_category else Path(DATA_PATH)
    os.makedirs(dest_dir, exist_ok=True)
    dest = dest_dir / safe_name

    with open(dest, "wb") as f:
        f.write(await file.read())

    try:
        docs   = load_documents()
        chunks = split_documents(docs)
        add_to_chroma(chunks)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

    return {
        "message": f"'{safe_name}' uploaded and indexed successfully.",
        "category": safe_category or "uncategorized",
    }


@app.delete("/reset")
def reset_database():
    """Wipe the ChromaDB vector store."""
    clear_database()
    return {"message": "Database cleared successfully."}