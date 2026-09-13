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
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_chroma import Chroma

from .classifier import build_classifier, classify_query
from .database import add_to_chroma, clear_database, load_documents, split_documents
from .embeddings import get_embedding_function
from .llm import conversational_answer, rag_answer
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

# ── Train classifier once at startup ──────────────────────────────────────────
classifier_model = build_classifier(model_type="naive_bayes")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _conversational_response(body: QueryRequest, label: str, predicted_class: int) -> QueryResponse:
    """Shared fallback path: answer without RAG context."""
    answer = conversational_answer(body.query, body.history)
    return QueryResponse(
        answer=answer,
        sources=[],
        classification=label,
        predicted_class=int(predicted_class),
        mode="conversational",
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
    Smart RAG + Conversational endpoint.
    - Chit-chat (class 0) -> goes straight to conversational LLM
    - Tax / board game queries -> tries RAG first, falls back to conversational if no docs found
    """
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Step 1: Classify
        predicted_class, label, year_filter = classify_query(classifier_model, body.query)

        # Step 2: Chit-chat -> skip RAG, go conversational immediately
        if predicted_class == 0:
            return _conversational_response(body, label, predicted_class)

        # Step 3: Knowledge query -> try RAG
        try:
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
            db_empty = len(db.get(include=[])["ids"]) == 0
        except Exception:
            db_empty, db = True, None

        if db_empty or db is None:
            return _conversational_response(body, label, predicted_class)

        # Step 4: Retrieve documents
        combined = retrieve_docs(body.query, year_filter, db)

        # No relevant docs found -> fall back to conversational
        if not combined:
            return _conversational_response(body, label, predicted_class)

        # Step 5: RAG answer with retrieved context
        context_text = "\n\n---\n\n".join(doc.page_content for doc in combined)
        answer = rag_answer(body.query, body.history, context_text)
        sources = [doc.metadata.get("id", "unknown") for doc in combined]

        return QueryResponse(
            answer=answer,
            sources=sources,
            classification=label,
            predicted_class=int(predicted_class),
            mode="rag",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF into the data/ folder and index it into ChromaDB."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    os.makedirs(DATA_PATH, exist_ok=True)
    dest = Path(DATA_PATH) / file.filename

    with open(dest, "wb") as f:
        f.write(await file.read())

    try:
        docs   = load_documents()
        chunks = split_documents(docs)
        add_to_chroma(chunks)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

    return {"message": f"'{file.filename}' uploaded and indexed successfully."}


@app.delete("/reset")
def reset_database():
    """Wipe the ChromaDB vector store."""
    clear_database()
    return {"message": "Database cleared successfully."}
