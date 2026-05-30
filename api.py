"""
RAG Chatbot API
Run with: uvicorn api:app --reload --port 8000
http://localhost:8000

Requirements:
    pip install fastapi uvicorn python-multipart
    (plus all deps from the original RAG project)
"""

import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from populate_database import load_documents, split_documents, add_to_chroma, clear_database
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
from get_embedding_function import get_embedding_function
from classifier import build_classifier, classify_query  # 🆕
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_PATH = "chroma"
DATA_PATH   = "data"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# 🆕 Train classifier once at startup
classifier_model = build_classifier(model_type="naive_bayes")


# ── Schemas ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    classification: str        # 🆕 e.g. "BIR Tax Query (Source Year: 2022)"
    predicted_class: int       # 🆕 e.g. 1

class StatusResponse(BaseModel):
    document_count: int
    chroma_path: str
    data_path: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the chatbot UI."""
    ui_path = Path(__file__).parent / "rag_chatbot_ui.html"
    if not ui_path.exists():
        return HTMLResponse("<h2>UI file not found. Place rag_chatbot_ui.html next to api.py</h2>", status_code=404)
    return HTMLResponse(ui_path.read_text(encoding="utf-8"))


@app.get("/status", response_model=StatusResponse)
def get_status():
    """Return the number of documents currently indexed in ChromaDB."""
    try:
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function(),
        )
        doc_count = len(db.get(include=[])["ids"])
    except Exception:
        doc_count = 0
    return StatusResponse(
        document_count=doc_count,
        chroma_path=CHROMA_PATH,
        data_path=DATA_PATH,
    )


@app.post("/query", response_model=QueryResponse)
def query_endpoint(body: QueryRequest):
    """Query the RAG pipeline and return an answer with classification label."""
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # ── Step 1: Classify the query ──────────────────────────────────
        predicted_class, label, year_filter = classify_query(
            classifier_model, body.query
        )

        # ── Step 2: Connect to ChromaDB ─────────────────────────────────
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function(),
        )

        # ── Step 3: Semantic search (with optional year filter) ─────────
        if year_filter:
            semantic_results = db.similarity_search(
                body.query,
                k=5,
                filter={"year": year_filter}
            )
        else:
            semantic_results = db.similarity_search(body.query, k=5)

        # ── Step 4: BM25 keyword search ─────────────────────────────────
        all_data = db.get(include=["documents", "metadatas"])
        all_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(all_data["documents"], all_data["metadatas"])
        ]
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = 5
        bm25_results = bm25_retriever.invoke(body.query)

        # ── Step 5: Merge & deduplicate ─────────────────────────────────
        seen     = set()
        combined = []
        for doc in semantic_results + bm25_results:
            doc_id = doc.metadata.get("id")
            if doc_id not in seen:
                seen.add(doc_id)
                combined.append(doc)

        if not combined:
            return QueryResponse(
                answer="I couldn't find any relevant information in the knowledge base.",
                sources=[],
                classification=label,
                predicted_class=int(predicted_class),
            )

        # ── Step 6: Generate answer ─────────────────────────────────────
        context_text = "\n\n---\n\n".join([doc.page_content for doc in combined])
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
            context=context_text, question=body.query
        )
        model         = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        response_text = model.invoke(prompt).content
        sources       = [doc.metadata.get("id", "unknown") for doc in combined]

        return QueryResponse(
            answer=response_text,
            sources=sources,
            classification=label,              # 🆕
            predicted_class=int(predicted_class),  # 🆕
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