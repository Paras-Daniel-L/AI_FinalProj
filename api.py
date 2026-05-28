"""
RAG Chatbot API
Run with: uvicorn api:app --reload --port 8000

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
from pydantic import BaseModel

from query_data import query_rag
from populate_database import load_documents, split_documents, add_to_chroma, clear_database
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
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
DATA_PATH = "data"


# ── Schemas ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

class StatusResponse(BaseModel):
    document_count: int
    chroma_path: str
    data_path: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "RAG Chatbot API is running."}


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
    """Query the RAG pipeline and return an answer with source chunk IDs."""
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function(),
        )
        results = db.similarity_search_with_score(body.query, k=5)

        if not results:
            return QueryResponse(
                answer="I couldn't find any relevant information in the knowledge base.",
                sources=[],
            )

        from langchain_core.prompts import ChatPromptTemplate
        from langchain_groq import ChatGroq

        PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
            context=context_text, question=body.query
        )
        model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        response_text = model.invoke(prompt).content
        sources = [doc.metadata.get("id", "unknown") for doc, _ in results]

        return QueryResponse(answer=response_text, sources=sources)

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
        docs = load_documents()
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
