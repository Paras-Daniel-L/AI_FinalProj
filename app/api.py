import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_chroma import Chroma

from .bm25_manager import build_and_save_bm25, load_bm25
from .cache import is_redis_available, redis_client
from .classifier import build_classifier, classify_query
from .database import (
    add_to_chroma,
    clear_database,
    load_documents,
    split_documents,
)
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
DATA_PATH = "data"
STATIC_DIR = Path(__file__).parent.parent / "static"

if STATIC_DIR.exists():
  app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Global singletons (initialized ONCE at server startup) ────────────────────
print("🚀 [Startup] Initializing Classifier...")
classifier_model = build_classifier(model_type="naive_bayes")

print("🚀 [Startup] Connecting to ChromaDB...")
global_db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=get_embedding_function(),  # Loads embedding function once
)

print("🚀 [Startup] Loading pre-built BM25 Index...")
global_bm25 = load_bm25()
print("✅ [Startup] System ready.\n")


def flush_chroma_cache():
  """Invalidate all cached ChromaDB retrieval records in Redis."""
  if is_redis_available():
    try:
      keys = redis_client.keys("chroma_rrf:*")
      if keys:
        redis_client.delete(*keys)
    except Exception:
      pass


def _conversational_response(
    body: QueryRequest, label: str, predicted_class: int
) -> QueryResponse:
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
  ui_path = STATIC_DIR / "index.html"
  if not ui_path.exists():
    return HTMLResponse("<h2>UI file not found.</h2>", status_code=404)
  return HTMLResponse(ui_path.read_text(encoding="utf-8"))


@app.get("/status", response_model=StatusResponse)
def get_status():
  try:
    doc_count = len(global_db.get(include=[])["ids"])
  except Exception:
    doc_count = 0
  return StatusResponse(
      document_count=doc_count, chroma_path=CHROMA_PATH, data_path=DATA_PATH
  )


@app.post("/query", response_model=QueryResponse)
def query_endpoint(body: QueryRequest):
  if not body.query.strip():
    raise HTTPException(status_code=400, detail="Query cannot be empty.")

  t0 = time.time()
  try:
    # Step 1: Classify
    print(f"\n--- [New Request] Query: '{body.query}' ---")
    predicted_class, label, year_filter = classify_query(
        classifier_model, body.query
    )
    print(f"⏱️ [Classification Complete] in {time.time() - t0:.3f}s")

    # Step 2: Chit-chat check
    if predicted_class == 0:
      print("💬 [Routing] Direct to conversational LLM (Class 0)")
      return _conversational_response(body, label, predicted_class)

    # Step 3: Hybrid Retrieval
    t_retrieval = time.time()
    print("🔎 [Retrieval] Starting hybrid search...")
    combined = retrieve_docs(
        body.query, year_filter, global_db, bm25_retriever=global_bm25
    )
    print(
        f"⏱️ [Retrieval Complete] Fetched {len(combined)} docs in"
        f" {time.time() - t_retrieval:.3f}s"
    )

    if not combined:
      print("⚠️ [Retrieval] No documents found. Falling back to conversational.")
      return _conversational_response(body, label, predicted_class)

    # Step 4: LLM Generation
    t_llm = time.time()
    print("🤖 [LLM] Sending context to Groq...")
    context_text = "\n\n---\n\n".join(doc.page_content for doc in combined)
    answer = rag_answer(body.query, body.history, context_text)
    print(f"⏱️ [LLM Complete] Answer generated in {time.time() - t_llm:.3f}s")

    sources = [doc.metadata.get("id", "unknown") for doc in combined]
    print(f"🎉 [Request Finished] Total elapsed time: {time.time() - t0:.3f}s\n")

    return QueryResponse(
        answer=answer,
        sources=sources,
        classification=label,
        predicted_class=int(predicted_class),
        mode="rag",
    )

  except Exception as e:
    print(f"❌ [Error in /query]: {e}")
    raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
  global global_bm25
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

    # Update BM25 and invalidate Redis cache
    global_bm25 = build_and_save_bm25(global_db)
    flush_chroma_cache()

  except Exception as e:
    dest.unlink(missing_ok=True)
    raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

  return {"message": f"'{file.filename}' uploaded and indexed successfully."}


@app.delete("/reset")
def reset_database():
  global global_bm25
  clear_database()
  global_bm25 = None
  build_and_save_bm25(global_db)
  flush_chroma_cache()
  return {"message": "Database and BM25 index cleared successfully."}