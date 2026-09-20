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
from .language import detect_taglish
from .llm import conversational_answer, rag_answer, rewrite_query, translate_to_taglish
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

# ── Auto-build BM25 if the pickle isn't on disk ────────────────────────────
# database.py now builds this automatically during ingestion, but this
# covers every other path that can leave the pickle missing (an older
# Chroma DB built before that change, someone deleting the pickle by hand,
# a fresh clone where only the Chroma folder was copied over, etc). Without
# this, retrieve_docs() silently falls back to semantic-only search and
# RRF fuses a single list against itself - nothing errors, it just quietly
# stops being hybrid retrieval.
if global_bm25 is None:
  print(
      "⚠️ [Startup] No BM25 index found on disk — building one now from"
      " ChromaDB so hybrid retrieval doesn't silently degrade to"
      " semantic-only search."
  )
  global_bm25 = build_and_save_bm25(global_db)
  if global_bm25 is None:
    print(
        "ℹ️ [Startup] ChromaDB is currently empty — BM25 index will be"
        " built automatically after the first document is added via"
        " /upload or `python -m app.database --reset`."
    )
  else:
    print("✅ [Startup] BM25 index built and saved.")

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
      verified=True,
      retries=0,
      degraded=False,
      # Reported for metadata consistency only - conversational_answer's own
      # SYSTEM_PROMPT already tells it to match the user's language, so no
      # separate rewrite/translate call happens on this path.
      language="taglish" if detect_taglish(body.query) else "english",
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

    # Step 3: Language detection + query rewriting
    # Detect on the raw query as typed (the classifier above already saw the
    # raw query too, and its training data includes Taglish phrasing, so
    # routing is unaffected). If Taglish, rewrite into formal English BEFORE
    # retrieval - both BM25 and semantic search work better against a
    # normalized query, and it gives the generation/verification steps a
    # single consistent language to reason in.
    is_taglish = detect_taglish(body.query)
    if is_taglish:
      t_rw = time.time()
      retrieval_query = rewrite_query(body.query)
      print(f"🌐 [Taglish] Detected. Rewritten query: '{retrieval_query}' "
            f"in {time.time() - t_rw:.3f}s")
    else:
      retrieval_query = body.query

    # Step 4: Hybrid Retrieval
    t_retrieval = time.time()
    print("🔎 [Retrieval] Starting hybrid search...")
    combined = retrieve_docs(
        retrieval_query, year_filter, global_db, bm25_retriever=global_bm25
    )
    print(
        f"⏱️ [Retrieval Complete] Fetched {len(combined)} docs in"
        f" {time.time() - t_retrieval:.3f}s"
    )

    if not combined:
      print("⚠️ [Retrieval] No documents found. Falling back to conversational.")
      return _conversational_response(body, label, predicted_class)

    # Step 5: LLM Generation + verification/retry loop (runs in English,
    # against the formal rewritten query if this started as Taglish)
    t_llm = time.time()
    print("🤖 [LLM] Sending context to Groq...")
    context_text = "\n\n---\n\n".join(doc.page_content for doc in combined)
    result = rag_answer(retrieval_query, body.history, context_text)
    print(
        f"⏱️ [LLM Complete] verified={result.verified} retries={result.retries_used}"
        f" degraded={result.degraded} in {time.time() - t_llm:.3f}s"
    )

    # Step 6: Translate back to Taglish - only for a verified, non-degraded
    # answer. The pre-written safe-fallback text is left in English by
    # design: it's a fixed, already-reviewed string, and translating it
    # would be one more unverified LLM call in the one path that's
    # specifically supposed to avoid unverified output.
    final_answer = result.answer
    if is_taglish and not result.degraded:
      t_tr = time.time()
      final_answer = translate_to_taglish(result.answer)
      print(f"🌐 [Taglish] Translated verified answer in {time.time() - t_tr:.3f}s")

    # If we degraded to the safe fallback, don't cite sources for an answer
    # we didn't actually generate from them.
    sources = [] if result.degraded else [doc.metadata.get("id", "unknown") for doc in combined]
    print(f"🎉 [Request Finished] Total elapsed time: {time.time() - t0:.3f}s\n")

    return QueryResponse(
        answer=final_answer,
        sources=sources,
        classification=label,
        predicted_class=int(predicted_class),
        mode="rag_fallback" if result.degraded else "rag",
        verified=result.verified,
        retries=result.retries_used,
        degraded=result.degraded,
        language="taglish" if is_taglish else "english",
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