import json
import time
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .cache import is_redis_available, make_cache_key, redis_client
from .schemas import ConvMessage

RETRIEVAL_CACHE_TTL = 86400  # 24 hours


def format_history(history: List[ConvMessage], max_turns: int = 5) -> str:
  if not history:
    return "(No previous conversation)"
  recent = history[-(max_turns * 2) :]
  lines = [
      f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
      for m in recent
  ]
  return "\n".join(lines)


def retrieve_docs(
    query: str,
    year_filter: Optional[str],
    db: Chroma,
    bm25_retriever: Optional[BM25Retriever] = None,
) -> List[Document]:
  """Hybrid semantic (ChromaDB) + BM25 retrieval with Redis caching & RRF."""
  cache_key = make_cache_key("chroma_rrf", query, year_filter)

  # 1. Redis Check
  if is_redis_available():
    try:
      cached_data = redis_client.get(cache_key)
      if cached_data:
        print("⚡ [Cache] Redis HIT — returning cached chunks instantly.")
        cached_docs = json.loads(cached_data)
        return [
            Document(
                page_content=item["page_content"], metadata=item["metadata"]
            )
            for item in cached_docs
        ]
    except Exception as e:
      print(f"⚠️ [Cache] Redis check error: {e}")

  print("🔄 [Cache] Redis MISS — querying ChromaDB & BM25...")

  # 2. ChromaDB Semantic Search
  semantic_results = []
  t_sem = time.time()
  print("   ↳ Running ChromaDB similarity_search (calling embedding model)...")

  if year_filter:
    try:
      semantic_results = db.similarity_search(
          query, k=5, filter={"year": str(year_filter)}
      )
    except Exception:
      pass

    if not semantic_results and year_filter.isdigit():
      try:
        semantic_results = db.similarity_search(
            query, k=5, filter={"year": int(year_filter)}
        )
      except Exception:
        pass

    if not semantic_results:
      print(
          f"   ↳ No documents found for year {year_filter}. Falling back to"
          " unfiltered ChromaDB search..."
      )
      semantic_results = db.similarity_search(query, k=5)
  else:
    semantic_results = db.similarity_search(query, k=5)

  print(
      f"   ↳ ChromaDB returned {len(semantic_results)} matches in"
      f" {time.time() - t_sem:.3f}s"
  )

  # 3. BM25 Search
  bm25_results = []
  if bm25_retriever is not None:
    t_bm = time.time()
    try:
      bm25_results = bm25_retriever.invoke(query)
      print(
          f"   ↳ BM25 returned {len(bm25_results)} matches in"
          f" {time.time() - t_bm:.3f}s"
      )
    except Exception as e:
      print(f"⚠️ BM25 search failed: {e}")
  else:
    # If you're seeing this on a freshly started server, it means the BM25
    # pickle wasn't found AND the startup auto-build in api.py also came up
    # empty (e.g. an empty Chroma DB). Hybrid retrieval is degraded to
    # semantic-only until a BM25 index exists.
    print("ℹ️ BM25 index not loaded; proceeding with semantic results only.")

  if not semantic_results and not bm25_results:
    return []

  # 4. Reciprocal Rank Fusion (RRF)
  k_constant = 60
  rrf_scores = {}
  doc_map = {}

  for rank, doc in enumerate(semantic_results):
    doc_id = doc.metadata.get("id", f"sem_{rank}")
    doc_map[doc_id] = doc
    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
        1.0 / (k_constant + rank + 1)
    )

  for rank, doc in enumerate(bm25_results):
    doc_id = doc.metadata.get("id", f"bm25_{rank}")
    doc_map[doc_id] = doc
    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
        1.0 / (k_constant + rank + 1)
    )

  sorted_doc_ids = sorted(
      rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
  )
  top_docs = [doc_map[doc_id] for doc_id in sorted_doc_ids[:5]]

  # 5. Populate Redis Cache
  if is_redis_available() and top_docs:
    try:
      payload = [
          {"page_content": d.page_content, "metadata": d.metadata}
          for d in top_docs
      ]
      redis_client.setex(cache_key, RETRIEVAL_CACHE_TTL, json.dumps(payload))
    except Exception:
      pass

  return top_docs