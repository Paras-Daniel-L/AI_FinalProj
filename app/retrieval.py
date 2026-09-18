"""
Retrieval helpers: conversation-history formatting and hybrid
(semantic + BM25) document search.

Pulled out of api.py so the retrieval logic can be tested and tuned
independently of the route handlers.
"""

from typing import List, Optional

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .schemas import ConvMessage


def format_history(history: List[ConvMessage], max_turns: int = 5) -> str:
    """Format recent conversation turns into a readable string for the prompt."""
    if not history:
        return "(No previous conversation)"
    recent = history[-(max_turns * 2):]
    lines = []
    for msg in recent:
        prefix = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{prefix}: {msg.content}")
    return "\n".join(lines)


def retrieve_docs(query: str, year_filter: Optional[str], db: Chroma) -> list[Document]:
    """Run hybrid semantic + BM25 retrieval and combine results using Reciprocal Rank Fusion (RRF)."""
    if year_filter:
        semantic_results = db.similarity_search(query, k=5, filter={"year": year_filter})
    else:
        semantic_results = db.similarity_search(query, k=5)

    all_data = db.get(include=["documents", "metadatas"])
    if not all_data["documents"]:
        return semantic_results

    all_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_data["documents"], all_data["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 5
    bm25_results = bm25_retriever.invoke(query)

    # Reciprocal Rank Fusion (RRF) Implementation
    k_constant = 60
    rrf_scores = {}
    doc_map = {}

    # Score Semantic Results
    for rank, doc in enumerate(semantic_results):
        doc_id = doc.metadata.get("id", f"sem_{rank}")
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_constant + rank + 1))

    # Score BM25 Results
    for rank, doc in enumerate(bm25_results):
        doc_id = doc.metadata.get("id", f"bm25_{rank}")
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_constant + rank + 1))

    # Sort documents by their combined RRF score descending and return the Top 5
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    return [doc_map[doc_id] for doc_id in sorted_doc_ids[:5]]