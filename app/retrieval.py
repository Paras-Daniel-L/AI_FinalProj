"""
Retrieval helpers: conversation-history formatting and hybrid
(semantic + BM25) document search, fused with Reciprocal Rank Fusion (RRF).

Pulled out of api.py so the retrieval logic can be tested and tuned
independently of the route handlers.

RRF background
--------------
    RRF(d) = sum over retrievers r of  1 / (k + rank_r(d))

`rank_r(d)` is the 1-indexed position of document `d` in retriever r's
ranked result list (the top hit has rank 1). A document that doesn't
appear in a given retriever's results simply contributes 0 from that
retriever — it isn't penalized beyond not being scored there.

`reciprocal_rank_fusion()` is a small, pure function: it takes any number
of already-ranked `Document` lists and returns one fused, deduplicated,
re-ranked list. It doesn't know or care whether the lists came from
Chroma, BM25, or anything else, so it's reusable if a third retriever
(e.g. a reranker) is ever added.

`retrieve_docs()` is the project-specific wiring: it runs the two
existing retrievers (Chroma semantic search, BM25 keyword search) against
this project's Chroma `db`, isolates failures in either one, and fuses
their outputs via `reciprocal_rank_fusion()`.
"""

import os
from typing import Dict, List, Optional, Sequence

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .schemas import ConvMessage

# ── RRF configuration ────────────────────────────────────────────────────
# All overridable via env vars so tuning doesn't require a code change.
# Defaults match the constants the fusion logic previously had hardcoded.
RRF_K: int = int(os.environ.get("RRF_K", "60"))
SEMANTIC_TOP_K: int = int(os.environ.get("SEMANTIC_TOP_K", "5"))
BM25_TOP_K: int = int(os.environ.get("BM25_TOP_K", "5"))
RRF_FINAL_TOP_K: int = int(os.environ.get("RRF_FINAL_TOP_K", "5"))

# ── Coarse relevance pre-filter (no-answer gate, layer 1 of 2) ───────────
# This is the retrieval-layer half of the hybrid no-answer strategy: cheap,
# and deliberately permissive by default (both thresholds off / no-op)
# because neither RRF score nor retriever-corroboration count has been
# calibrated against a labeled eval set yet. It exists to reject the
# unambiguous "nothing plausible came back" case before paying for a
# generation call — NOT to make fine-grained relevance judgments. The real
# precision gate is the LLM-side grounding check in app/llm.py's
# rag_answer() (the NO_ANSWER sentinel), because "does this text actually
# answer the question" isn't something a rank-fusion score alone can tell.
MIN_RRF_SCORE: float = float(os.environ.get("MIN_RRF_SCORE", "0.0"))
MIN_RETRIEVER_CORROBORATION: int = int(os.environ.get("MIN_RETRIEVER_CORROBORATION", "0"))


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


def _canonical_id(doc: Document, retriever_label: str, rank: int) -> str:
    """
    The project's canonical chunk identifier is metadata["id"], stamped by
    `calculate_chunk_ids()` in database.py (e.g. "2022:source.pdf:3:0").
    Every chunk that went through normal ingestion has one.

    Documents missing it (malformed metadata, a hand-built Document in a
    test, etc.) fall back to a content hash rather than document text
    itself or a rank-based placeholder — rank-based fallbacks would treat
    "rank 1 from retriever A" and "rank 1 from retriever B" as the same
    document, which silently merges two unrelated chunks.
    """
    doc_id = doc.metadata.get("id")
    if doc_id:
        return doc_id
    return f"__no_id__:{retriever_label}:{hash(doc.page_content) & 0xffffffff}"


def reciprocal_rank_fusion(
    ranked_lists: Sequence[List[Document]],
    k: int = RRF_K,
    top_k: Optional[int] = RRF_FINAL_TOP_K,
    source_labels: Optional[Sequence[str]] = None,
) -> List[Document]:
    """
    Fuse any number of independently-ranked Document lists into one
    deduplicated, RRF-scored, descending-sorted list.

    Args:
        ranked_lists: one ranked list of Documents per retriever. A list
            may be empty (that retriever found nothing) without affecting
            the others. Malformed entries within a list (non-Document
            items) are skipped and logged, not fatal.
        k: the RRF constant. Higher k flattens the influence of rank
           position; lower k rewards top ranks more sharply.
        top_k: how many fused documents to return. None returns all of them.
        source_labels: optional human-readable names for each list, used
            only for logging (e.g. ["semantic", "bm25"]).

    Returns:
        Documents ordered by descending combined RRF score, each appearing
        exactly once (deduplicated on metadata["id"], see _canonical_id).
    """
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}
    # How many DISTINCT retrievers surfaced this doc at all (0/1 per
    # retriever, not per-rank) — corroboration across retrievers is a
    # stronger relevance signal than either retriever's score alone. Also
    # used by filter_by_relevance() as the retrieval-layer half of the
    # no-answer gate.
    retriever_counts: Dict[str, int] = {}
    source_retrievers: Dict[str, List[str]] = {}
    per_retriever_counts = []

    for i, ranked_list in enumerate(ranked_lists):
        label = (
            source_labels[i]
            if source_labels and i < len(source_labels)
            else f"retriever_{i}"
        )

        if not ranked_list:
            per_retriever_counts.append((label, 0))
            continue

        contributed = 0
        seen_in_this_retriever = set()
        for rank, doc in enumerate(ranked_list, start=1):  # rank starts at 1
            if not isinstance(doc, Document):
                print(f"⚠️  [RRF] {label}: skipping malformed entry at rank {rank}")
                continue
            doc_id = _canonical_id(doc, label, rank)
            doc_map.setdefault(doc_id, doc)  # keep first-seen copy of the doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
            if doc_id not in seen_in_this_retriever:
                seen_in_this_retriever.add(doc_id)
                retriever_counts[doc_id] = retriever_counts.get(doc_id, 0) + 1
                source_retrievers.setdefault(doc_id, []).append(label)
            contributed += 1
        per_retriever_counts.append((label, contributed))

    sorted_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)
    if top_k is not None:
        sorted_ids = sorted_ids[:top_k]

    counts_str = ", ".join(f"{label}={n}" for label, n in per_retriever_counts)
    print(
        f"🔀 [RRF] inputs: {counts_str} | unique docs: {len(rrf_scores)} "
        f"| passing to generation: {len(sorted_ids)} (k={k})"
    )

    # Stamp fusion metadata onto a fresh Document per result rather than
    # mutating the retrievers' original objects in place (those may be
    # reused/cached upstream). Existing metadata (id, year, source, page,
    # etc.) is preserved untouched.
    fused_docs = []
    for doc_id in sorted_ids:
        base = doc_map[doc_id]
        enriched_metadata = dict(base.metadata)
        enriched_metadata["_rrf_score"] = rrf_scores[doc_id]
        enriched_metadata["_retriever_count"] = retriever_counts.get(doc_id, 1)
        enriched_metadata["_source_retrievers"] = source_retrievers.get(doc_id, [])
        fused_docs.append(Document(page_content=base.page_content, metadata=enriched_metadata))

    return fused_docs


def filter_by_relevance(
    docs: List[Document],
    min_score: float = MIN_RRF_SCORE,
    min_retriever_count: int = MIN_RETRIEVER_CORROBORATION,
) -> List[Document]:
    """
    Drop fused documents that don't meet the coarse relevance bar (reads
    the `_rrf_score` / `_retriever_count` metadata reciprocal_rank_fusion()
    stamps onto each result). With the default thresholds (0.0 / 0) this
    is a no-op — every fused document passes, since neither threshold has
    been calibrated against real query data yet (see proposal: calibrate
    empirically against a small eval set of answerable / weakly-related /
    out-of-scope queries, rather than picking a number arbitrarily).

    Docs missing the fusion metadata (e.g. hand-built Documents in a test,
    or a future retriever wired in without going through
    reciprocal_rank_fusion()) are treated as passing by default, not
    silently dropped.
    """
    survivors = []
    for doc in docs:
        score = doc.metadata.get("_rrf_score", float("inf"))
        count = doc.metadata.get("_retriever_count", min_retriever_count)
        if score >= min_score and count >= min_retriever_count:
            survivors.append(doc)

    dropped = len(docs) - len(survivors)
    if dropped:
        print(
            f"🚫 [Relevance] dropped {dropped} doc(s) below coarse threshold "
            f"(min_score={min_score}, min_retriever_count={min_retriever_count})"
        )
    return survivors


def retrieve_docs(
    query: str,
    year_filter: Optional[str],
    db: Chroma,
    semantic_top_k: int = SEMANTIC_TOP_K,
    bm25_top_k: int = BM25_TOP_K,
    rrf_k: int = RRF_K,
    final_top_k: int = RRF_FINAL_TOP_K,
) -> List[Document]:
    """
    Hybrid retrieval: run dense (Chroma semantic) and sparse (BM25)
    retrieval independently, then fuse their rankings with
    `reciprocal_rank_fusion()`.

    A failure in either retriever is caught and logged; it degrades to
    whatever the other retriever found rather than raising, so one bad
    call to Chroma or a corrupt BM25 index doesn't take down the whole
    `/query` request.
    """
    # ── Dense / semantic retrieval (Chroma) ──────────────────────────
    semantic_results: List[Document] = []
    try:
        if year_filter:
            semantic_results = db.similarity_search(
                query, k=semantic_top_k, filter={"year": year_filter}
            )
        else:
            semantic_results = db.similarity_search(query, k=semantic_top_k)
    except Exception as e:
        print(f"⚠️  [Retrieval] semantic search failed, continuing without it: {e}")
        semantic_results = []

    # ── Sparse / keyword retrieval (BM25) ────────────────────────────
    bm25_results: List[Document] = []
    try:
        all_data = db.get(include=["documents", "metadatas"])
        if all_data["documents"]:
            all_docs = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(all_data["documents"], all_data["metadatas"])
            ]
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = bm25_top_k
            bm25_results = bm25_retriever.invoke(query)
    except Exception as e:
        print(f"⚠️  [Retrieval] BM25 search failed, continuing without it: {e}")
        bm25_results = []

    if not semantic_results and not bm25_results:
        print("🔀 [RRF] both retrievers returned 0 results — nothing to fuse")
        return []

    fused = reciprocal_rank_fusion(
        [semantic_results, bm25_results],
        k=rrf_k,
        top_k=final_top_k,
        source_labels=["semantic", "bm25"],
    )

    # Coarse relevance pre-filter — see filter_by_relevance()'s docstring.
    # An empty return here (same as the "0 results" case above) is what
    # api.py treats as "insufficient evidence" and routes to a localized
    # no-answer response instead of generation.
    return filter_by_relevance(fused)