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
import re
import threading
from typing import Dict, List, Optional, Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

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

# Optional semantic-distance gate (OFF by default). Chroma returns a distance
# (lower = closer; default metric is squared L2). When set, a chunk found ONLY
# by the semantic retriever and farther than this is dropped; chunks that BM25
# also found are kept (two retrievers agreeing is stronger evidence than one
# distance). The right value must be calibrated on labeled queries — turn on
# the "📏 [Semantic]" log lines below, compare answerable vs out-of-scope
# queries, then set e.g. MAX_SEMANTIC_DISTANCE=1.1 in .env.
_max_dist = os.environ.get("MAX_SEMANTIC_DISTANCE", "").strip()
MAX_SEMANTIC_DISTANCE: Optional[float] = float(_max_dist) if _max_dist else None


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


def source_title(doc: Document) -> str:
    """
    Human-readable issuance name for a chunk, from its PDF filename, e.g.
    "data\\2024\\RMC No. 34-2024.pdf" -> "RMC No. 34-2024". Handles both
    Windows and POSIX separators; falls back to "Unknown source".
    """
    source = str(doc.metadata.get("source") or "")
    name = re.split(r"[\\/]", source)[-1]
    name = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE).strip()
    return name or "Unknown source"


def source_label(doc: Document, number: int) -> str:
    """
    "[1] RMC No. 34-2024, p.3 (2024)" — the same label is used as the
    chunk header in the prompt context AND as the entry in the response's
    `sources`, so a citation like [1] in the answer maps 1:1 to a source.
    PyPDF pages are 0-indexed; the label shows the human page number.
    """
    parts = f"[{number}] {source_title(doc)}"
    page = doc.metadata.get("page")
    if isinstance(page, int):
        parts += f", p.{page + 1}"
    category = doc.metadata.get("year")
    if category:
        parts += f" ({category})"
    return parts


def format_context(docs: List[Document]) -> str:
    """
    Build the RETRIEVED DOCUMENTS text for the generation and verifier
    prompts, one numbered, labeled block per chunk:

        [1] RMC No. 34-2024, p.3 (2024)
        <chunk text>

    Numbering follows the fused ranking order, and matches the labels
    returned by `source_labels()`.
    """
    blocks = [
        f"{source_label(doc, i)}\n{doc.page_content.strip()}"
        for i, doc in enumerate(docs, start=1)
    ]
    return "\n\n---\n\n".join(blocks)


def source_labels(docs: List[Document]) -> List[str]:
    """Labels for the response's `sources`, numbered like format_context()."""
    return [source_label(doc, i) for i, doc in enumerate(docs, start=1)]


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
    max_distance: Optional[float] = MAX_SEMANTIC_DISTANCE,
) -> List[Document]:
    """
    Drop fused documents that don't meet the coarse relevance bar (reads
    the `_rrf_score` / `_retriever_count` / `_semantic_distance` metadata
    stamped during retrieval and fusion). With the defaults (0.0 / 0 / None)
    this is a no-op — none of the thresholds have been calibrated against
    labeled queries yet, so none is enforced.

    `max_distance` only ever drops a chunk that the semantic retriever found
    alone (retriever_count < 2): a chunk BM25 also surfaced is kept however
    far its embedding is.

    Docs missing the metadata (hand-built Documents in a test, a future
    retriever) are treated as passing, not silently dropped.
    """
    survivors = []
    for doc in docs:
        score = doc.metadata.get("_rrf_score", float("inf"))
        count = doc.metadata.get("_retriever_count", min_retriever_count)
        distance = doc.metadata.get("_semantic_distance")
        too_far = (
            max_distance is not None
            and distance is not None
            and distance > max_distance
            and count < 2
        )
        if score >= min_score and count >= min_retriever_count and not too_far:
            survivors.append(doc)

    dropped = len(docs) - len(survivors)
    if dropped:
        print(
            f"🚫 [Relevance] dropped {dropped} doc(s) below coarse threshold "
            f"(min_score={min_score}, min_retriever_count={min_retriever_count}, "
            f"max_distance={max_distance})"
        )
    return survivors


# ── BM25 (sparse) retrieval ──────────────────────────────────────────────
# Previously rebuilt from every chunk in Chroma on EVERY query, ignored
# year_filter, tokenized with str.split() (case- and punctuation-sensitive:
# "RMC" != "rmc", "34-2024?" != "34-2024"), and — via get_top_n — returned k
# arbitrary documents even when the query shared no terms with the corpus.

_TOKEN_RE = re.compile(r"\w+(?:-\w+)*", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """
    Lowercased word tokens; hyphenated tokens are kept whole AND split, so
    an issuance number like "34-2024" matches "34-2024" and bare "2024".
    """
    tokens: List[str] = []
    for m in _TOKEN_RE.finditer((text or "").lower()):
        tok = m.group(0)
        tokens.append(tok)
        if "-" in tok:
            tokens.extend(tok.split("-"))
    return tokens


class _BM25Index:
    def __init__(self, docs: List[Document]):
        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in docs]) if docs else None

    def search(self, query: str, k: int, year: Optional[str] = None) -> List[Document]:
        """
        Top-k documents with a POSITIVE BM25 score (no term overlap -> none),
        optionally restricted to one year category.

        Scores always come from the FULL-corpus index and the year is applied
        afterwards to the ranking. A per-year index would compute term
        weights from only a handful of chunks (2001 has ~1 page), and BM25's
        IDF goes to zero/negative on tiny corpora — every score would be
        non-positive and nothing would ever match.
        """
        query_tokens = tokenize(query)
        if self.bm25 is None or not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        candidates = [
            i for i in range(len(scores))
            if year is None or self.docs[i].metadata.get("year") == year
        ]
        ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)[:k]
        results = []
        for i in ranked:
            if scores[i] <= 0:
                break
            meta = dict(self.docs[i].metadata)
            meta["_bm25_score"] = float(scores[i])
            results.append(Document(page_content=self.docs[i].page_content, metadata=meta))
        return results


_bm25_lock = threading.Lock()
_bm25_state = {"key": None, "index": None}


def _get_bm25_index(db: Chroma) -> _BM25Index:
    """
    Return the cached full-corpus BM25 index, rebuilding only when the set
    of chunk ids in Chroma changes (re-ingest, upload, reset). Checking that
    costs one ids-only fetch; the full text fetch and index build happen
    once per index state instead of once per query.
    """
    ids = db.get(include=[])["ids"]
    key = (len(ids), hash(tuple(sorted(ids))))
    with _bm25_lock:
        if _bm25_state["key"] != key:
            data = db.get(include=["documents", "metadatas"])
            docs = [
                Document(page_content=text, metadata=meta or {})
                for text, meta in zip(data["documents"], data["metadatas"])
            ]
            _bm25_state.update(key=key, index=_BM25Index(docs))
            print(f"🗂️  [BM25] index built: {len(docs)} chunk(s)")
        return _bm25_state["index"]


def _semantic_search(
    db: Chroma, query: str, k: int, year_filter: Optional[str]
) -> List[Document]:
    """Chroma search that keeps each hit's distance in metadata["_semantic_distance"]."""
    kwargs = {"filter": {"year": year_filter}} if year_filter else {}
    results = []
    for doc, distance in db.similarity_search_with_score(query, k=k, **kwargs):
        meta = dict(doc.metadata)
        meta["_semantic_distance"] = float(distance)
        results.append(Document(page_content=doc.page_content, metadata=meta))
    return results


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
    retrieval independently — BOTH restricted to `year_filter` when one is
    given — then fuse their rankings with `reciprocal_rank_fusion()`.

    year_filter is only ever set from an explicit, in-corpus year/citation
    in the query (see classifier.get_year_filter), so applying it to BM25
    too can't hard-exclude the right chunks on an ML misguess the way the
    old classifier-driven filter could.

    A failure in either retriever is caught and logged; it degrades to
    whatever the other retriever found rather than raising.
    """
    semantic_results: List[Document] = []
    try:
        semantic_results = _semantic_search(db, query, semantic_top_k, year_filter)
        if semantic_results:
            dists = ", ".join(f"{d.metadata['_semantic_distance']:.3f}" for d in semantic_results)
            print(f"📏 [Semantic] distances (best first): {dists}")
    except Exception as e:
        print(f"⚠️  [Retrieval] semantic search failed, continuing without it: {e}")
        semantic_results = []

    bm25_results: List[Document] = []
    try:
        bm25_results = _get_bm25_index(db).search(query, bm25_top_k, year=year_filter)
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

    # An empty return here (same as the "0 results" case above) is what
    # api.py treats as "insufficient evidence" and routes to a localized
    # no-answer response instead of generation.
    return filter_by_relevance(fused)