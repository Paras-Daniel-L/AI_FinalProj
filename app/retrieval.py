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

from . import progress
from . import rerank as reranker
from .language import AMBIGUOUS_FILIPINO_MARKERS, ENGLISH_MARKERS, FILIPINO_MARKERS
from .schemas import ConvMessage
from .textproc import (
    ids_in_metadata,
    issuance_title,
    overlap_coefficient,
    parse_issuance_ids,
    split_sentences,
)

# ── RRF configuration ────────────────────────────────────────────────────
# All overridable via env vars so tuning doesn't require a code change.
#
# With the reranker ON (RERANKER=jina, the default) retrieval is two-stage:
#   stage 1 (recall):    semantic top-20 + BM25 top-20 + issuance-id matches,
#                        fused by RRF into a candidate pool of 30;
#   stage 2 (precision): a cross-encoder scores every candidate against the
#                        question, near-duplicates are dropped, and only the
#                        chunks that clear the score threshold are kept
#                        (1 to RERANK_MAX_DOCS, not a fixed 5).
# With RERANKER=none the defaults fall back to the original single stage
# (5 + 5 -> RRF top 5), so the old pipeline can be re-run as an ablation.
RERANK_ON: bool = reranker.enabled()
RRF_K: int = int(os.environ.get("RRF_K", "60"))
SEMANTIC_TOP_K: int = int(os.environ.get("SEMANTIC_TOP_K", "20" if RERANK_ON else "5"))
BM25_TOP_K: int = int(os.environ.get("BM25_TOP_K", "20" if RERANK_ON else "5"))
# Without the reranker this is the number of chunks passed to generation;
# with it, this is the size of the candidate pool the reranker scores.
RRF_FINAL_TOP_K: int = int(os.environ.get("RRF_FINAL_TOP_K", "30" if RERANK_ON else "5"))
# Fallback used when the reranker is on but fails for a request (API outage):
# the original behavior — the RRF top 5.
FALLBACK_TOP_K: int = int(os.environ.get("FALLBACK_TOP_K", "5"))

# ── Reranker dynamic cut ─────────────────────────────────────────────────
# Keep a candidate when its rerank score is >= RERANK_MIN_SCORE AND
# >= RERANK_RELATIVE x the best candidate's score; always keep at least
# RERANK_MIN_DOCS and at most RERANK_MAX_DOCS. These defaults are starting
# points — calibrate them on a DEV set (not T-TED) with
#   python -m rag_eval.retrieval_eval --sweep
# which reports gold-source recall vs. chunks kept for a grid of values.
RERANK_MIN_SCORE: float = float(os.environ.get("RERANK_MIN_SCORE", "0.15"))
RERANK_RELATIVE: float = float(os.environ.get("RERANK_RELATIVE", "0.4"))
RERANK_MIN_DOCS: int = int(os.environ.get("RERANK_MIN_DOCS", "1"))
RERANK_MAX_DOCS: int = int(os.environ.get("RERANK_MAX_DOCS", "3"))
# Added to the rerank score of a chunk that belongs to an issuance the user
# named ("ayon sa RR 14-2022"). The question says which document it is
# about; a small boost lets that break ties without overriding a much more
# relevant chunk from another issuance (e.g. one that amends it).
ID_MATCH_BOOST: float = float(os.environ.get("ID_MATCH_BOOST", "0.15"))
ID_MATCH_TOP_K: int = int(os.environ.get("ID_MATCH_TOP_K", "8"))
# Two kept chunks whose word sets overlap this much (|A∩B| / min(|A|,|B|))
# say the same thing (a Digest and the full issuance, or overlapping chunks);
# only the higher-scored one is kept and the slot goes to the next candidate.
DEDUP_OVERLAP: float = float(os.environ.get("DEDUP_OVERLAP", "0.8"))

# ── Evidence compression (sentence level) — OFF by default ───────────────
# Even a perfectly chosen chunk mixes the sentence that answers the question
# with sentences that don't (in the T-TED run, sending ONLY the chunks that
# contained a relevant sentence still capped Context Relevance at ~0.43).
# When on, every sentence of the kept chunks is scored by the reranker and
# only the relevant ones (plus list lead-ins they depend on) are passed to
# the generator. Turn on for an A/B run and check Answer Correctness does not
# drop before adopting it.
EVIDENCE_COMPRESSION: bool = os.environ.get("EVIDENCE_COMPRESSION", "0").strip() in ("1", "true", "True", "yes")
SENTENCE_MIN_SCORE: float = float(os.environ.get("SENTENCE_MIN_SCORE", "0.1"))
SENTENCE_RELATIVE: float = float(os.environ.get("SENTENCE_RELATIVE", "0.3"))
SENTENCE_MIN_KEEP: int = int(os.environ.get("SENTENCE_MIN_KEEP", "2"))

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
    return issuance_title(str(doc.metadata.get("source") or ""))


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
    # Text read from a scan by OCR can contain misread digits (e.g. an RDO code
    # "93B" read as "938"). Flag it so the user knows to check the original PDF.
    if doc.metadata.get("ocr"):
        parts += " · OCR"
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


def _demo_hits(docs: List[Document], score_key: Optional[str] = None, limit: int = 8) -> List[Dict[str, object]]:
    """A short list of retrieved chunks for the demo's process view
    (app/progress.py). Only built when a demo listener is active."""
    out = []
    for i, doc in enumerate(docs[:limit], start=1):
        item: Dict[str, object] = {"label": source_label(doc, i).split("] ", 1)[-1],
                                   "preview": progress.preview(doc.page_content, 160)}
        if score_key and doc.metadata.get(score_key) is not None:
            item["score"] = round(float(doc.metadata[score_key]), 4)
        out.append(item)
    return out


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

# Function words carry no topic: dropped from the QUERY side only (BM25's IDF
# already discounts them in documents). Filipino ones matter most — the
# corpus is English, so "ang"/"ng"/"sa" are rare in it, which gives them a
# HIGH idf and lets a stray "sa" pull in unrelated chunks.
_QUERY_STOPWORDS = (
    set(FILIPINO_MARKERS) | set(AMBIGUOUS_FILIPINO_MARKERS) | set(ENGLISH_MARKERS)
    | {"a", "an", "of", "to", "in", "on", "at", "by", "be", "it", "its", "as", "if",
       "any", "there", "need", "required", "ayon", "ilalim", "ukol", "tungkol",
       "kailangan", "bang", "ba", "nga", "pa", "mag", "ni", "si", "kay"}
)

# Filipino -> English tax vocabulary, used ONLY to add English BM25 terms to
# a Filipino/Taglish query (the embedding model handles cross-lingual
# matching on its own; BM25 can't). Additive: the original words are kept.
# Deliberately limited to core, general tax vocabulary — topic words that
# appear in specific T-TED questions were left out so the evaluation set
# doesn't leak into the system. Extend it from a DEV set or real user logs.
FILIPINO_TAX_GLOSSARY: Dict[str, str] = {
    "buwis": "tax", "kita": "income", "kinita": "income", "sahod": "salary compensation",
    "suweldo": "salary compensation", "sweldo": "salary compensation", "benta": "sales",
    "bentahan": "sale sales", "binebenta": "sale sell", "bumili": "purchase", "bili": "purchase",
    "bayad": "payment fee cost", "bayarin": "payment fee", "multa": "penalty",
    "parusa": "penalty", "resibo": "receipt invoice", "negosyo": "business",
    "negosyante": "business taxpayer", "empleyado": "employee", "manggagawa": "employee worker",
    "porsyento": "percent rate", "porsiyento": "percent rate",
    "taunang": "annual", "taon": "year annual", "buwanan": "monthly",
    "rehistrado": "registered",
    "pagpaparehistro": "registration", "magparehistro": "register registration",
    "ari-arian": "property", "lupa": "land real property", "bahay": "house residential",
   
    "pagbabayad": "payment", "magbayad": "pay payment", "ibawas": "deduct deduction",
    "bawas": "deduction", "kaltas": "withholding", "magkaltas": "withhold withholding",
    "pagkakaltas": "withholding", "exempted": "exempt exemption", "libre": "exempt free",
    "deadline": "deadline due date", "takdang": "due deadline", "petsa": "date",
   
    "kumpanya": "company corporation",
    "korporasyon": "corporation", "dayuhan": "foreign nonresident", "mamamayan": "citizen",
    "pamana": "estate inheritance", "regalo": "donation donor", "donasyon": "donation",
    "upa": "rent lease", "paupahan": "lease rental",
   
   
    "nagbebenta": "seller sell", "mamimili": "buyer customer",
   
}


def tokenize(text: str, query: bool = False) -> List[str]:
    """
    Lowercased word tokens; hyphenated tokens are kept whole AND split, so
    an issuance number like "34-2024" matches "34-2024" and bare "2024".

    query=True additionally drops function words and adds English glossary
    terms for Filipino tax words (see FILIPINO_TAX_GLOSSARY).
    """
    tokens: List[str] = []
    for m in _TOKEN_RE.finditer((text or "").lower()):
        tok = m.group(0)
        if query and tok in _QUERY_STOPWORDS:
            continue
        tokens.append(tok)
        if "-" in tok:
            tokens.extend(tok.split("-"))
        if query and tok in FILIPINO_TAX_GLOSSARY:
            tokens.extend(FILIPINO_TAX_GLOSSARY[tok].split())
    return tokens


def _bm25_text(doc: Document) -> str:
    """A chunk's text for BM25: its context header (issuance title + subject,
    stamped at ingestion) plus its content, so "RMC No. 19-2022" matches
    page 5 of that issuance even though page 5 never names it."""
    header = doc.metadata.get("context_header") or ""
    return f"{header}\n{doc.page_content}" if header else doc.page_content


class _BM25Index:
    def __init__(self, docs: List[Document]):
        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(_bm25_text(d)) for d in docs]) if docs else None

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
        query_tokens = tokenize(query, query=True)
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


def _id_match_search(db: Chroma, query: str, ids: List[str], k: int) -> List[Document]:
    """
    Chunks that BELONG to an issuance the question names ("ayon sa RR
    14-2022"), ranked by semantic similarity within that issuance. Needs the
    `issuance_id` metadata stamped by sentence-mode ingestion; on an index
    built without it this simply finds nothing.
    """
    if not ids:
        return []
    where = {"issuance_id": ids[0]} if len(ids) == 1 else {"issuance_id": {"$in": ids}}
    results = []
    for doc, distance in db.similarity_search_with_score(query, k=k, filter=where):
        meta = dict(doc.metadata)
        meta["_semantic_distance"] = float(distance)
        results.append(Document(page_content=doc.page_content, metadata=meta))
    return results


def _rerank_text(doc: Document) -> str:
    """What the cross-encoder reads for a chunk: header (issuance + subject) + text."""
    return _bm25_text(doc)


def dynamic_cut(
    scored: List[Document],
    min_score: Optional[float] = None,
    relative: Optional[float] = None,
    min_docs: Optional[int] = None,
    max_docs: Optional[int] = None,
    dedup_overlap: Optional[float] = None,
) -> List[Document]:
    """
    `scored` is sorted by descending _rerank_score. Drops near-duplicates of
    an already-kept chunk, then keeps chunks that clear both the absolute and
    the relative threshold, within [min_docs, max_docs]. Parameters default
    to the RERANK_* settings; rag_eval/retrieval_eval.py passes others to
    sweep them offline.
    """
    min_score = RERANK_MIN_SCORE if min_score is None else min_score
    relative = RERANK_RELATIVE if relative is None else relative
    min_docs = RERANK_MIN_DOCS if min_docs is None else min_docs
    max_docs = RERANK_MAX_DOCS if max_docs is None else max_docs
    dedup_overlap = DEDUP_OVERLAP if dedup_overlap is None else dedup_overlap
    if not scored:
        return []
    top = scored[0].metadata["_rerank_score"]
    kept: List[Document] = []
    for doc in scored:
        if len(kept) >= max_docs:
            break
        if any(overlap_coefficient(doc.page_content, k.page_content) >= dedup_overlap for k in kept):
            continue
        score = doc.metadata["_rerank_score"]
        passes = score >= min_score and score >= relative * top
        if passes or len(kept) < min_docs:
            kept.append(doc)
    return kept


def score_candidates(query: str, candidates: List[Document], query_ids: List[str]) -> List[Document]:
    """Cross-encoder scores for every candidate (+ ID_MATCH_BOOST for chunks
    of a named issuance), sorted best first. Raises reranker.RerankError."""
    scores = reranker.rerank(query, [_rerank_text(d) for d in candidates])
    scored = []
    for doc, score in zip(candidates, scores):
        meta = dict(doc.metadata)
        own = meta.get("issuance_id") or ""
        id_match = bool(own) and own in query_ids
        meta["_rerank_raw"] = float(score)
        meta["_id_match"] = id_match
        meta["_rerank_score"] = float(score) + (ID_MATCH_BOOST if id_match else 0.0)
        scored.append(Document(page_content=doc.page_content, metadata=meta))
    scored.sort(key=lambda d: d.metadata["_rerank_score"], reverse=True)
    return scored


def rerank_and_cut(query: str, candidates: List[Document], query_ids: List[str]) -> List[Document]:
    """
    Stage 2 of retrieval: score every candidate with the cross-encoder, add
    ID_MATCH_BOOST to chunks of an issuance the question names, then keep
    only what clears the dynamic cut. Raises reranker.RerankError on failure
    (retrieve_docs falls back to the RRF top-k).
    """
    scored = score_candidates(query, candidates, query_ids)
    kept = dynamic_cut(scored)
    summary = ", ".join(f"{d.metadata['_rerank_score']:.2f}" for d in scored[:8])
    print(
        f"🎯 [Rerank] top scores: {summary} | kept {len(kept)}/{len(scored)} "
        f"(min={RERANK_MIN_SCORE}, rel={RERANK_RELATIVE}, max={RERANK_MAX_DOCS})"
    )
    return kept


_LIST_LEAD_RE = re.compile(r"^\s*(\(?[a-zA-Z0-9]{1,4}[.)]|\([a-zA-Z0-9]{1,4}\)|[•▪●◦\-–])\s")


def compress_evidence(query: str, docs: List[Document]) -> List[Document]:
    """
    Sentence-level evidence selection (EVIDENCE_COMPRESSION=1). Every
    sentence of the kept chunks is scored by the reranker in one call; a
    sentence is kept when it clears SENTENCE_MIN_SCORE and SENTENCE_RELATIVE
    x the best sentence, and the best SENTENCE_MIN_KEEP sentences are always
    kept. A kept list item also keeps the sentence that introduces the list
    (the one ending in ":"), since "(a) ... has not exceeded ₱500,000" means
    nothing without "shall not withhold if:". Omitted runs are marked " … ".

    The generator, the verifier and the evaluation all see the compressed
    text (it is what retrieval returned), so the scores stay honest. The full
    chunk text is kept in metadata["_full_text"] for the trace. A chunk with
    no kept sentence is dropped. On any reranker failure the chunks are
    returned uncompressed.
    """
    per_doc = [split_sentences(d.page_content) for d in docs]
    flat = [(i, j, s) for i, sents in enumerate(per_doc) for j, s in enumerate(sents)]
    if not flat:
        return docs
    try:
        scores = reranker.rerank(query, [s for _i, _j, s in flat])
    except reranker.RerankError as e:
        print(f"⚠️  [Compress] reranker failed, passing chunks uncompressed: {e}")
        return docs

    best = max(scores)
    order = sorted(range(len(flat)), key=lambda n: scores[n], reverse=True)
    keep = {n for n in order[:SENTENCE_MIN_KEEP]}
    keep |= {n for n, sc in enumerate(scores) if sc >= SENTENCE_MIN_SCORE and sc >= SENTENCE_RELATIVE * best}
    keep_pairs = {(flat[n][0], flat[n][1]) for n in keep}

    # A kept list item pulls in the lead-in sentence (ending ":") above it.
    for (i, j) in list(keep_pairs):
        if _LIST_LEAD_RE.match(per_doc[i][j]):
            for back in range(j - 1, -1, -1):
                keep_pairs.add((i, back))
                if per_doc[i][back].rstrip().endswith(":") or not _LIST_LEAD_RE.match(per_doc[i][back]):
                    break

    out: List[Document] = []
    for i, doc in enumerate(docs):
        idx = sorted(j for (d, j) in keep_pairs if d == i)
        if not idx:
            continue
        parts: List[str] = []
        prev = -1
        for j in idx:
            if prev != -1 and j != prev + 1:
                parts.append("…")
            parts.append(per_doc[i][j])
            prev = j
        meta = dict(doc.metadata)
        meta["_full_text"] = doc.page_content
        meta["_compressed"] = True
        meta["_sentences_kept"] = f"{len(idx)}/{len(per_doc[i])}"
        out.append(Document(page_content=" ".join(parts), metadata=meta))
    kept_n = sum(len([1 for (d, _j) in keep_pairs if d == i]) for i in range(len(docs)))
    print(f"🗜️  [Compress] kept {kept_n}/{len(flat)} sentence(s) across {len(out)} chunk(s)")
    return out or docs


def retrieve_candidates(
    query: str,
    year_filter: Optional[str],
    db: Chroma,
    semantic_top_k: int = SEMANTIC_TOP_K,
    bm25_top_k: int = BM25_TOP_K,
    rrf_k: int = RRF_K,
    final_top_k: int = RRF_FINAL_TOP_K,
) -> List[Document]:
    """Stage 1 only (see retrieve_docs): the RRF-fused candidate list."""
    query_ids = parse_issuance_ids(query)
    if query_ids:
        print(f"🔖 [Issuance] query names: {', '.join(query_ids)}")

    semantic_results: List[Document] = []
    progress.emit("semantic", "running", k=semantic_top_k)
    try:
        semantic_results = _semantic_search(db, query, semantic_top_k, year_filter)
        if semantic_results:
            dists = ", ".join(f"{d.metadata['_semantic_distance']:.3f}" for d in semantic_results[:8])
            print(f"📏 [Semantic] distances (best first): {dists}")
    except Exception as e:
        print(f"⚠️  [Retrieval] semantic search failed, continuing without it: {e}")
        semantic_results = []
    if progress.active():
        progress.emit("semantic", n=len(semantic_results), k=semantic_top_k, year=year_filter,
                      hits=_demo_hits(semantic_results, "_semantic_distance"))

    bm25_results: List[Document] = []
    try:
        bm25_results = _get_bm25_index(db).search(query, bm25_top_k, year=year_filter)
    except Exception as e:
        print(f"⚠️  [Retrieval] BM25 search failed, continuing without it: {e}")
        bm25_results = []
    if progress.active():
        progress.emit("bm25", n=len(bm25_results), k=bm25_top_k, year=year_filter,
                      hits=_demo_hits(bm25_results, "_bm25_score"))

    id_results: List[Document] = []
    if query_ids:
        try:
            id_results = _id_match_search(db, query, query_ids, ID_MATCH_TOP_K)
            print(f"🔖 [Issuance] {len(id_results)} chunk(s) from the named issuance(s)")
        except Exception as e:
            print(f"⚠️  [Retrieval] issuance-id search failed, continuing without it: {e}")
        if progress.active():
            progress.emit("issuance_id", ids=query_ids, n=len(id_results), hits=_demo_hits(id_results))

    if not semantic_results and not bm25_results and not id_results:
        print("🔀 [RRF] all retrievers returned 0 results — nothing to fuse")
        return []

    lists, labels = [semantic_results, bm25_results], ["semantic", "bm25"]
    if id_results:
        lists.append(id_results)
        labels.append("issuance_id")
    fused = reciprocal_rank_fusion(lists, k=rrf_k, top_k=final_top_k, source_labels=labels)
    if progress.active():
        progress.emit("fusion", n=len(fused), rrf_k=rrf_k, retrievers=labels,
                      both=sum(1 for d in fused if (d.metadata.get("_retriever_count") or 0) >= 2),
                      hits=_demo_hits(fused, "_rrf_score"))
    return fused


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
    Hybrid retrieval.

    Stage 1 (recall): dense (Chroma semantic) and sparse (BM25) retrieval run
    independently — BOTH restricted to `year_filter` when one is given — plus,
    when the question names an issuance number, a semantic search restricted
    to that issuance's chunks. The ranked lists are fused with
    `reciprocal_rank_fusion()`.

    Stage 2 (precision, when the reranker is on): `rerank_and_cut()` keeps
    only the candidates a cross-encoder judges relevant (1..RERANK_MAX_DOCS),
    then `compress_evidence()` optionally trims them to their relevant
    sentences. If the reranker fails, the RRF top FALLBACK_TOP_K is used —
    the original behavior.

    year_filter is only ever set from an explicit, in-corpus year/citation
    in the query (see classifier.get_year_filter), so applying it to BM25
    too can't hard-exclude the right chunks on an ML misguess the way the
    old classifier-driven filter could.

    A failure in any retriever is caught and logged; it degrades to whatever
    the others found rather than raising.
    """
    fused = retrieve_candidates(query, year_filter, db, semantic_top_k, bm25_top_k, rrf_k, final_top_k)
    if not fused:
        return []
    query_ids = parse_issuance_ids(query)
    use_reranker = RERANK_ON

    if use_reranker and fused:
        n_candidates = len(fused)
        progress.emit("rerank", "running", n_candidates=n_candidates)
        try:
            fused = rerank_and_cut(query, fused, query_ids)
        except reranker.RerankError as e:
            print(f"⚠️  [Rerank] failed, falling back to RRF top {FALLBACK_TOP_K}: {e}")
            fused = fused[:FALLBACK_TOP_K]
            progress.emit("rerank", "failed", error=str(e), fallback_top_k=FALLBACK_TOP_K)
        else:
            if progress.active():
                progress.emit("rerank", n_candidates=n_candidates, kept=len(fused),
                              min_score=RERANK_MIN_SCORE, relative=RERANK_RELATIVE, max_docs=RERANK_MAX_DOCS,
                              hits=_demo_hits(fused, "_rerank_score"))
            if EVIDENCE_COMPRESSION and fused:
                fused = compress_evidence(query, fused)

    # An empty return here (same as the "0 results" case above) is what
    # api.py treats as "insufficient evidence" and routes to a localized
    # no-answer response instead of generation.
    return filter_by_relevance(fused)


def retrieval_settings() -> Dict[str, object]:
    """Every retrieval setting that changes what the generator sees — part
    of the answer-cache fingerprint (app/cache.py) and of evaluation logs."""
    return {
        "rrf_k": RRF_K, "semantic_top_k": SEMANTIC_TOP_K, "bm25_top_k": BM25_TOP_K,
        "rrf_final_top_k": RRF_FINAL_TOP_K, "min_rrf_score": MIN_RRF_SCORE,
        "min_retriever_corroboration": MIN_RETRIEVER_CORROBORATION,
        "max_semantic_distance": MAX_SEMANTIC_DISTANCE,
        "reranker": reranker.RERANK_MODEL if RERANK_ON else "none",
        "rerank_min_score": RERANK_MIN_SCORE, "rerank_relative": RERANK_RELATIVE,
        "rerank_min_docs": RERANK_MIN_DOCS, "rerank_max_docs": RERANK_MAX_DOCS,
        "id_match_boost": ID_MATCH_BOOST, "id_match_top_k": ID_MATCH_TOP_K,
        "dedup_overlap": DEDUP_OVERLAP, "fallback_top_k": FALLBACK_TOP_K,
        "evidence_compression": EVIDENCE_COMPRESSION,
        "sentence_min_score": SENTENCE_MIN_SCORE, "sentence_relative": SENTENCE_RELATIVE,
        "sentence_min_keep": SENTENCE_MIN_KEEP,
    }
