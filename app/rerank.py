"""
Cross-encoder reranking (Jina Reranker API) — the precision stage of retrieval.

Why: hybrid retrieval (semantic + BM25 + RRF) is good at FINDING the right
issuance — in the T-TED run the gold source was the #1 chunk for 95/100
questions — but the pipeline then always passed a fixed 5 chunks, so the
generator (and the Context Relevance judge) saw ~4 off-topic chunks for every
useful one. A bi-encoder / BM25 score can rank, but it can't say "this chunk
actually answers this question" well enough to decide how many to keep. A
cross-encoder reads the question and the chunk TOGETHER and gives a
calibrated-ish relevance score, which is what the dynamic cut in
retrieval.py thresholds on.

Model: jina-reranker-v2-base-multilingual by default — multilingual matters
here because most user questions are Filipino/Taglish while the corpus is
English. Uses the same JINA_API_KEY as the embeddings, and only `requests`.

Settings (env vars):
    RERANKER          "jina" (default) or "none". "none" restores the original
                      fixed top-k behavior exactly (thesis ablation baseline).
    RERANK_MODEL      default jina-reranker-v2-base-multilingual
    RERANK_TIMEOUT    seconds, default 30

Failure policy: rerank() raises RerankError; retrieval.py catches it and falls
back to the original RRF top-k, so a reranker outage degrades quality, never
availability.
"""

import os
from typing import List, Sequence

from dotenv import load_dotenv

from . import jina_http

load_dotenv()

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
RERANKER = os.environ.get("RERANKER", "jina").strip().lower()
RERANK_MODEL = os.environ.get("RERANK_MODEL", "jina-reranker-v2-base-multilingual").strip()
RERANK_TIMEOUT = float(os.environ.get("RERANK_TIMEOUT", "30"))
# Live queries must not hang for minutes on a rate limit — a reranker that
# can't answer quickly falls back to the RRF top-k (see retrieval.py). The
# evaluation scripts raise both values at start-up so an evaluated answer is
# never silently produced by the fallback path.
MAX_RETRIES = int(os.environ.get("RERANK_MAX_RETRIES", "3"))
BACKOFF_MAX = float(os.environ.get("RERANK_BACKOFF_MAX", "8"))
# Longest a LIVE query waits for rate-limit budget before falling back to the
# RRF top-k (None = wait as long as needed; the evaluation scripts use that).
_mw = os.environ.get("RERANK_MAX_WAIT", "10").strip()
MAX_WAIT = float(_mw) if _mw else None
# Jina caps documents per request; sentences for evidence compression can
# exceed a single batch on long contexts.
MAX_DOCS_PER_CALL = 100


class RerankError(RuntimeError):
    pass


def use_evaluation_retries() -> None:
    """Called by the evaluation scripts: wait out rate limits (up to ~4 min)
    instead of falling back, so every evaluated answer used the reranker."""
    global MAX_RETRIES, BACKOFF_MAX, MAX_WAIT
    MAX_RETRIES = max(MAX_RETRIES, jina_http.JINA_MAX_RETRIES)
    BACKOFF_MAX = max(BACKOFF_MAX, jina_http.JINA_BACKOFF_MAX)
    MAX_WAIT = None


def enabled() -> bool:
    return RERANKER not in ("", "none", "off", "0", "false")


def _post(query: str, documents: List[str]) -> List[float]:
    key = os.environ.get("JINA_API_KEY", "").strip()
    if not key:
        raise RerankError("JINA_API_KEY is not set")
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": len(documents),
        "return_documents": False,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        resp = jina_http.post(
            JINA_RERANK_URL, payload, headers, timeout=RERANK_TIMEOUT,
            max_retries=MAX_RETRIES, backoff_max=BACKOFF_MAX, max_wait=MAX_WAIT,
            label="Jina rerank",
        )
    except jina_http.JinaRetryError as e:
        raise RerankError(str(e)) from e
    try:
        body = resp.json()
    except ValueError as e:
        raise RerankError(f"Jina rerank returned non-JSON (HTTP {resp.status_code})") from e
    if "results" not in body:
        raise RerankError(f"Jina rerank error (HTTP {resp.status_code}): {body.get('detail') or body}")
    scores = [0.0] * len(documents)
    for item in body["results"]:
        scores[int(item["index"])] = float(item["relevance_score"])
    return scores


def rerank(query: str, documents: Sequence[str]) -> List[float]:
    """
    Relevance score for each document against `query`, in INPUT order
    (not sorted). Scores are in [0, 1]; higher = more relevant.
    """
    docs = [d if d.strip() else " " for d in documents]
    if not docs:
        return []
    scores: List[float] = []
    for i in range(0, len(docs), MAX_DOCS_PER_CALL):
        scores.extend(_post(query, docs[i : i + MAX_DOCS_PER_CALL]))
    return scores
