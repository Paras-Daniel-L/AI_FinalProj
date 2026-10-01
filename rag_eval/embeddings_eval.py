"""
Jina embeddings for Answer Relevance's cosine-similarity step — separate
from app/embeddings.py on purpose.

app/embeddings.py's JinaTaskEmbeddings is asymmetric by design: it embeds a
QUERY with task="retrieval.query" and a document CHUNK with
task="retrieval.passage", because those two roles are genuinely different
(a question and the passage that answers it are not phrased alike). Answer
Relevance is a different kind of comparison: it asks how similar TWO
QUESTIONS are to each other (the original query and a question reverse-
engineered from the answer) — a symmetric comparison, for which Jina v3
offers a dedicated "text-matching" task. Reusing retrieval.query for both
sides would quietly measure something else (how alike they'd rank as
queries against a passage index, not how alike they are as questions).

This module is a small, independent client for that one task, so
app/embeddings.py — which is production code the live app depends on —
never needs to grow an evaluation-only code path.
"""

import os
import time
from typing import List, Sequence

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
MODEL_NAME = "jina-embeddings-v3"
MATCHING_TASK = "text-matching"
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3


class EmbeddingError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get("JINA_API_KEY", "").strip()
    if not key:
        raise EmbeddingError("JINA_API_KEY is not set (needed for Answer Relevance's embeddings).")
    return key


def embed_for_matching(texts: Sequence[str]) -> List[List[float]]:
    """Embeds each text with Jina's symmetric text-matching task. Order is
    preserved and matches the input order (matches app/embeddings.py's own
    behavior of re-sorting the API's response by its `index` field first)."""
    if not texts:
        return []
    payload = {"model": MODEL_NAME, "task": MATCHING_TASK, "input": list(texts)}
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    last_error = "unknown error"
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(JINA_API_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}"
            else:
                body = resp.json()
                if "data" in body:
                    items = sorted(body["data"], key=lambda e: e["index"])
                    return [item["embedding"] for item in items]
                raise EmbeddingError(f"Jina API error (HTTP {resp.status_code}): {body.get('detail') or body}")
        except (requests.ConnectionError, requests.Timeout) as e:
            last_error = str(e)
        time.sleep(2 ** attempt)
    raise EmbeddingError(f"Jina text-matching embedding failed after {MAX_RETRIES} attempts: {last_error}")


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
