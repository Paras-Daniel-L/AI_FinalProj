"""
Jina embeddings with the correct asymmetric-retrieval adapters.

Why this replaces langchain_community.embeddings.JinaEmbeddings
---------------------------------------------------------------
jina-embeddings-v3 has task-specific LoRA adapters. For retrieval, queries
must be embedded with task="retrieval.query" and documents/chunks with
task="retrieval.passage". The community wrapper never sends a `task` field
(its request body is just {"input": ..., "model": ...}), and Jina's docs say
that omitting it means "no specific LoRA adapter will be used" — so both
queries and passages were embedded with the generic base model. This class
sends the right task on each side. It also batches large inputs and retries
transient API errors, neither of which the wrapper did.

IMPORTANT: vectors made with different tasks are not interchangeable. An
index built with the old wrapper must be rebuilt (`python -m app.database
--reset`) — see check_index_compatible().
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

from . import jina_http
from .textproc import chunking_config

load_dotenv()

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
MODEL_NAME = "jina-embeddings-v3"
QUERY_TASK = "retrieval.query"
PASSAGE_TASK = "retrieval.passage"

BATCH_SIZE = int(os.environ.get("JINA_BATCH_SIZE", "64"))
# Also cap each request by ESTIMATED tokens, so one batch of long chunks
# can't take a large bite of the per-minute token budget at once (pacing is
# smoother and a 429 retry re-sends less). See app/jina_http.py.
MAX_TOKENS_PER_REQUEST = int(os.environ.get("JINA_MAX_TOKENS_PER_REQUEST", "20000"))
REQUEST_TIMEOUT = 60
# Retries/backoff for HTTP 429 (rate limit) and 5xx: see app/jina_http.py.

# Recorded next to the Chroma index so a stale index (built without task
# adapters) is detected instead of silently degrading retrieval.
# The chunking settings are part of it too: an index built from legacy
# 800-character chunks must not be mixed with sentence-aligned ones.
EMBEDDING_CONFIG: Dict[str, str] = {
    "model": MODEL_NAME,
    "query_task": QUERY_TASK,
    "passage_task": PASSAGE_TASK,
    **chunking_config(),
}
CONFIG_FILENAME = "embedding_config.json"


class JinaTaskEmbeddings(Embeddings):
    """Jina v3 embeddings: retrieval.passage for documents, retrieval.query for queries."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = MODEL_NAME):
        key = api_key or os.environ.get("JINA_API_KEY")
        if not key:
            raise ValueError("JINA_API_KEY is not set (add it to your .env file).")
        self.model_name = model_name
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {key}",
                "Accept-Encoding": "identity",
                "Content-Type": "application/json",
            }
        )

    def _post(self, texts: List[str], task: str) -> List[List[float]]:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "task": task,
            "input": texts,
        }
        try:
            resp = jina_http.post(
                JINA_API_URL, payload, timeout=REQUEST_TIMEOUT,
                session=self.session, label="Jina embeddings",
            )
        except jina_http.JinaRetryError as e:
            raise RuntimeError(str(e)) from e
        body = resp.json()
        if "data" in body:
            items = sorted(body["data"], key=lambda e: e["index"])
            return [item["embedding"] for item in items]
        raise RuntimeError(
            f"Jina API error (HTTP {resp.status_code}): {body.get('detail') or body}"
        )

    def _embed(self, texts: List[str], task: str) -> List[List[float]]:
        out: List[List[float]] = []
        batch: List[str] = []
        batch_tokens = 0
        for text in texts:
            t = jina_http.estimate_tokens(text)
            if batch and (len(batch) >= BATCH_SIZE or batch_tokens + t > MAX_TOKENS_PER_REQUEST):
                out.extend(self._post(batch, task))
                batch, batch_tokens = [], 0
            batch.append(text)
            batch_tokens += t
        if batch:
            out.extend(self._post(batch, task))
        return out

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts, PASSAGE_TASK)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], QUERY_TASK)[0]


def get_embedding_function() -> JinaTaskEmbeddings:
    return JinaTaskEmbeddings()


# ── Index compatibility guard ─────────────────────────────────────────────

def _config_path(chroma_path: str) -> str:
    return os.path.join(chroma_path, CONFIG_FILENAME)


def write_index_config(chroma_path: str) -> None:
    """Stamp the embedding config next to the index (call after a successful add)."""
    os.makedirs(chroma_path, exist_ok=True)
    with open(_config_path(chroma_path), "w", encoding="utf-8") as f:
        json.dump(EMBEDDING_CONFIG, f)


def check_index_compatible(chroma_path: str, has_documents: bool) -> Optional[str]:
    """
    Returns None if the index is safe to use with the current embedding
    config, otherwise a message explaining how to fix it.

    An index with documents but no config file was built by the old wrapper
    (no task adapters); a config that differs from EMBEDDING_CONFIG was
    built with different settings. Mixing them with new query vectors gives
    silently worse retrieval, so callers should surface this loudly.
    An empty index is always fine.
    """
    if not has_documents:
        return None
    path = _config_path(chroma_path)
    if not os.path.exists(path):
        return (
            "The index was built without Jina task adapters (no "
            f"{CONFIG_FILENAME}). Rebuild it: python -m app.database --reset"
        )
    try:
        with open(path, encoding="utf-8") as f:
            stored = json.load(f)
    except (OSError, ValueError):
        return f"Could not read {CONFIG_FILENAME}. Rebuild: python -m app.database --reset"
    if stored != EMBEDDING_CONFIG:
        return (
            f"The index was built with {stored}, but the code now uses "
            f"{EMBEDDING_CONFIG}. Rebuild: python -m app.database --reset"
        )
    return None
