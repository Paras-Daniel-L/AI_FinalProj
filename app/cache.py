"""
Exact-match answer cache — SQLite, in-process, no server (no Redis/Docker).

Why it exists: one /query can cost up to 2 * MAX_RAG_RETRIES model calls
(generate + verify per attempt). Asking the same question again should not
pay that again.

Why it is deliberately conservative: this is a hallucination-mitigation
system, so the cache must never be a way to serve an answer that the current
pipeline would not produce. Rules:

1. EXACT match on a lightly normalized query — not a semantic/similarity
   match. An embedding cannot reliably tell "RMC 24-2026" from "RMC 25-2026"
   or "1.5%" from "15%"; an exact key can. Normalization only folds case,
   Unicode form, whitespace, curly quotes and leading/trailing punctuation.
   It never touches digits, hyphens, periods or % inside the query.
2. Only VERIFIED answers (mode "rag") are stored. Refusals and errors are
   never cached, so a transient failure can't become permanent.
3. Only first-turn questions (empty conversation history) are cached or
   served: a follow-up like "what about 2025?" depends on the earlier turns,
   which the key doesn't capture.
4. The key also contains (a) a hash of the chunk ids currently in Chroma and
   (b) a fingerprint of every setting that shapes the answer — models,
   temperatures, attempt count, prompt text, retrieval settings, embedding
   config. Change any of them and old entries simply stop matching. Ingesting
   new chunks or resetting the database additionally clears the cache
   (database.py), because re-ingesting can change a chunk's text while keeping
   its id.
5. Entries expire after CACHE_TTL_DAYS (default 7; 0 = never).
6. Any cache error is logged and treated as a miss. The cache can slow
   nothing down and break nothing.

Settings (env vars, all optional):
    CACHE_ENABLED=1        set 0 to turn the cache off entirely
    CACHE_PATH=cache/answers.sqlite3
    CACHE_TTL_DAYS=7

Maintenance (project root):
    python -m app.cache --stats
    python -m app.cache --clear      # do this after editing code such as the
                                     # year-filter regex, which isn't fingerprinted

This module deliberately imports only the standard library at the top so that
database.py can call clear_cache() without pulling in the LLM stack.
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
import unicodedata
from contextlib import closing
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv

load_dotenv()

CACHE_ENABLED: bool = os.environ.get("CACHE_ENABLED", "1").strip() not in ("0", "false", "False", "")
CACHE_PATH: str = os.environ.get("CACHE_PATH", os.path.join("cache", "answers.sqlite3"))
_ttl_days = float(os.environ.get("CACHE_TTL_DAYS", "7"))
CACHE_TTL_SECONDS: float = _ttl_days * 86400 if _ttl_days > 0 else 0.0  # 0 = never expires

_SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    key            TEXT PRIMARY KEY,
    query          TEXT NOT NULL,
    answer         TEXT NOT NULL,
    sources        TEXT NOT NULL,
    classification TEXT NOT NULL,
    language       TEXT,
    created_at     REAL NOT NULL,
    hits           INTEGER NOT NULL DEFAULT 0
)
"""

# Curly quotes -> straight (NFKC already turns non-breaking spaces into spaces).
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
_EDGE_PUNCT = " .,;:!?\"'"


# ── Key construction ─────────────────────────────────────────────────────

def normalize_query(query: str) -> str:
    """
    Conservative normalization for the cache key: NFKC, casefold, curly
    quotes -> straight, collapsed whitespace, and leading/trailing
    punctuation stripped ("What is the VAT rate?" == "what is the vat rate").

    Nothing INSIDE the text is changed: "RMC 24-2026" vs "RMC 25-2026" and
    "1.5%" vs "15%" must stay different keys. Returns "" for a query with no
    content (callers must not cache it).
    """
    text = unicodedata.normalize("NFKC", query or "").translate(_QUOTES).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(_EDGE_PUNCT)


def index_version(chunk_ids: Iterable[str]) -> str:
    """Fingerprint of which chunks are in the index right now."""
    ids = sorted(chunk_ids)
    h = hashlib.sha1()
    for cid in ids:
        h.update(cid.encode("utf-8", "replace"))
        h.update(b"\0")
    return f"{len(ids)}:{h.hexdigest()}"


@lru_cache(maxsize=1)
def config_fingerprint() -> str:
    """
    Hash of everything (other than the query and the index contents) that
    shapes an answer. Imported lazily to keep this module's top-level imports
    stdlib-only. Constant for the life of the process, so computed once.
    """
    from . import llm, prompts, retrieval
    from .embeddings import EMBEDDING_CONFIG

    prompt_text = "\n".join(
        [prompts.SYSTEM_PROMPT, prompts.RAG_PROMPT, prompts.VERIFIER_SYSTEM_PROMPT, prompts.VERIFY_PROMPT]
    )
    parts: Dict[str, Any] = {
        "generator": llm.GENERATOR_MODEL,
        "verifier": llm.VERIFIER_MODEL,
        "generation_temperature": llm.GENERATION_TEMPERATURE,
        "verifier_temperature": llm.VERIFIER_TEMPERATURE,
        "max_attempts": llm.MAX_RAG_RETRIES,
        "max_tokens": [llm.GENERATOR_MAX_TOKENS, llm.VERIFIER_MAX_TOKENS],
        "providers": [llm.GENERATOR_PROVIDERS, llm.VERIFIER_PROVIDERS],
        "prompts": hashlib.sha1(prompt_text.encode("utf-8")).hexdigest(),
        "retrieval": retrieval.retrieval_settings(),
        "embedding": EMBEDDING_CONFIG,
    }
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def make_key(query: str, chunk_ids: Iterable[str]) -> Optional[str]:
    """Cache key for `query` against the current index, or None if not cacheable."""
    normalized = normalize_query(query)
    if not normalized:
        return None
    material = json.dumps(
        [normalized, index_version(chunk_ids), config_fingerprint()], ensure_ascii=False
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


# ── Storage ──────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    directory = os.path.dirname(CACHE_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(CACHE_PATH, timeout=10)
    conn.execute(_SCHEMA)
    return conn


def get(key: str) -> Optional[Dict[str, Any]]:
    """The cached entry for `key`, or None (miss, expired, or any error)."""
    if not CACHE_ENABLED or not key:
        return None
    try:
        with closing(_connect()) as conn:
            row = conn.execute(
                "SELECT answer, sources, classification, language, created_at "
                "FROM answers WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            answer, sources, classification, language, created_at = row
            if CACHE_TTL_SECONDS and time.time() - created_at > CACHE_TTL_SECONDS:
                conn.execute("DELETE FROM answers WHERE key = ?", (key,))
                conn.commit()
                return None
            conn.execute("UPDATE answers SET hits = hits + 1 WHERE key = ?", (key,))
            conn.commit()
            return {
                "answer": answer,
                "sources": json.loads(sources),
                "classification": classification,
                "language": language,
            }
    except Exception as e:  # never let the cache break a query
        print(f"⚠️  [Cache] read failed, treating as a miss: {e}")
        return None


def put(
    key: str,
    query: str,
    answer: str,
    sources: List[str],
    classification: str,
    language: Optional[str],
) -> None:
    """Store a VERIFIED answer. Callers must not pass refusals or errors."""
    if not CACHE_ENABLED or not key:
        return
    try:
        with closing(_connect()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO answers "
                "(key, query, answer, sources, classification, language, created_at, hits) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                (key, normalize_query(query), answer, json.dumps(sources, ensure_ascii=False),
                 classification, language, time.time()),
            )
            conn.commit()
    except Exception as e:
        print(f"⚠️  [Cache] write failed, continuing without caching: {e}")


def clear_cache() -> int:
    """Delete every cached answer. Returns how many were removed (0 if none/no file)."""
    if not os.path.exists(CACHE_PATH):
        return 0
    try:
        with closing(_connect()) as conn:
            removed = conn.execute("DELETE FROM answers").rowcount
            conn.commit()
        if removed:
            print(f"🧹 [Cache] cleared {removed} cached answer(s).")
        return removed
    except Exception as e:
        print(f"⚠️  [Cache] clear failed: {e}")
        return 0


def stats() -> Dict[str, Any]:
    """Entry count and total hits (for the CLI / debugging)."""
    if not os.path.exists(CACHE_PATH):
        return {"entries": 0, "total_hits": 0, "path": CACHE_PATH, "enabled": CACHE_ENABLED}
    with closing(_connect()) as conn:
        entries, hits = conn.execute("SELECT COUNT(*), COALESCE(SUM(hits), 0) FROM answers").fetchone()
    return {"entries": entries, "total_hits": hits, "path": CACHE_PATH, "enabled": CACHE_ENABLED}


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer cache maintenance.")
    parser.add_argument("--clear", action="store_true", help="Delete every cached answer.")
    parser.add_argument("--stats", action="store_true", help="Show entry count and hits.")
    args = parser.parse_args()
    if args.clear:
        print(f"Removed {clear_cache()} cached answer(s).")
    if args.stats or not args.clear:
        print(stats())


if __name__ == "__main__":
    main()
