"""
One rate limiter and retry policy for every Jina API call (embeddings,
reranker, evaluation embeddings).

Why: the free Jina tier allows 100 requests and 100,000 tokens PER MINUTE.
Rebuilding the index sends ~900 chunks (header + text, ~300 tokens each) —
roughly 270,000 tokens — and the old code sent them as fast as it could, so
the second minute's worth of batches hit HTTP 429. The old retry loop then
gave up after 1 + 2 + 4 = 7 seconds, far too short for a per-minute window
to reset, and `python -m app.database --reset` died partway through.

Two layers:
  1. THROTTLE (before sending). A sliding 60-second window of the requests
     and tokens this process has sent. A request waits until it fits under
     JINA_RPM requests and JINA_TPM tokens (each x JINA_SAFETY, default 0.9,
     to leave headroom for Jina counting tokens slightly differently from our
     estimate). After each reply the estimate is replaced by the token count
     Jina reports in `usage`, so the window tracks real usage.
  2. RETRY (after a failure). HTTP 429 and 5xx, timeouts and connection
     errors are retried; the wait honors the server's Retry-After header,
     otherwise backs off exponentially (2, 4, 8, ... capped at backoff_max).
     Any other status (bad key, out of credits, bad request) is returned to
     the caller immediately — retrying can't fix those.

Every wait is printed, so a long pause reads as "rate limited", not "hung".

Limits are per PROCESS. Running the web app and an evaluation at the same
time means two processes sharing one Jina quota; lower JINA_RPM/JINA_TPM for
both (e.g. halve them) if you need to do that.

Settings (env vars):
    JINA_RPM            default 100     requests per minute allowed by your plan
    JINA_TPM            default 100000  tokens per minute allowed by your plan
    JINA_SAFETY         default 0.9     fraction of each limit actually used
    JINA_MAX_RETRIES    default 8       (ingestion/evaluation can wait ~4 min)
    JINA_BACKOFF_MAX    default 60      seconds
"""

import math
import os
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional, Tuple

import requests

JINA_RPM = int(os.environ.get("JINA_RPM", "100"))
JINA_TPM = int(os.environ.get("JINA_TPM", "100000"))
JINA_SAFETY = float(os.environ.get("JINA_SAFETY", "0.9"))
JINA_MAX_RETRIES = int(os.environ.get("JINA_MAX_RETRIES", "8"))
JINA_BACKOFF_MAX = float(os.environ.get("JINA_BACKOFF_MAX", "60"))

WINDOW_S = 60.0
# Jina v3 / reranker-v2 use an XLM-RoBERTa tokenizer: ~4 characters per token
# for clean English, fewer for Filipino and OCR noise. 3 is deliberately
# conservative so the estimate rarely undershoots.
CHARS_PER_TOKEN = 3.0


class JinaRetryError(RuntimeError):
    """Retries ran out on a retryable failure, or the throttle would have had
    to wait longer than the caller allows (max_wait)."""


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text or "") / CHARS_PER_TOKEN))


def estimate_payload_tokens(payload: Dict[str, Any]) -> int:
    """Token estimate for an embeddings payload ({"input": [...]}) or a rerank
    payload ({"query": ..., "documents": [...]}). Jina bills a rerank as one
    (query + document) pair per document, so the query counts once per doc."""
    if "documents" in payload:
        q = estimate_tokens(str(payload.get("query", "")))
        return sum(q + estimate_tokens(str(d)) for d in payload.get("documents") or [])
    inputs = payload.get("input") or []
    if isinstance(inputs, str):
        inputs = [inputs]
    return sum(estimate_tokens(str(t)) for t in inputs)


class RateLimiter:
    """Sliding-window limiter on requests/minute and tokens/minute.
    Thread-safe (uvicorn serves requests from a thread pool). `clock` and
    `sleep` are injectable for tests."""

    def __init__(self, rpm: int, tpm: int, safety: float = 0.9,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.max_requests = max(1, int(rpm * safety))
        self.max_tokens = max(1, int(tpm * safety))
        self._clock, self._sleep = clock, sleep
        self._events: Deque[Tuple[float, int, int]] = deque()   # (sent_at, tokens, seq)
        self._seq = 0
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= WINDOW_S:
            self._events.popleft()

    def _wait_needed(self, tokens: int, now: float) -> float:
        """Seconds until a request of `tokens` fits (0 = now). Caller holds the lock."""
        self._prune(now)
        used = sum(t for _s, t, _q in self._events)
        # A single request bigger than the whole budget can never fit; let it
        # through once the window is empty rather than waiting forever.
        tokens = min(tokens, self.max_tokens)
        if len(self._events) < self.max_requests and used + tokens <= self.max_tokens:
            return 0.0
        # Walk the window oldest-first until enough requests/tokens have aged out.
        freed_tokens, n_events = 0, len(self._events)
        for i, (sent_at, t, _q) in enumerate(self._events):
            freed_tokens += t
            remaining = n_events - (i + 1)
            if remaining < self.max_requests and used - freed_tokens + tokens <= self.max_tokens:
                return max(0.0, sent_at + WINDOW_S - now)
        return WINDOW_S

    def acquire(self, tokens: int, label: str = "Jina", max_wait: Optional[float] = None) -> int:
        """Block until a request of `tokens` fits, record it, and return a
        handle for settle(). Raises JinaRetryError if the wait would exceed
        max_wait (used by live reranking, which falls back instead)."""
        announced = False
        while True:
            with self._lock:
                now = self._clock()
                wait = self._wait_needed(tokens, now)
                if wait <= 0:
                    self._seq += 1
                    self._events.append((now, tokens, self._seq))
                    return self._seq
                n_req, used = len(self._events), sum(t for _s, t, _q in self._events)
            if max_wait is not None and wait > max_wait:
                raise JinaRetryError(
                    f"{label}: rate-limit budget exhausted (would wait {wait:.0f}s, limit {max_wait:.0f}s)"
                )
            if not announced and wait >= 1:
                print(f"⏳ [{label}] pacing for the Jina rate limit "
                      f"({n_req}/{self.max_requests} requests, "
                      f"{used:,}/{self.max_tokens:,} tokens in the last minute); "
                      f"waiting {wait:.0f}s", flush=True)
                announced = True
            self._sleep(min(wait, 5.0) if wait > 0 else 0.05)

    def settle(self, handle: int, actual_tokens: Optional[int]) -> None:
        """Replace a request's estimated tokens with what Jina reported."""
        if actual_tokens is None:
            return
        with self._lock:
            for i, (sent_at, _t, seq) in enumerate(self._events):
                if seq == handle:
                    self._events[i] = (sent_at, int(actual_tokens), seq)
                    return


LIMITER = RateLimiter(JINA_RPM, JINA_TPM, JINA_SAFETY)


def _retry_after(resp: requests.Response) -> Optional[float]:
    value = resp.headers.get("Retry-After", "").strip()
    try:
        return max(0.0, float(value)) if value else None
    except ValueError:
        return None  # an HTTP-date; fall back to exponential backoff


def _reported_tokens(resp: requests.Response) -> Optional[int]:
    try:
        usage = resp.json().get("usage") or {}
        value = usage.get("total_tokens")
        return int(value) if value is not None else None
    except (ValueError, AttributeError, TypeError):
        return None


def post(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    *,
    timeout: float = 60,
    max_retries: Optional[int] = None,
    backoff_max: Optional[float] = None,
    max_wait: Optional[float] = None,
    session: Optional[requests.Session] = None,
    label: str = "Jina",
    limiter: Optional[RateLimiter] = None,
) -> requests.Response:
    """POST through the throttle and the retry policy above. Returns the first
    non-retryable response (the caller checks its body); raises JinaRetryError
    when every attempt hit a retryable failure, or when the throttle would
    have to wait longer than `max_wait` seconds."""
    tries = max(1, max_retries if max_retries is not None else JINA_MAX_RETRIES)
    cap = backoff_max if backoff_max is not None else JINA_BACKOFF_MAX
    lim = limiter or LIMITER
    sender = session.post if session is not None else requests.post
    tokens = estimate_payload_tokens(payload)
    last_error = "unknown error"
    for attempt in range(tries):
        wait: Optional[float] = None
        handle = lim.acquire(tokens, label=label, max_wait=max_wait)
        try:
            resp = sender(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code != 429 and resp.status_code < 500:
                lim.settle(handle, _reported_tokens(resp))
                return resp
            last_error = f"HTTP {resp.status_code}"
            wait = _retry_after(resp)
        except (requests.ConnectionError, requests.Timeout) as e:
            last_error = f"{type(e).__name__}: {e}"
        if attempt == tries - 1:
            break
        if wait is None:
            wait = min(cap, 2.0 * (2 ** attempt))
        wait = min(wait, cap)
        if max_wait is not None and wait > max_wait:
            break
        reason = "rate limited" if last_error == "HTTP 429" else "error"
        print(f"⏳ [{label}] {reason} ({last_error}); waiting {wait:.0f}s before retry "
              f"{attempt + 2}/{tries}", flush=True)
        time.sleep(wait)
    raise JinaRetryError(f"{label} failed after {attempt + 1} attempt(s): {last_error}")
