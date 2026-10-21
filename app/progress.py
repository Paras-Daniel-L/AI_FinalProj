"""
Live progress events for the demo page (rag_eval/demo_web.py, /demo).

The pipeline calls `emit(step, status, **data)` at each stage of the system
architecture (input check, language, intent, retrieval, fusion, reranking,
generation, verification ...). Nothing listens during normal chatbot use or
during the thesis evaluation, so every call is a no-op there and the
pipeline behaves exactly as before: `emit` never raises, never changes a
value and never makes a model or network call.

The demo installs a listener for the duration of one run with
`listening(callback)`. The listener is a context variable, so it belongs to
the thread (or context) that installed it: two systems running at the same
time in two threads (Sagot AI and a baseline) each report to their own
listener and their events never mix.

Event shape (a plain dict, JSON-serializable):
    {"step": "verify", "status": "done", "t_ms": 1834, "data": {...}}
status is "running" (a step started) or "done" / "skipped" / "failed".
"""

import contextvars
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

Listener = Callable[[Dict[str, Any]], None]

_listener: contextvars.ContextVar[Optional[Listener]] = contextvars.ContextVar("sagot_progress_listener", default=None)
_started: contextvars.ContextVar[Optional[float]] = contextvars.ContextVar("sagot_progress_started", default=None)


def active() -> bool:
    """True when a demo listener is installed (lets callers skip building
    a large payload that nobody would read)."""
    return _listener.get() is not None


def emit(step: str, status: str = "done", **data: Any) -> None:
    """Report one pipeline step to the demo listener, if there is one.
    Never raises: a broken listener must not break the answer."""
    listener = _listener.get()
    if listener is None:
        return
    started = _started.get()
    event = {
        "step": step,
        "status": status,
        "t_ms": int((time.monotonic() - started) * 1000) if started else None,
        "data": data,
    }
    try:
        listener(event)
    except Exception as e:  # pragma: no cover - defensive
        print(f"⚠️  [Progress] listener failed on {step}: {e}")


@contextmanager
def listening(listener: Listener) -> Iterator[None]:
    """Send every emit() made in this context (thread) to `listener`."""
    token = _listener.set(listener)
    token_t = _started.set(time.monotonic())
    try:
        yield
    finally:
        _listener.reset(token)
        _started.reset(token_t)


def preview(text: Optional[str], limit: int = 280) -> str:
    """A short single-line excerpt for showing a chunk in the process view."""
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
