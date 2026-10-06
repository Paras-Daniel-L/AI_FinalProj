"""
The live demo page (http://localhost:8000/demo): the chatbot as a chat, but
every answer shows (1) the system's process, step by step, live, following
the system architecture, and (2) three metrics — Groundedness, Context
Relevance and Answer Relevance (see rag_eval/demo_metrics.py for exactly how
each is computed). There are no built-in questions: the operator types any
question, like in the normal chat.

Compare modes (v1.5: the demo compares Sagot AI with REVIE only):
    none   Sagot AI only
    revie  Sagot AI vs REVIE, BIR's chatbot (its answer is pasted in)
The LLM-only (C0) and standard-RAG (C1) baselines were removed in v1.4/v1.5:
after the dense-search floor (v1.3) standard RAG showed no meaningful
difference, so the demo focuses on the project vs REVIE. Their code is in
git tag v1.3 (app/baselines.py) if the ablation is ever needed again.

This page does not change the thesis evaluation tool (/eval) or the chatbot.
Its run endpoint lives under /eval/api/run/ on purpose: app/guard.py locks
that prefix to the host computer or to teammates who enter the team access
code (TEAM_ACCESS_CODE, see SHARING.md), because every run makes paid model
calls (generator, verifier, judge, Jina). Each visitor runs one question at a
time, at most DEMO_MAX_CONCURRENT at once overall.

The run endpoint streams newline-delimited JSON (one event per line):
    {"type": "start",  "systems": [...], "metrics_available": bool, ...}
    {"type": "step",   "system": "sagot", "step": "rerank", "status": "done", "data": {...}}
    {"type": "answer", "system": "sagot", "answer": "...", "sources": [...], ...}
    {"type": "metric", "system": "sagot", "metric": "groundedness", "status": "running"|"done", "result": {...}}
    {"type": "system_done", "system": "sagot", "elapsed_s": 12.3}
    {"type": "done"}
"""

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app import guard, progress
from app.schemas import ComputationState

from . import demo_metrics, judge

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"

SAGOT, REVIE = "sagot", "revie"
COMPARE_MODES: Dict[str, List[str]] = {
    "none": [SAGOT],
    "revie": [SAGOT, REVIE],
}

# Sagot AI replies that are not an answer to a tax question: no metrics.
_NO_METRIC_MODES = {"greeting", "chat", "clarify", "rejected", "computation_input"}
RUN_TIMEOUT_S = 600

DEMO_MAX_CONCURRENT = max(1, int(os.environ.get("DEMO_MAX_CONCURRENT", "3")))

router = APIRouter()
# One run per visitor at a time (a double-click must not double the bill), and
# a few at once overall so teammates testing together don't block each other.
_slots = threading.BoundedSemaphore(DEMO_MAX_CONCURRENT)
_active: set = set()
_active_lock = threading.Lock()


class DemoRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    compare: str = "none"
    reference: str = Field("", max_length=5000)
    revie_answer: str = Field("", max_length=10000)
    computation_state: Optional[ComputationState] = None


def _systems_info() -> Dict[str, Dict[str, Any]]:
    from app.llm import GENERATOR_MODEL, MAX_RAG_RETRIES, VERIFIER_MODEL
    from app.retrieval import DENSE_FLOOR_K
    return {
        SAGOT: {"name": "Sagot AI", "tag": "Our framework",
                "description": (f"Hybrid retrieval (semantic + BM25 + issuance match → RRF → reranker"
                                f"{f', plus the top {DENSE_FLOOR_K} dense hits' if DENSE_FLOOR_K > 0 else ''}), "
                                f"language trigger, {GENERATOR_MODEL} generator, {VERIFIER_MODEL} verifier "
                                f"(up to {MAX_RAG_RETRIES} attempts), fails closed.")},
        REVIE: {"name": "REVIE", "tag": "BIR's chatbot",
                "description": "BIR's own REVIE chatbot. Its answer is pasted in; which documents it used is not visible."},
    }


def _version_label() -> str:
    from app import version
    return version.effective_label()


def _judge_status() -> Dict[str, Any]:
    try:
        return {"judge_model": judge.require_independent_judge(), "judge_error": None}
    except judge.JudgeConfigError as e:
        return {"judge_model": judge.JUDGE_MODEL or None, "judge_error": str(e)}


@router.get("/demo", include_in_schema=False)
def demo_page():
    page = STATIC_DIR / "demo.html"
    if not page.exists():
        raise HTTPException(404, "static/demo.html is missing.")
    return FileResponse(page, media_type="text/html")


@router.get("/team/api/status")
def team_status(request: Request):
    """For static/js/team-access.js: is a team code in use, is the one this
    browser sent right, and is this the host computer (which needs none)."""
    return {"enabled": guard.team_code_enabled(), "valid": guard.has_team_code(request),
            "host": guard.is_host_request(request)}


@router.get("/demo/api/config")
def demo_config(request: Request):
    from app.llm import GENERATOR_MODEL, VERIFIER_MODEL
    can_run = guard.can_run_evaluations(request)
    return {
        "can_run": can_run,
        "team_code_enabled": guard.team_code_enabled(),
        "lock_reason": None if can_run else (
            "Enter the team access code to run questions. Ask the person hosting Sagot AI for it."
            if guard.team_code_enabled() else
            "You are viewing this page through the public link. The demo makes paid model calls, so it only "
            "runs on the host computer at http://localhost:8000/demo."),
        **_judge_status(),
        "version": _version_label(),
        "generator_model": GENERATOR_MODEL,
        "verifier_model": VERIFIER_MODEL,
        "systems": _systems_info(),
        "compare_modes": COMPARE_MODES,
        "metrics": demo_metrics.DEFINITIONS,
    }


# ── One system, run in its own thread ─────────────────────────────────────────

def _run_sagot(req: DemoRequest) -> Dict[str, Any]:
    from fastapi import HTTPException as _HTTPException
    from app.api import run_query
    from app.schemas import QueryRequest
    from .run_evaluation import _rebuild_context

    try:
        response, trace = run_query(QueryRequest(
            query=req.query, bypass_cache=True, computation_state=req.computation_state))
    except _HTTPException as e:
        raise RuntimeError(f"Sagot AI could not answer: {e.detail}") from None
    mode = response.mode
    if mode.startswith("computation"):
        no_ctx = ("This was a tax computation: the figures come from the verified tax-rule files, "
                  "not from retrieved documents, so there are no documents to check claims against.")
    elif response.outcome in ("no_retrieval", "no_index"):
        no_ctx = "Sagot AI found no supporting documents and declined to answer instead of guessing."
    else:
        no_ctx = "No documents are retrieved for this kind of message."
    return {
        "answer": response.answer,
        "context": _rebuild_context(trace.get("retrieved") or []),
        "sources": response.sources,
        "outcome": response.outcome,
        "mode": mode,
        "attempts": response.attempts,
        "language": response.language,
        "classification": response.classification,
        "computation": response.computation.model_dump() if response.computation else None,
        "no_context_reason": no_ctx,
        "skip_metrics": mode in _NO_METRIC_MODES,
    }


def _run_revie(req: DemoRequest) -> Dict[str, Any]:
    progress.emit("question", query=req.query)
    progress.emit("external", source="REVIE (pasted by the operator)", answer=req.revie_answer)
    progress.emit("final", outcome="answered",
                  note="REVIE's retrieval and checks are not visible from outside, so only its answer is scored.")
    return {"answer": req.revie_answer.strip(), "context": "", "sources": [], "outcome": "answered",
            "mode": "external", "skip_metrics": False,
            "no_context_reason": "REVIE does not show which documents (if any) its answer is based on, "
                                 "so there is nothing to check its claims against."}


def _worker(system: str, req: DemoRequest, put: Callable[[Dict[str, Any]], None], metrics_ok: bool) -> None:
    started = time.monotonic()
    try:
        with progress.listening(lambda ev: put({"type": "step", "system": system, **ev})):
            out = _run_sagot(req) if system == SAGOT else _run_revie(req)
        answer_event = {k: v for k, v in out.items() if k not in ("context", "no_context_reason", "skip_metrics")}
        put({"type": "answer", "system": system, **answer_event,
             "has_context": bool(out["context"].strip()), "answer_elapsed_s": round(time.monotonic() - started, 1)})

        if out["skip_metrics"]:
            put({"type": "metrics_skipped", "system": system,
                 "reason": "Metrics are computed for answers to tax questions, not for greetings, "
                           "clarifying questions or computation follow-up questions."})
        elif not metrics_ok:
            put({"type": "metrics_skipped", "system": system,
                 "reason": "The judge model is not configured (JUDGE_MODEL in .env), so the answer is not scored."})
        else:
            demo_metrics.score_demo(
                req.query, out["context"], out["answer"], req.reference,
                on_metric=lambda m, status, result: put(
                    {"type": "metric", "system": system, "metric": m, "status": status, "result": result}),
                no_context_reason=out["no_context_reason"],
            )
    except Exception as e:
        put({"type": "error", "system": system, "message": f"{type(e).__name__}: {e}"})
    finally:
        put({"type": "system_done", "system": system, "elapsed_s": round(time.monotonic() - started, 1)})


def _release(who: str) -> None:
    with _active_lock:
        _active.discard(who)
    _slots.release()


@router.post("/eval/api/run/demo")
def run_demo(req: DemoRequest, request: Request):
    systems = COMPARE_MODES.get(req.compare)
    if systems is None:
        raise HTTPException(422, f"Unknown compare mode {req.compare!r}.")
    if REVIE in systems and not req.revie_answer.strip():
        raise HTTPException(422, "Paste REVIE's answer to this question first.")
    who = guard.client_ip(request)
    with _active_lock:
        if who in _active:
            raise HTTPException(409, "Your previous question is still running. Wait for it to finish.")
        if not _slots.acquire(blocking=False):
            raise HTTPException(429, "The demo is busy answering other testers' questions. Try again in a minute.")
        _active.add(who)

    status = _judge_status()
    events: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    put = events.put
    metrics_ok = status["judge_error"] is None

    def supervise() -> None:
        # The lock is released when every system has finished, not when the
        # browser stops reading: a closed tab must not block the next run.
        try:
            threads = [threading.Thread(target=_worker, args=(s, req, put, metrics_ok),
                                        daemon=True, name=f"demo-{s}") for s in systems]
            for t in threads:
                t.start()
            for t in threads:
                t.join(RUN_TIMEOUT_S)
        finally:
            _release(who)

    try:
        threading.Thread(target=supervise, daemon=True, name="demo-supervisor").start()
    except Exception:
        _release(who)
        raise

    def stream():
        yield json.dumps({"type": "start", "systems": systems, "compare": req.compare,
                          "metrics_available": metrics_ok,
                          "judge_model": status["judge_model"], "judge_error": status["judge_error"],
                          "has_reference": bool(req.reference.strip())}, ensure_ascii=False) + "\n"
        remaining, deadline = set(systems), time.monotonic() + RUN_TIMEOUT_S
        while remaining:
            try:
                event = events.get(timeout=max(0.1, min(15.0, deadline - time.monotonic())))
            except queue.Empty:
                if time.monotonic() >= deadline:
                    yield json.dumps({"type": "error", "system": None,
                                      "message": "The run took too long and was stopped."}) + "\n"
                    break
                yield json.dumps({"type": "ping"}) + "\n"  # keeps the connection alive
                continue
            if event.get("type") == "system_done":
                remaining.discard(event["system"])
            yield json.dumps(event, ensure_ascii=False, default=str) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
