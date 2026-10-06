"""
RAG Chatbot API  –  Sagot AI

Run from the project root with:
    uvicorn main:app --reload --port 8000
http://localhost:8000

Requirements:
    pip install fastapi uvicorn python-multipart
    (plus all deps from the original RAG project)
"""

import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_chroma import Chroma

from . import cache as answer_cache
from . import intent as intent_router
from . import recovery
from . import trace as trace_log
from .classifier import NO_SOURCES_LABEL, get_year_filter, label_from_docs
from .computation import dialogue as computation
from .computation.extract import extract as extract_inputs
from .computation.extract import looks_like_new_question
from .database import add_to_chroma, clear_database, load_documents, split_documents
from .embeddings import check_index_compatible, get_embedding_function
from .greetings import is_greeting
from .guard import guard_middleware
from .language import LanguageResult, detect_language
from .language import language_from_label as _language_from_label
from .llm import OUTCOME_ERROR, OUTCOME_VERIFIED, run_rag
from .prompts import get_greeting_message
from .retrieval import format_context, retrieve_docs, source_label, source_labels
from .sanitize import rejection_message, sanitize_history, sanitize_query
from .schemas import ComputationState, QueryRequest, QueryResponse, StatusResponse

load_dotenv()

app = FastAPI(title="Sagot AI API", version="2.0.0")

# Public-sharing guard: per-IP rate limit, daily cap, /upload+/reset locked to
# the host computer (see app/guard.py).
app.middleware("http")(guard_middleware)

# No CORS middleware on purpose: the UI is served by this same server, so the
# browser never needs cross-origin permission. The old allow_origins=["*"]
# let any website's scripts call this API (including /reset) from the
# owner's own browser.

CHROMA_PATH = "chroma"
DATA_PATH   = "data"

# Conversation history is OFF by default: every question is answered on its
# own. Retrieval never looks at history, so history could only reach the
# generator's prompt — where it (a) made every question after the first in a
# chat skip the answer cache, (b) added tokens to every call, and (c) let the
# generator reuse facts from EARLIER answers that the verifier never sees, so
# the verifier flagged them as unsupported and forced extra retries. Set
# USE_HISTORY=1 to bring it back (the cache then only serves first-turn
# questions). Real follow-ups ("and what about 2025?") need a
# question-rewriting step first; see the open "query rewriter" item.
USE_HISTORY = os.environ.get("USE_HISTORY", "0").strip() in ("1", "true", "True", "yes")

# api.py lives in app/, so the project root (and static/) is one level up.
STATIC_DIR = Path(__file__).parent.parent / "static"

# Serves /static/css/style.css and /static/js/app.js referenced by index.html.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Helpers ────────────────────────────────────────────────────────────────────

_warned_stale_index = False


def _warn_if_stale_index() -> None:
    """Print (once) if the Chroma index was built with different embedding
    settings than the query side now uses — retrieval would silently degrade."""
    global _warned_stale_index
    if _warned_stale_index:
        return
    problem = check_index_compatible(CHROMA_PATH, has_documents=True)
    if problem:
        _warned_stale_index = True
        print(f"\n🚨 [Embeddings] {problem}\n")


def _greeting_response(lang: LanguageResult) -> QueryResponse:
    """
    A bare greeting ("hi", "kumusta ka") with nothing else attached — see
    app/greetings.py's is_greeting() for exactly what qualifies. Returns a
    fixed, localized greeting string; NO LLM call happens on this path.

    Sagot AI has no general-knowledge/open-domain conversation mode
    anymore (see prompts.SYSTEM_PROMPT) — this exists only so a plain
    "hello" doesn't come back as a no-answer refusal.
    Routing and retrieval never run for this path.
    """
    return QueryResponse(
        answer=get_greeting_message(lang.label),
        sources=[],
        classification="Greeting",
        mode="greeting",
        language=lang.label,
    )


def _no_answer_response(
    lang: LanguageResult,
    reason: str = recovery.REASON_NO_EVIDENCE,
    intent: Optional[intent_router.Intent] = None,
    docs=None,
) -> QueryResponse:
    """
    No supported answer: the DB is empty, retrieval found nothing, the
    generator said NO_ANSWER, or no draft passed verification. This used to
    return one fixed "couldn't find" sentence (a dead end). It now returns
    conversational guidance (app/recovery.py): an honest status, a
    clarifying question for the user's topic, example questions, the closest
    documents retrieval found (titles only), and what the bot can do. Still
    no generation call and no tax facts on this path, and the sources are
    NOT listed as if they supported an answer.
    """
    topic = intent.topic if intent else "general"
    if reason == recovery.REASON_NO_EVIDENCE and intent is not None and intent.label == intent_router.OUT_OF_SCOPE:
        reason = recovery.REASON_OUT_OF_SCOPE
    return QueryResponse(
        answer=recovery.guidance(lang.label, reason, topic, docs),
        sources=[],
        classification=NO_SOURCES_LABEL,
        mode=recovery.mode_for(reason),
        language=lang.label,
    )


def _clarify_response(lang: LanguageResult, intent: intent_router.Intent) -> QueryResponse:
    """A message too short to retrieve on ("tax?", "VAT", "deadline?"):
    ask what the user means instead of retrieving on one word."""
    return QueryResponse(
        answer=recovery.guidance(lang.label, recovery.REASON_VAGUE, intent.topic),
        sources=[],
        classification="Clarification needed",
        mode=recovery.mode_for(recovery.REASON_VAGUE),
        language=lang.label,
    )


def _chat_response(lang: LanguageResult) -> QueryResponse:
    """ "What can you do?" / "sino ka?": fixed capability reply, no model call."""
    return QueryResponse(
        answer=recovery.chat_reply(lang.label),
        sources=[],
        classification="General",
        mode="chat",
        language=lang.label,
    )


def _kb_lookup(title: str, page: int) -> Optional[str]:
    """
    Is the knowledge-base page a tax rule points to (rule.kb_references)
    actually in the index? A metadata-only Chroma lookup — no embedding call.
    Returns a source label like "BIR_Income_Tax_FAQ, p.5 (faq)" or None.
    """
    try:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
        hit = db.get(where={"$and": [{"title": title}, {"page": int(page) - 1}]}, include=["metadatas"], limit=1)
        metas = hit.get("metadatas") or []
        if metas:
            return f"{title}, p.{page} ({metas[0].get('year', '')})"
    except Exception as e:
        print(f"⚠️  [Computation] knowledge-base reference lookup failed: {e}")
    return None


def _wants_computation(query: str, intent: intent_router.Intent, state: Optional[ComputationState]) -> bool:
    """
    TAX_COMPUTATION intent, or — while a computation is open — a message that
    carries a value for it ("800,000", "2025", "8%", "actually it's 850k",
    "cancel"). A real new question that merely contains a year ("What is the
    deadline for 2025?") goes to RAG instead, and the open computation is kept.
    """
    if intent.label == intent_router.TAX_COMPUTATION:
        return True
    if state is None:
        return False
    ex = extract_inputs(query)
    if ex.cancel:
        return True
    if looks_like_new_question(query, ex):
        return False
    if ex.has_values:
        return True
    # While inputs are being collected, a short non-question reply is an
    # attempt to answer ("eight hundred thousand", "ok"): let the computation
    # ask again instead of searching the documents for it. A short message
    # naming another tax ("VAT registration") is a question, not a reply.
    return (
        state.status == "collecting" and bool(state.awaiting) and not ex.unsupported_tax
        and len(query.split()) <= 4 and "?" not in query
    )


def _rejected_response(lang: LanguageResult, message: str) -> QueryResponse:
    """
    The input was cleaned and then refused before any retrieval or model call
    (empty, too long, too many questions, too many issuances — see
    app/sanitize.py). Fixed localized text asking the user to narrow the
    question; costs nothing and is never cached.
    """
    return QueryResponse(
        answer=message,
        sources=[],
        classification="Query not processed",
        mode="rejected",
        language=lang.label,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the chatbot UI shell (CSS/JS load separately via the /static mount)."""
    ui_path = STATIC_DIR / "index.html"
    if not ui_path.exists():
        return HTMLResponse(
            "<h2>UI file not found. Expected static/index.html at the project root.</h2>",
            status_code=404
        )
    return HTMLResponse(ui_path.read_text(encoding="utf-8"))


@app.get("/status", response_model=StatusResponse)
def get_status():
    """Return the number of documents currently indexed in ChromaDB."""
    try:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
        doc_count = len(db.get(include=[])["ids"])
    except Exception:
        doc_count = 0
    return StatusResponse(document_count=doc_count, chroma_path=CHROMA_PATH, data_path=DATA_PATH)


def _retrieval_trace(docs) -> list:
    """What retrieval handed to the models, including the excerpt text (a
    re-index can change a chunk's text while keeping its id)."""
    out = []
    for i, doc in enumerate(docs, start=1):
        m = doc.metadata
        out.append({
            "rank": i,
            "label": source_label(doc, i),
            "id": m.get("id"),
            "year": m.get("year"),
            "page": m.get("page"),
            "ocr": bool(m.get("ocr")),
            "rrf_score": m.get("_rrf_score"),
            "retrievers": m.get("_source_retrievers"),
            "semantic_distance": m.get("_semantic_distance"),
            "bm25_score": m.get("_bm25_score"),
            "rerank_score": m.get("_rerank_score"),
            "id_match": m.get("_id_match"),
            "compressed": bool(m.get("_compressed")),
            "sentences_kept": m.get("_sentences_kept"),
            # `text` is exactly what the generator/verifier saw (compressed
            # when EVIDENCE_COMPRESSION=1); the uncompressed chunk is kept
            # separately so a reviewer can check what was trimmed.
            "text": doc.page_content,
            "full_text": m.get("_full_text"),
        })
    return out


def run_query(body: QueryRequest) -> Tuple[QueryResponse, Dict[str, Any]]:
    """
    The whole /query pipeline, callable in-process (the evaluation harness
    uses this directly, without HTTP, the rate limit or the daily cap).

    Returns (response, trace). The trace — outcome, retrieval, every draft
    and verdict, cost — is also appended to logs/queries.jsonl (app/trace.py),
    including when something fails part-way (then outcome = "error" and the
    exception is re-raised; a failed model call becomes HTTPException(500)).

    Trace outcomes: rejected_input | greeting | chat | clarification |
    no_index | cache_hit | no_retrieval | generator_no_answer | verified |
    verification_failed | error, plus the computation outcomes in
    app/computation/dialogue.py (computation_done, computation_needs_input,
    computation_invalid_input, computation_unsupported_year,
    computation_no_rule, computation_not_eligible, computation_cancelled).
    The four "no supported answer" outcomes keep their names (rag_eval counts
    them as refusals); only what the user is shown changed (app/recovery.py).
    """
    started = time.monotonic()
    tr: Dict[str, Any] = {
        "request_id": uuid.uuid4().hex[:12],
        "ts": trace_log.now_iso(),
        "query_raw": body.query,
        "bypass_cache": body.bypass_cache,
        "use_history": USE_HISTORY,
    }
    try:
        return _run_query(body, tr, started)
    except HTTPException:
        raise  # its trace was already written
    except Exception as e:
        tr.update(
            outcome="error",
            error=f"{type(e).__name__}: {e}",
            total_ms=int((time.monotonic() - started) * 1000),
        )
        trace_log.write(tr)
        raise


def _run_query(body: QueryRequest, tr: Dict[str, Any], started: float) -> Tuple[QueryResponse, Dict[str, Any]]:
    request_id = tr["request_id"]

    # Filled in as the request is routed; finish() attaches them to the reply.
    ctx: Dict[str, Any] = {"intent": None, "open_state": None}

    def finish(response: QueryResponse, outcome: str, attempts: int = 0) -> Tuple[QueryResponse, Dict[str, Any]]:
        response.outcome, response.attempts, response.request_id = outcome, attempts, request_id
        if ctx["intent"] is not None:
            response.intent = ctx["intent"].label
        # A computation that is still collecting inputs survives an unrelated
        # turn (a greeting, a RAG question): hand it back so the user can
        # continue, and say so. A completed one is dropped.
        open_state = ctx["open_state"]
        if open_state is not None and response.computation is None and not response.mode.startswith("computation"):
            response.computation = open_state
            response.answer = f"{response.answer}\n\n_{recovery.continue_computation_note(response.language or 'english')}_"
        tr.update(
            outcome=outcome,
            mode=response.mode,
            cached=response.cached,
            answer=response.answer,
            sources=response.sources,
            classification=response.classification,
            total_ms=int((time.monotonic() - started) * 1000),
        )
        tr.setdefault("cost", None)
        trace_log.write(tr)
        return response, tr

    # Step 0a: Sanitize (app/sanitize.py). Cleans the text (invisible and
    # control characters, Unicode form, all whitespace incl. newlines
    # -> single spaces) and refuses a question the 5-excerpt evidence
    # budget can't answer (too long, too many questions, too many
    # issuances) with a fixed localized message — no retrieval, no
    # model call, nothing cached. From here on ONLY the cleaned `query`
    # and `history` are used, never the raw request fields.
    checked = sanitize_query(body.query)
    query = checked.query
    history = sanitize_history(body.history) if USE_HISTORY else []
    tr.update(query=query, history_turns=len(history))
    if query == body.query:
        tr.pop("query_raw")  # only keep the raw text when cleaning changed it

    # Step 0b: Detect the user's language once, from the cleaned query,
    # before anything else runs. This value — not a re-derivation from a
    # possibly-mutated prompt later — is what travels through retrieval
    # into generation. See app/language.py.
    lang = detect_language(query)
    tr.update(language=lang.label, language_hits={"filipino": lang.filipino_hits, "english": lang.english_hits})

    # A computation reply can be very short ("8%") and still be valid input.
    short_computation_reply = (
        checked.reason == "empty" and body.computation_state is not None
        and extract_inputs(query).has_values
    )
    if not checked.ok and not short_computation_reply:
        print(f"🧼 [Sanitize] rejected ({checked.reason}): {checked.info}")
        tr.update(sanitize_reason=checked.reason, sanitize_info=checked.info)
        return finish(_rejected_response(lang, rejection_message(checked, lang.label)), "rejected_input")

    # Step 1: Bare greeting ("hi", "kumusta ka") -> fixed reply, no LLM
    # call, no retrieval. Anything with real content attached to a
    # greeting-looking prefix does NOT match here and falls through to
    # the normal pipeline below.
    state: Optional[ComputationState] = body.computation_state
    if state is not None:
        # A bare reply like "800,000" or "2025" has no language markers; keep
        # answering in the language the computation started in.
        if lang.filipino_hits == 0 and lang.english_hits == 0 and state.language in ("english", "filipino", "taglish"):
            lang = _language_from_label(state.language)
            tr["language"] = lang.label
        tr["computation_state_in"] = state.model_dump()
        if state.status == "collecting":
            ctx["open_state"] = state

    if is_greeting(query):
        return finish(_greeting_response(lang), "greeting")

    # Step 1b: Intent (app/intent.py). CHAT and TAX_COMPUTATION leave the RAG
    # path here; TAX_INFORMATION and OUT_OF_SCOPE both continue to retrieval
    # (the knowledge base, not a word list, decides what is in scope).
    intent = intent_router.classify(query)
    ctx["intent"] = intent
    tr["intent"] = intent.to_dict()

    if intent.label == intent_router.CHAT:
        return finish(_chat_response(lang), "chat")

    if _wants_computation(query, intent, state):
        if intent.label == intent_router.TAX_COMPUTATION and state is not None and state.status == "completed" \
                and not extract_inputs(query).has_values:
            state = None  # "calculate my income tax" again = a new computation
        ctx["open_state"] = None
        turn = computation.handle(query, lang.label, state, kb_lookup=_kb_lookup)
        tr["computation"] = {"outcome": turn.outcome, "state_out": turn.state.model_dump() if turn.state else None,
                             "result": turn.result}
        return finish(QueryResponse(
            answer=turn.answer,
            sources=turn.sources,
            classification=turn.classification,
            mode=turn.mode,
            language=lang.label,
            computation=turn.state,
        ), turn.outcome)

    # Step 1c: Too short to retrieve on ("tax?", "VAT", "deadline?"): ask a
    # clarifying question instead of retrieving on one word.
    if intent.vague:
        return finish(_clarify_response(lang, intent), "clarification")

    # Step 2: Route (year filter from an explicit year/citation in the
    # query, or None for whole-corpus search). Rule-based; the
    # classification LABEL is derived later, from what retrieval found.
    year_filter = get_year_filter(query)
    tr["year_filter"] = year_filter

    # Step 3: Knowledge query -> try RAG. An empty/unreachable DB means
    # there is no evidence to ground an answer in, so this returns a
    # no-answer response — there is no open-domain fallback to drop to.
    try:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
        chunk_ids = db.get(include=[])["ids"]
        db_empty = len(chunk_ids) == 0
    except Exception as e:
        db_empty, db, chunk_ids = True, None, []
        tr["index_error"] = f"{type(e).__name__}: {e}"

    if db_empty or db is None:
        return finish(_no_answer_response(lang, recovery.REASON_NO_INDEX, intent), "no_index")
    tr["index_chunks"] = len(chunk_ids)

    _warn_if_stale_index()

    # Step 3b: Answer cache (exact match on the normalized query — see
    # app/cache.py for why it is deliberately not a semantic cache). With
    # history off (the default) every question is looked up; with
    # USE_HISTORY=1 only first-turn questions are. The key includes the
    # index contents and the model/prompt/retrieval settings, so a changed
    # index or pipeline never serves an old answer.
    cache_key = None
    if not answer_cache.CACHE_ENABLED:
        print("💾 [Cache] off (CACHE_ENABLED=0)")
        tr["cache"] = "off"
    elif body.bypass_cache:
        print("💾 [Cache] bypassed by this request")
        tr["cache"] = "bypassed"
    elif history:
        print("💾 [Cache] skipped — this is a follow-up and history is in use (USE_HISTORY=1)")
        tr["cache"] = "skipped"
    else:
        cache_key = answer_cache.make_key(query, chunk_ids)
        hit = answer_cache.get(cache_key) if cache_key else None
        if hit:
            print("💾 [Cache] hit — serving a previously verified answer, no model calls")
            tr["cache"] = "hit"
            return finish(QueryResponse(
                answer=hit["answer"],
                sources=hit["sources"],
                classification=hit["classification"],
                mode="rag",
                language=hit["language"] or lang.label,
                cached=True,
            ), "cache_hit")
        print("💾 [Cache] miss — running the full pipeline")
        tr["cache"] = "miss"

    # Step 4: Retrieve documents (RRF-fused, then coarse-relevance
    # filtered — see retrieve_docs()/filter_by_relevance() in
    # retrieval.py). Empty here means either nothing was retrieved or
    # nothing cleared the relevance bar.
    t_retrieval = time.monotonic()
    combined = retrieve_docs(query, year_filter, db)
    tr["retrieval_ms"] = int((time.monotonic() - t_retrieval) * 1000)
    tr["retrieved"] = _retrieval_trace(combined)

    if not combined:
        return finish(_no_answer_response(lang, recovery.REASON_NO_EVIDENCE, intent), "no_retrieval")

    # Step 4b: Label the query from the retrieved evidence — which
    # corpus categories (year folders / faq) the chunks came from.
    label = label_from_docs(combined)

    # Step 5: Generate -> verify loop (app/llm.py run_rag), in the user's
    # language, on numbered labeled excerpts ([1] RMC No. 34-2024, p.3
    # (2024)) so the answer can cite sources and the verifier can check
    # each citation. The result says exactly what happened: verified on
    # attempt n, generator said NO_ANSWER, all attempts rejected, or error.
    context_text = format_context(combined)
    rag = run_rag(query, history, context_text, language=lang)
    tr["rag"] = rag.to_dict()
    tr["cost"] = rag.cost
    n_attempts = len(rag.attempts)

    if rag.outcome == OUTCOME_ERROR:
        response = _no_answer_response(lang, recovery.REASON_NO_EVIDENCE, intent)
        finish(response, "error", n_attempts)
        raise HTTPException(status_code=500, detail=rag.error or "Model call failed.")

    if rag.outcome != OUTCOME_VERIFIED:
        # generator_no_answer or verification_failed: conversational guidance
        # (app/recovery.py), never the unverified draft. The retrieved
        # documents are named only as "closest documents", not as sources
        # supporting an answer.
        reason = recovery.REASON_UNVERIFIED if rag.outcome == "verification_failed" else recovery.REASON_NO_EVIDENCE
        return finish(_no_answer_response(lang, reason, intent, combined), rag.outcome, n_attempts)

    # Same numbering as the context, so a [n] in the answer = sources[n-1].
    sources = source_labels(combined)

    # Only a verified answer ever reaches this point, so it is safe to
    # store. Refusals and errors are never cached.
    if cache_key:
        answer_cache.put(cache_key, query, rag.answer, sources, label, lang.label)

    return finish(QueryResponse(
        answer=rag.answer,
        sources=sources,
        classification=label,
        mode="rag",
        language=lang.label,
    ), OUTCOME_VERIFIED, n_attempts)


@app.post("/query", response_model=QueryResponse)
def query_endpoint(body: QueryRequest):
    """
    Grounded BIR-tax RAG endpoint. No general-knowledge fallback exists:
    every non-greeting query is routed (explicit-year filter, see
    classifier.py), retrieved, labeled from the retrieved documents'
    own metadata, then answered strictly from those documents or refused
    via the canned no-answer response. See run_query().
    """
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        response, _trace = run_query(body)
        return response
    except HTTPException:
        raise
    except Exception as e:  # already traced by run_query()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), category: str = Form(default="")):
    """
    Upload a PDF and index it into ChromaDB.

    `category` is optional and maps to the same folder-based `year` tag
    load_documents() uses for everything else (e.g. "2022", "2026", "faq").
    Pass one of the existing YEAR_FOLDERS values if this document should
    stay reachable through the classifier's year_filter; leave it blank
    to drop the file straight into data/ root, tagged "uncategorized" by
    load_documents() — still fully searchable, just not classifier-
    filterable by year.

    Filenames and categories come from the client, so both are reduced to
    a bare basename via Path(...).name before touching the filesystem —
    otherwise a filename like "../../x.pdf" or a category like "../.env"
    could write outside DATA_PATH.
    """
    safe_name = Path(file.filename or "").name
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_category = Path(category or "").name
    dest_dir = Path(DATA_PATH) / safe_category if safe_category else Path(DATA_PATH)
    os.makedirs(dest_dir, exist_ok=True)
    dest = dest_dir / safe_name

    with open(dest, "wb") as f:
        f.write(await file.read())

    try:
        docs   = load_documents()
        chunks = split_documents(docs)
        add_to_chroma(chunks)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

    return {
        "message": f"'{safe_name}' uploaded and indexed successfully.",
        "category": safe_category or "uncategorized",
    }


@app.delete("/reset")
def reset_database():
    """Wipe the ChromaDB vector store."""
    clear_database()
    return {"message": "Database cleared successfully."}
