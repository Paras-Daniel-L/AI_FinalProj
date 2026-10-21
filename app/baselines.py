"""
Baseline systems for the live demo's comparison (/demo, rag_eval/demo_web.py).

They exist to show WHERE Sagot AI's hallucination mitigation comes from, by
removing parts of the framework while holding the language model fixed:

    C0  The generator model alone (deepseek/deepseek-v4.1-flash by default):
        no retrieval, no grounding rules, no language trigger, no verifier.
        Shows how much the raw LLM hallucinates on BIR questions.

    C1  Standard RAG: dense (semantic) search only — no BM25, no issuance-id
        match, no RRF fusion, no reranker, no year filter — top C1_TOP_K
        chunks (default 4, LangChain's default k) stuffed into LangChain's
        classic question-answering prompt; one generation, no language
        trigger, no verifier. Shows the effect of retrieval by itself.

    (C2 is REVIE, BIR's own chatbot. It has no API, so its answer is pasted
     into the demo by the operator; nothing in this module runs for it.)

Both baselines use the SAME model, temperature, output cap and provider
pinning as Sagot AI's generator (app/llm.py), and the SAME Chroma index and
embedding model as Sagot AI's semantic retriever. Only the framework differs,
so any difference in the scores is due to the framework, not the model.

Nothing here is used by the chatbot (/query) or by the thesis evaluation
(rag_eval/run_evaluation.py). Each function reports its steps through
app/progress.py so the demo can show them live.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from . import llm_client, progress
from .llm import (
    GENERATION_TEMPERATURE,
    GENERATOR_MAX_TOKENS,
    GENERATOR_MODEL,
    GENERATOR_PROVIDERS,
)

load_dotenv()

CHROMA_PATH = "chroma"  # same index as app/api.py
C1_TOP_K = int(os.environ.get("C1_TOP_K", "4"))

SYSTEM_C0 = "C0"
SYSTEM_C1 = "C1"

# A plain assistant persona: no grounding rules, no language instruction,
# no "say you don't know" rule beyond what a default assistant has.
BASELINE_SYSTEM_PROMPT = "You are a helpful assistant."

# LangChain's classic "stuff" question-answering prompt (the default prompt
# of load_qa_chain / RetrievalQA) — the textbook "standard RAG" template.
STANDARD_RAG_PROMPT = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""


@dataclass
class BaselineResult:
    system: str
    answer: str
    context: str = ""                       # what the model was given ("" for C0)
    sources: List[str] = field(default_factory=list)
    outcome: str = "answered"               # answered | error
    error: Optional[str] = None
    elapsed_ms: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


def _generate(user_prompt: str, role: str) -> llm_client.ChatResult:
    return llm_client.chat(
        BASELINE_SYSTEM_PROMPT,
        user_prompt,
        model=GENERATOR_MODEL,
        temperature=GENERATION_TEMPERATURE,
        max_tokens=GENERATOR_MAX_TOKENS,
        provider_order=GENERATOR_PROVIDERS,
        role=role,
    )


def run_c0(query: str) -> BaselineResult:
    """C0: the question goes straight to the generator model."""
    started = time.monotonic()
    query = " ".join((query or "").split())
    progress.emit("question", query=query)
    progress.emit("llm_only", "running", model=GENERATOR_MODEL)
    try:
        gen = _generate(query, "baseline:c0")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        progress.emit("llm_only", "failed", model=GENERATOR_MODEL, error=err)
        return BaselineResult(SYSTEM_C0, "", outcome="error", error=err,
                              elapsed_ms=int((time.monotonic() - started) * 1000))
    progress.emit("llm_only", model=GENERATOR_MODEL, draft=gen.text, truncated=gen.truncated)
    progress.emit("final", outcome="answered", note="No retrieval and no verifier: the answer is shown as generated.")
    return BaselineResult(SYSTEM_C0, gen.text, elapsed_ms=int((time.monotonic() - started) * 1000),
                          meta={"generator": gen.meta()})


def _open_db():
    from langchain_chroma import Chroma
    from .embeddings import get_embedding_function
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())


def run_c1(query: str, top_k: Optional[int] = None, db=None) -> BaselineResult:
    """C1: dense top-k retrieval + LangChain's standard QA prompt, one pass."""
    from .retrieval import _semantic_search, format_context, source_label, source_labels

    started = time.monotonic()
    k = top_k or C1_TOP_K
    query = " ".join((query or "").split())
    elapsed = lambda: int((time.monotonic() - started) * 1000)
    progress.emit("question", query=query)

    progress.emit("dense", "running", k=k)
    try:
        db = db or _open_db()
        docs = _semantic_search(db, query, k, None)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        progress.emit("dense", "failed", k=k, error=err)
        return BaselineResult(SYSTEM_C1, "", outcome="error", error=err, elapsed_ms=elapsed())
    context_text = format_context(docs)
    progress.emit("dense", k=k, n=len(docs), excerpts=[
        {"label": source_label(d, i).split("] ", 1)[-1],
         "score": round(float(d.metadata.get("_semantic_distance", 0.0)), 4),
         "text": d.page_content}
        for i, d in enumerate(docs, start=1)])

    progress.emit("llm_rag", "running", model=GENERATOR_MODEL, n_excerpts=len(docs))
    try:
        gen = _generate(STANDARD_RAG_PROMPT.format(context=context_text, question=query), "baseline:c1")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        progress.emit("llm_rag", "failed", model=GENERATOR_MODEL, error=err)
        return BaselineResult(SYSTEM_C1, "", context=context_text, sources=source_labels(docs),
                              outcome="error", error=err, elapsed_ms=elapsed())
    progress.emit("llm_rag", model=GENERATOR_MODEL, draft=gen.text, truncated=gen.truncated)
    progress.emit("final", outcome="answered", note="No verifier: the answer is shown as generated.")
    return BaselineResult(SYSTEM_C1, gen.text, context=context_text, sources=source_labels(docs),
                          elapsed_ms=elapsed(), meta={"generator": gen.meta(), "top_k": k})
