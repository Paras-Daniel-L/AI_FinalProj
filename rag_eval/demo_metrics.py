"""
The three metrics shown on the live demo page (/demo).

The demo shows exactly three scores, named as in the paper, but two of them
are computed with a different formula than the thesis evaluation
(rag_eval/ragas_metrics.py, which is NOT changed by this module):

    Groundedness       unchanged:
                       claims in the answer supported by the retrieved
                       context / all claims in the answer
    Context Relevance  computed with the evaluation's Answer Relevance formula:
                       mean cosine similarity between the user's question and
                       N questions reverse-engineered from the ANSWER
    Answer Relevance   computed with the evaluation's Answer Correctness formula:
                       facts in the operator's REFERENCE answer that the answer
                       states correctly / all facts in the reference
                       (contradicted facts are counted and shown separately)

The page shows each metric's actual formula next to its score, so a viewer
always knows what a number measures.

All judging is done by the same independent JUDGE_MODEL and the same judge
prompts as the evaluation (rag_eval/judge.py); every ratio is computed here
in Python from the judge's lists, never asked of the judge as a number.
A score that cannot be computed is None with a plain-language reason — never 0.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

from . import judge, ragas_metrics

GROUNDEDNESS, CONTEXT_RELEVANCE, ANSWER_RELEVANCE = "groundedness", "context_relevance", "answer_relevance"
DEMO_METRICS = (GROUNDEDNESS, CONTEXT_RELEVANCE, ANSWER_RELEVANCE)

DEFINITIONS: Dict[str, Dict[str, str]] = {
    GROUNDEDNESS: {
        "name": "Groundedness",
        "question": "Is every claim in the answer backed by the documents the system retrieved?",
        "formula": "claims supported by the retrieved documents ÷ all claims in the answer",
        "needs": "retrieved documents",
    },
    CONTEXT_RELEVANCE: {
        "name": "Context Relevance",
        "question": "Does the answer stay on the question that was asked?",
        "formula": ("average cosine similarity between the user's question and "
                    f"{judge.JUDGE_ANSWER_QUESTIONS} questions generated from the answer"),
        "needs": "an answer",
    },
    ANSWER_RELEVANCE: {
        "name": "Answer Relevance",
        "question": "Does the answer state the facts of the correct (reference) answer?",
        "formula": ("facts in the reference answer stated correctly ÷ all facts in the reference answer "
                    "(contradicted facts shown separately)"),
        "needs": "a reference answer",
    },
}


def _na(metric: str, reason: str) -> Dict[str, Any]:
    return {"metric": metric, **DEFINITIONS[metric], "score": None, "summary": "", "note": reason, "items": []}


def groundedness(context: str, answer: str, no_context_reason: str = "") -> Dict[str, Any]:
    if not context.strip():
        return _na(GROUNDEDNESS, no_context_reason or
                   "This system retrieved no documents, so there is nothing to check the answer's claims against.")
    result = judge.judge_groundedness(context, answer)
    if result.error:
        return _na(GROUNDEDNESS, f"The judge's reply could not be read ({result.error}).")
    if not result.claims:
        return _na(GROUNDEDNESS, "The answer makes no checkable factual claims (for example, it declines to answer).")
    supported = sum(1 for c in result.claims if c.supported)
    return {
        "metric": GROUNDEDNESS, **DEFINITIONS[GROUNDEDNESS],
        "score": result.score, "note": None,
        "summary": f"{supported} of {len(result.claims)} claims are backed by the retrieved documents",
        "items": [{"text": c.claim, "verdict": "supported" if c.supported else "unsupported"} for c in result.claims],
    }


def context_relevance(query: str, answer: str) -> Dict[str, Any]:
    result = ragas_metrics.score_answer_relevance(query, answer)
    if result.score is None:
        return _na(CONTEXT_RELEVANCE, f"Could not be computed ({result.error}).")
    n = len(result.similarities)
    return {
        "metric": CONTEXT_RELEVANCE, **DEFINITIONS[CONTEXT_RELEVANCE],
        "score": result.score, "note": None,
        "summary": f"average similarity of {n} questions generated from the answer to the user's question",
        "items": [{"text": q, "similarity": s} for q, s in zip(result.generated_questions, result.similarities)],
    }


def answer_relevance(reference: str, answer: str) -> Dict[str, Any]:
    if not reference.strip():
        return _na(ANSWER_RELEVANCE, "No reference answer was entered, so there is no correct answer to compare with.")
    result = judge.judge_answer_correctness(reference, answer)
    if result.error:
        return _na(ANSWER_RELEVANCE, f"The judge's reply could not be read ({result.error}).")
    if not result.facts:
        return _na(ANSWER_RELEVANCE, "The judge found no checkable facts in the reference answer.")
    correct = sum(1 for f in result.facts if f.verdict == "correct")
    summary = f"{correct} of {len(result.facts)} reference facts are stated correctly"
    if result.n_contradicted:
        summary += f"; {result.n_contradicted} contradicted"
    return {
        "metric": ANSWER_RELEVANCE, **DEFINITIONS[ANSWER_RELEVANCE],
        "score": result.score, "note": None, "summary": summary,
        "contradicted": result.n_contradicted,
        "items": [{"text": f.fact, "verdict": f.verdict, "evidence": f.evidence} for f in result.facts],
    }


def score_demo(
    query: str,
    context: str,
    answer: str,
    reference: str = "",
    on_metric: Optional[Callable[[str, str, Optional[Dict[str, Any]]], None]] = None,
    no_context_reason: str = "",
) -> Dict[str, Dict[str, Any]]:
    """Compute the three demo metrics (in parallel: they are independent
    judge calls). `on_metric(metric, status, result)` is called with
    status "running" when one starts and "done" with its result."""
    jobs: Dict[str, Callable[[], Dict[str, Any]]] = {
        GROUNDEDNESS: lambda: groundedness(context, answer, no_context_reason),
        CONTEXT_RELEVANCE: lambda: context_relevance(query, answer),
        ANSWER_RELEVANCE: lambda: answer_relevance(reference, answer),
    }

    def run(metric: str) -> Dict[str, Any]:
        if on_metric:
            on_metric(metric, "running", None)
        try:
            result = jobs[metric]()
        except Exception as e:  # a failing judge call must not lose the other metrics
            result = _na(metric, f"Could not be computed ({type(e).__name__}: {e}).")
        if on_metric:
            on_metric(metric, "done", result)
        return result

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="demo-metric") as pool:
        futures = {m: pool.submit(run, m) for m in DEMO_METRICS}
        return {m: f.result() for m, f in futures.items()}
