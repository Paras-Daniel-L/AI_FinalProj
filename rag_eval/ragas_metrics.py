"""
The three RAGAS metrics from the SOP's Statistical Treatment section, wired
together: Groundedness and Context Relevance come straight from judge.py;
Answer Relevance additionally needs the embedding step in embeddings_eval.py.

    Groundedness     = |claims in the answer supported by retrieved context| / |total claims|
    Context Relevance = |sentences in retrieved context relevant to the query| / |total sentences|
    Answer Relevance  = mean cosine similarity(embed(question reverse-engineered
                        from the answer), embed(original query)), averaged over
                        JUDGE_ANSWER_QUESTIONS generated questions

Plus one SUPPLEMENTARY metric, only when a ground truth is given:
    Answer Correctness = |ground-truth facts the answer states correctly| /
                         |ground-truth facts|   (contradicted facts counted
                         separately — see judge.py)

Every ratio here is computed in this file from a judge-returned LIST's
length — never asked of an LLM as a number (see json_util.py's docstring).

score_response() is the one function the evaluation runner needs: given one
(query, context, answer) triple, from EITHER system (Sagot AI or a manually
transcribed REVIE answer), it returns all three scores plus every
intermediate judgment, so a thesis appendix can show its work.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from . import embeddings_eval, judge


@dataclass
class ScoredResponse:
    query: str
    context: str
    answer: str
    groundedness: judge.GroundednessResult
    context_relevance: judge.ContextRelevanceResult
    answer_relevance: judge.AnswerRelevanceResult
    errors: List[str] = field(default_factory=list)
    ground_truth: str = ""
    # Supplementary (not one of the SOP's three): the answer vs. the ground
    # truth. An error-state (score None) when no ground truth was given.
    answer_correctness: judge.AnswerCorrectnessResult = field(
        default_factory=lambda: judge.AnswerCorrectnessResult(error="no ground truth was provided")
    )

    def to_row(self) -> Dict[str, Any]:
        """Flat dict for a CSV/DataFrame — one row per scored response. Scores
        that are None (undefined for this item — see each Result's docstring)
        stay None here rather than becoming 0.0, so aggregation can exclude
        them explicitly instead of silently pulling an average down."""
        return {
            "query": self.query,
            "groundedness": self.groundedness.score,
            "n_claims": len(self.groundedness.claims),
            "n_claims_supported": sum(1 for c in self.groundedness.claims if c.supported),
            "context_relevance": self.context_relevance.score,
            "n_sentences": len(self.context_relevance.sentences),
            "n_sentences_relevant": sum(1 for s in self.context_relevance.sentences if s.relevant),
            "answer_relevance": self.answer_relevance.score,
            "n_reverse_questions": len(self.answer_relevance.generated_questions),
            "answer_correctness": self.answer_correctness.score,
            "n_facts": len(self.answer_correctness.facts),
            "n_facts_correct": sum(1 for f in self.answer_correctness.facts if f.verdict == "correct"),
            "n_facts_contradicted": self.answer_correctness.n_contradicted,
            "errors": "; ".join(self.errors) or None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Full detail (every claim, sentence and reverse question with its
        judgment) — for an appendix or for re-checking a surprising score."""
        return {
            "query": self.query,
            "context": self.context,
            "answer": self.answer,
            "groundedness": asdict(self.groundedness),
            "context_relevance": asdict(self.context_relevance),
            "answer_relevance": asdict(self.answer_relevance),
            "ground_truth": self.ground_truth,
            "answer_correctness": asdict(self.answer_correctness),
            "errors": self.errors,
        }


def scored_from_dict(d: Dict[str, Any]) -> ScoredResponse:
    """Rebuild a ScoredResponse from its to_dict() form (a checkpointed unit),
    so a saved result renders exactly like a live one. Units scored before
    Answer Correctness existed simply come back without it (score None)."""
    g, c, a = d.get("groundedness") or {}, d.get("context_relevance") or {}, d.get("answer_relevance") or {}
    ac = d.get("answer_correctness")
    return ScoredResponse(
        query=d.get("query", ""), context=d.get("context", ""), answer=d.get("answer", ""),
        groundedness=judge.GroundednessResult(
            claims=[judge.ClaimJudgement(**x) for x in g.get("claims", [])],
            raw=g.get("raw", ""), error=g.get("error")),
        context_relevance=judge.ContextRelevanceResult(
            sentences=[judge.SentenceJudgement(**x) for x in c.get("sentences", [])],
            raw=c.get("raw", ""), error=c.get("error")),
        answer_relevance=judge.AnswerRelevanceResult(
            original_query=a.get("original_query", ""), generated_questions=a.get("generated_questions", []),
            similarities=a.get("similarities", []), raw=a.get("raw", ""), error=a.get("error")),
        errors=d.get("errors", []),
        ground_truth=d.get("ground_truth", ""),
        answer_correctness=(judge.AnswerCorrectnessResult(
            facts=[judge.FactJudgement(**x) for x in ac.get("facts", [])],
            raw=ac.get("raw", ""), error=ac.get("error"))
            if ac else judge.AnswerCorrectnessResult(error="scored before Answer Correctness was added")),
    )


def score_answer_relevance(query: str, answer: str, n: int = judge.JUDGE_ANSWER_QUESTIONS) -> judge.AnswerRelevanceResult:
    questions = judge.generate_reverse_questions(answer, n=n)
    if not questions:
        return judge.AnswerRelevanceResult(
            original_query=query, error="judge did not return any reverse-engineered questions"
        )
    try:
        vectors = embeddings_eval.embed_for_matching([query] + questions)
    except embeddings_eval.EmbeddingError as e:
        return judge.AnswerRelevanceResult(
            original_query=query, generated_questions=questions, error=f"embedding failed: {e}"
        )
    query_vec, question_vecs = vectors[0], vectors[1:]
    similarities = [embeddings_eval.cosine_similarity(query_vec, v) for v in question_vecs]
    return judge.AnswerRelevanceResult(
        original_query=query, generated_questions=questions, similarities=similarities,
    )


def score_response(
    query: str, context: str, answer: str, n_reverse_questions: int = judge.JUDGE_ANSWER_QUESTIONS,
    ground_truth: str = "",
) -> ScoredResponse:
    """
    Score one response on all three metrics. `context` should be the exact
    retrieved-excerpt text the answer was (or, for REVIE, would have been)
    grounded in — for Sagot AI, that's the `text` fields from a trace's
    `retrieved` list (see app/trace.py), joined the same way format_context()
    does. For REVIE, there usually IS no retrieved context to check against
    (see the note in ragas_metrics.md / the conversation with the user about
    what REVIE actually is) — pass context="" and Groundedness/Context
    Relevance will both correctly come back as None (undefined) rather than
    a misleading 0.0; Answer Relevance still runs, since it only needs the
    query and the answer.
    """
    errors: List[str] = []

    if context.strip():
        groundedness = judge.judge_groundedness(context, answer)
        context_relevance = judge.judge_context_relevance(query, context)
    else:
        groundedness = judge.GroundednessResult(error="no retrieved context was provided")
        context_relevance = judge.ContextRelevanceResult(error="no retrieved context was provided")
    if groundedness.error:
        errors.append(f"groundedness: {groundedness.error}")
    if context_relevance.error:
        errors.append(f"context_relevance: {context_relevance.error}")

    answer_relevance = score_answer_relevance(query, answer, n=n_reverse_questions)
    if answer_relevance.error:
        errors.append(f"answer_relevance: {answer_relevance.error}")

    # Supplementary Answer Correctness — only when a ground truth is given.
    # Its absence is not an error (the three SOP metrics don't need it).
    if ground_truth.strip():
        answer_correctness = judge.judge_answer_correctness(ground_truth, answer)
        if answer_correctness.error:
            errors.append(f"answer_correctness: {answer_correctness.error}")
    else:
        answer_correctness = judge.AnswerCorrectnessResult(error="no ground truth was provided")

    return ScoredResponse(
        query=query, context=context, answer=answer,
        groundedness=groundedness, context_relevance=context_relevance,
        answer_relevance=answer_relevance, errors=errors,
        ground_truth=ground_truth, answer_correctness=answer_correctness,
    )
