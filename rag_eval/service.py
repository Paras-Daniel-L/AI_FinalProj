"""
Evaluation logic shared by the web tool (rag_eval/web.py): evaluate ONE
question for one or two chatbots, and summarize a package of them.

Three modes:
    "sagot"     Sagot AI is run live on the question; its own retrieved
                excerpts are the context. The user supplies the question and
                (optionally) the ground truth and expected language.
    "external"  Another chatbot (e.g. BIR's REVIE). Nothing is run — the
                user pastes that chatbot's answer (and its context, if it
                exposes one; most don't).
    "compare"   Both, side by side, on the same question and ground truth.

Every score that can't be computed stays None, with a plain-language reason
in `notes` — the same rule as ragas_metrics.py/stats.py: a missing score is
never shown as 0 (see those modules' docstrings for why that matters).
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.language import detect_language

from . import ragas_metrics, stats

MODE_SAGOT, MODE_EXTERNAL, MODE_COMPARE = "sagot", "external", "compare"
MODES = (MODE_SAGOT, MODE_EXTERNAL, MODE_COMPARE)
SAGOT_NAME = "Sagot AI"

METRICS = ("groundedness", "context_relevance", "answer_relevance", "answer_correctness")
SOP_METRICS = ("groundedness", "context_relevance", "answer_relevance")

# Sagot AI outcomes that mean "the chatbot declined to answer" (the canned
# no-answer message) — reported as a refusal rate next to the scores.
REFUSAL_OUTCOMES = {"verification_failed", "generator_no_answer", "no_retrieval", "no_index"}

_ENGLISH = {"english", "en", "eng", "0"}
_NON_ENGLISH = {"taglish", "filipino", "tagalog", "tl", "fil", "non-english", "non_english", "nonenglish", "1"}


def parse_expected_language(value: Any) -> Optional[int]:
    """'english' -> 0, 'taglish'/'filipino'/... -> 1, blank -> None. Raises
    ValueError on anything else, so a typo is reported instead of silently
    becoming a wrong ground-truth label for the trigger metrics."""
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in _ENGLISH:
        return 0
    if text in _NON_ENGLISH:
        return 1
    raise ValueError(f"expected_language must be 'english' or 'taglish' (got {value!r})")


@dataclass
class EvalInput:
    question: str
    ground_truth: str = ""
    expected_language: Optional[int] = None   # 0 = English, 1 = Taglish/Filipino, None = not given
    other_answer: str = ""
    other_context: str = ""
    id: str = ""


def _notes(scored: ragas_metrics.ScoredResponse, has_context: bool, has_ground_truth: bool) -> Dict[str, str]:
    """Why each missing score is missing, in words a panel member can read."""
    notes: Dict[str, str] = {}
    if scored.groundedness.score is None:
        if not has_context:
            notes["groundedness"] = "No context was provided, so there is nothing to check the answer's claims against."
        elif scored.groundedness.error:
            notes["groundedness"] = f"The judge's reply could not be read ({scored.groundedness.error})."
        else:
            notes["groundedness"] = "The answer makes no checkable factual claims (for example, it declines to answer)."
    if scored.context_relevance.score is None:
        if not has_context:
            notes["context_relevance"] = "No context was provided, so there is no retrieval to rate."
        else:
            notes["context_relevance"] = f"The judge's reply could not be read ({scored.context_relevance.error})."
    if scored.answer_relevance.score is None:
        notes["answer_relevance"] = f"Could not be computed ({scored.answer_relevance.error})."
    if scored.answer_correctness.score is None:
        if (scored.answer_correctness.error or "").startswith("scored before"):
            notes["answer_correctness"] = ("Scored before Answer Correctness was added. Re-run "
                                           "`python -m rag_eval.run_evaluation --subset all` to add it "
                                           "(the chatbot is not called again).")
        elif not has_ground_truth:
            notes["answer_correctness"] = "No ground truth was provided, so there is no correct answer to compare with."
        elif scored.answer_correctness.error:
            notes["answer_correctness"] = f"The judge's reply could not be read ({scored.answer_correctness.error})."
        else:
            notes["answer_correctness"] = "The judge found no checkable facts in the ground truth."
    return notes


def side_from_scored(system: str, name: str, scored: ragas_metrics.ScoredResponse,
                     meta: Optional[Dict[str, Any]] = None, elapsed_s: Optional[float] = None) -> Dict[str, Any]:
    """One chatbot's evaluated answer, in the shape the web UI renders. Also
    used to show the offline thesis run's checkpointed units the same way."""
    row = scored.to_row()
    meta = meta or {}
    outcome = meta.get("outcome")
    return {
        "system": system,
        "system_name": name,
        "answer": scored.answer,
        "context": scored.context,
        "metrics": {m: row.get(m) for m in METRICS},
        "counts": {k: row.get(k) for k in (
            "n_claims", "n_claims_supported", "n_sentences", "n_sentences_relevant",
            "n_reverse_questions", "n_facts", "n_facts_correct", "n_facts_contradicted")},
        "details": {
            "claims": [vars(c) for c in scored.groundedness.claims],
            "sentences": [vars(s) for s in scored.context_relevance.sentences],
            "reverse_questions": [
                {"question": q, "similarity": sim}
                for q, sim in zip(scored.answer_relevance.generated_questions, scored.answer_relevance.similarities)
            ],
            "facts": [vars(f) for f in scored.answer_correctness.facts],
        },
        "notes": _notes(scored, bool(scored.context.strip()), bool(scored.ground_truth.strip())),
        "outcome": outcome,
        "refused": outcome in REFUSAL_OUTCOMES if outcome else None,
        "meta": {k: v for k, v in meta.items() if k not in ("answer", "context")},
        "elapsed_s": elapsed_s,
    }


def _run_sagot(question: str) -> Dict[str, Any]:
    from .run_evaluation import run_sagot  # lazy: imports the live pipeline (Chroma, embeddings)
    return run_sagot(question)


def evaluate_side(system: str, inp: EvalInput, other_name: str = "Other chatbot") -> Dict[str, Any]:
    started = time.monotonic()
    name = SAGOT_NAME if system == MODE_SAGOT else (other_name or "Other chatbot")
    try:
        if system == MODE_SAGOT:
            sagot = _run_sagot(inp.question)
            answer, context, meta = sagot["answer"], sagot["context"], sagot
        else:
            if not inp.other_answer.strip():
                raise ValueError(f"No answer from {name} was provided for this question.")
            answer, context, meta = inp.other_answer, inp.other_context, {}
        scored = ragas_metrics.score_response(inp.question, context, answer, ground_truth=inp.ground_truth)
        return side_from_scored(system, name, scored, meta, round(time.monotonic() - started, 1))
    except Exception as e:  # one failing side must not lose the other side's result
        return {"system": system, "system_name": name, "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(time.monotonic() - started, 1)}


def language_check(inp: EvalInput) -> Dict[str, Any]:
    """Sagot AI's language trigger on this question (free — pure Python)."""
    result = detect_language(inp.question)
    predicted = 0 if result.label == "english" else 1
    return {
        "detected_label": result.label,
        "predicted_non_english": bool(predicted),
        "expected_non_english": None if inp.expected_language is None else bool(inp.expected_language),
        "correct": None if inp.expected_language is None else predicted == inp.expected_language,
    }


def evaluate_item(inp: EvalInput, mode: str, other_name: str = "Other chatbot") -> Dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    systems = {MODE_SAGOT: [MODE_SAGOT], MODE_EXTERNAL: [MODE_EXTERNAL],
               MODE_COMPARE: [MODE_SAGOT, MODE_EXTERNAL]}[mode]
    return {
        "id": inp.id,
        "question": inp.question,
        "ground_truth": inp.ground_truth,
        "expected_language": inp.expected_language,
        "sides": [evaluate_side(s, inp, other_name) for s in systems],
        # Language detection is Sagot AI's own component, so it is only
        # measured when Sagot AI is one of the systems under test.
        "language": language_check(inp) if MODE_SAGOT in systems else None,
    }


# ── Package summary ─────────────────────────────────────────────────────────

def _mean(values: List[Optional[float]]) -> Optional[float]:
    real = [v for v in values if v is not None]
    return sum(real) / len(real) if real else None


def _side_summary(sides: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [s for s in sides if "error" not in s]
    judged = [s for s in ok if s["metrics"]["answer_correctness"] is not None]
    with_outcome = [s for s in ok if s.get("outcome")]
    return {
        "system_name": sides[0]["system_name"] if sides else "",
        "n": len(sides),
        "n_errors": len(sides) - len(ok),
        "metrics": {
            m: {"mean": _mean([s["metrics"][m] for s in ok]),
                "n_scored": sum(1 for s in ok if s["metrics"][m] is not None)}
            for m in METRICS
        },
        "contradictions": {
            "judged": len(judged),
            "with_contradiction": sum(1 for s in judged if (s["counts"]["n_facts_contradicted"] or 0) > 0),
        },
        # Only Sagot AI reports an outcome; another chatbot's refusals can't be
        # detected automatically, so its refusal count is None (not 0).
        "refusals": ({"n": len(with_outcome), "refused": sum(1 for s in with_outcome if s["refused"])}
                     if with_outcome else None),
    }


def summarize(items: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    systems = {MODE_SAGOT: [MODE_SAGOT], MODE_EXTERNAL: [MODE_EXTERNAL],
               MODE_COMPARE: [MODE_SAGOT, MODE_EXTERNAL]}[mode]
    per_system = {}
    for i, system in enumerate(systems):
        per_system[system] = _side_summary([it["sides"][i] for it in items if len(it["sides"]) > i])

    trigger = None
    if MODE_SAGOT in systems:
        labeled = [it["language"] for it in items if it.get("language") and it["language"]["expected_non_english"] is not None]
        if labeled:
            cm = stats.confusion_metrics(
                [int(l["expected_non_english"]) for l in labeled],
                [int(l["predicted_non_english"]) for l in labeled],
            )
            trigger = {**cm.__dict__, "min_recall": stats.MIN_TRIGGER_RECALL,
                       "max_fpr": stats.MAX_FALSE_POSITIVE_RATE, "n_unlabeled": len(items) - len(labeled)}

    comparison = None
    if mode == MODE_COMPARE:
        comparison = {}
        for m in METRICS:
            a = [it["sides"][0].get("metrics", {}).get(m) if "error" not in it["sides"][0] else None for it in items]
            b = [it["sides"][1].get("metrics", {}).get(m) if "error" not in it["sides"][1] else None for it in items]
            comparison[m] = stats.paired_comparison(a, b, metric_name=m).__dict__

    return {"mode": mode, "n_items": len(items), "systems": per_system,
            "trigger": trigger, "comparison": comparison}
