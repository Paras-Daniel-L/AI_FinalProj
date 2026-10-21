"""
Sagot AI system version.

One place that says which version of the framework is running. It is shown
on the live demo page, sent as the FastAPI app version, and written into every
evaluation run's summary.json (rag_eval/run_evaluation.py), so a results
folder always says which system produced it.

Version history (see VERSIONS.md for details):
    v1.0  RAG baseline evaluated in the thesis ("v1 results", commit fd35002)
    v1.1  + conversational recovery and deterministic tax computation (2dc82f3)
    v1.2  + live demo page (/demo) with C0/C1/C2 comparison; concise prompt v2
    v1.3  + dense-search floor (the generator always sees the top-4 dense hits
            as well as the reranker's picks); prompt v3 (combine all relevant
            excerpts, up to ~80 words); version label
    v1.4  demo compares Sagot AI with the LLM alone (C0) and REVIE (C2);
            standard RAG (C1) removed from the demo. Answers same as v1.3.
    v1.5  demo compares Sagot AI with REVIE only (C0 removed too); team
            access code so teammates can test through the ngrok link.
            Answers same as v1.3.
    v1.5.1  cleanup: one README (file-by-file guide), no Docker/Cloudflare
            files, unused app/baselines.py removed, EVAL_THESIS_DIR setting
            for the Thesis results tab. Answers same as v1.3.
    v1.5.2  tax computation without rounding. RAG answers same as v1.3.
    v1.6  hallucination mitigation, groundedness and relevance:
            evidence gate, non-answer detector, conditional dense floor,
            sentence-level evidence selection on by default, deterministic
            grounding check + claim repair, verifier v2 (also checks the
            draft answers the question), prompt v4 (answer shape follows the
            question; Oo/Hindi only for yes/no), extra Filipino BM25 terms,
            and routing fixes from the 2026-10-07 log review.

The answer-generation behavior of an older version can be reproduced with
settings in .env (the code paths are kept); see REPRODUCE below and
VERSIONS.md. Every v1.6 change has its own switch, so each one can be turned
off alone for the ablation table (tools/dev_tune.py --ablation).
"""

SYSTEM_VERSION = "1.6.0"
SYSTEM_VERSION_LABEL = "v1.6"

# What each v1.6 switch is set to when it is OFF (= the v1.5.2 behavior).
V16_OFF = {
    "DENSE_FLOOR_RELATIVE": "0",
    "EVIDENCE_COMPRESSION": "0",
    "EVIDENCE_GATE_MIN_SCORE": "0",
    "NON_ANSWER_DETECTION": "0",
    "GROUNDING_CHECK": "0",
    "CLAIM_REPAIR": "0",
    "VERIFIER_RELEVANCE": "0",
    "QUERY_GLOSSARY_V16": "0",
}
V16_ON = {
    "DENSE_FLOOR_RELATIVE": "0.5",
    "EVIDENCE_COMPRESSION": "1",
    "EVIDENCE_GATE_MIN_SCORE": "0.14",
    "NON_ANSWER_DETECTION": "1",
    "GROUNDING_CHECK": "1",
    "CLAIM_REPAIR": "1",
    "VERIFIER_RELEVANCE": "1",
    "QUERY_GLOSSARY_V16": "1",
}

# The .env settings that reproduce each version's answer generation.
REPRODUCE = {
    "v1.1": {"RAG_PROMPT_VERSION": "1", "DENSE_FLOOR_K": "0", **V16_OFF},
    "v1.2": {"RAG_PROMPT_VERSION": "2", "DENSE_FLOOR_K": "0", **V16_OFF},
    "v1.3": {"RAG_PROMPT_VERSION": "3", "DENSE_FLOOR_K": "4", **V16_OFF},
    "v1.4": {"RAG_PROMPT_VERSION": "3", "DENSE_FLOOR_K": "4", **V16_OFF},   # demo-only change: same answers as v1.3
    "v1.5": {"RAG_PROMPT_VERSION": "3", "DENSE_FLOOR_K": "4", **V16_OFF},   # demo + sharing change: same answers as v1.3
    "v1.5.1": {"RAG_PROMPT_VERSION": "3", "DENSE_FLOOR_K": "4", **V16_OFF},  # docs/cleanup only: same answers as v1.3
    "v1.5.2": {"RAG_PROMPT_VERSION": "3", "DENSE_FLOOR_K": "4", **V16_OFF},  # computation-only change: same RAG answers as v1.3
    "v1.6": {"RAG_PROMPT_VERSION": "4", "DENSE_FLOOR_K": "4", **V16_ON},
}


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def current_settings() -> dict:
    """The answer-shaping settings in effect now, as .env strings."""
    from . import answerability, grounding, prompts, retrieval
    return {
        "RAG_PROMPT_VERSION": prompts.RAG_PROMPT_VERSION,
        "DENSE_FLOOR_K": _fmt(retrieval.DENSE_FLOOR_K),
        "DENSE_FLOOR_RELATIVE": _fmt(retrieval.DENSE_FLOOR_RELATIVE),
        "EVIDENCE_COMPRESSION": _fmt(retrieval.EVIDENCE_COMPRESSION),
        "EVIDENCE_GATE_MIN_SCORE": _fmt(answerability.EVIDENCE_GATE_MIN_SCORE),
        "NON_ANSWER_DETECTION": _fmt(answerability.NON_ANSWER_DETECTION),
        "GROUNDING_CHECK": _fmt(grounding.GROUNDING_CHECK),
        "CLAIM_REPAIR": _fmt(grounding.CLAIM_REPAIR),
        "VERIFIER_RELEVANCE": _fmt(prompts.VERIFIER_RELEVANCE),
        "QUERY_GLOSSARY_V16": _fmt(retrieval.QUERY_GLOSSARY_V16),
    }


def effective_label() -> str:
    """The version label the CURRENT settings actually reproduce, e.g. "v1.6",
    or "v1.6 (custom settings)" when .env overrides them into a mix."""
    current = current_settings()
    # Newest first: v1.4 to v1.5.2 answer exactly like v1.3, so the newest wins the tie.
    for label, settings in reversed(list(REPRODUCE.items())):
        if all(_same(current.get(k), v) for k, v in settings.items()):
            return label
    return f"{SYSTEM_VERSION_LABEL} (custom settings)"


def _same(a, b) -> bool:
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)
