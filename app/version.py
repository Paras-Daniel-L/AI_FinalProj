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
    v1.5.2  tax computation without rounding: every figure is exact (no
            rounding to centavos or pesos); repeating per-period quotients
            are marked with "…", never rounded; amounts finer than a
            centavo are asked again. Fixes: semi-monthly pay read as monthly,
            "self-employed" treated as mixed income. RAG answers same as v1.3.

The answer-generation behavior of an older version can be reproduced with
settings in .env (the code paths are kept):
    v1.0 / v1.1   RAG_PROMPT_VERSION=1  DENSE_FLOOR_K=0
    v1.2          RAG_PROMPT_VERSION=2  DENSE_FLOOR_K=0
    v1.3 – v1.5.2 RAG_PROMPT_VERSION=3  DENSE_FLOOR_K=4   (the defaults)
"""

SYSTEM_VERSION = "1.5.2"
SYSTEM_VERSION_LABEL = "v1.5.2"

# The .env settings that reproduce each version's answer generation.
REPRODUCE = {
    "v1.1": {"RAG_PROMPT_VERSION": "1", "DENSE_FLOOR_K": "0"},
    "v1.2": {"RAG_PROMPT_VERSION": "2", "DENSE_FLOOR_K": "0"},
    "v1.3": {"RAG_PROMPT_VERSION": "3", "DENSE_FLOOR_K": "4"},
    "v1.4": {"RAG_PROMPT_VERSION": "3", "DENSE_FLOOR_K": "4"},   # demo-only change: same answers as v1.3
    "v1.5": {"RAG_PROMPT_VERSION": "3", "DENSE_FLOOR_K": "4"},   # demo + sharing change: same answers as v1.3
    "v1.5.1": {"RAG_PROMPT_VERSION": "3", "DENSE_FLOOR_K": "4"},  # docs/cleanup only: same answers as v1.3
    "v1.5.2": {"RAG_PROMPT_VERSION": "3", "DENSE_FLOOR_K": "4"},  # computation-only change: same RAG answers as v1.3
}


def effective_label() -> str:
    """The version label the CURRENT settings actually reproduce, e.g. "v1.5",
    or "v1.5 (custom settings)" when .env overrides them into a mix."""
    from . import prompts, retrieval
    current = {"RAG_PROMPT_VERSION": prompts.RAG_PROMPT_VERSION, "DENSE_FLOOR_K": str(retrieval.DENSE_FLOOR_K)}
    # Newest first: v1.4 to v1.5.2 answer exactly like v1.3, so the current version wins the tie.
    for label, settings in reversed(list(REPRODUCE.items())):
        if settings == current:
            return label
    return f"{SYSTEM_VERSION_LABEL} (custom settings)"
