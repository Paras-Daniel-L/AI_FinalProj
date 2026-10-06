# Sagot AI versions

The running version is in `app/version.py`, shown on the live demo page ("Live demo · v1.3") and written into every evaluation run's `summary.json` (`system_version`, `reproduces`). Each delivered zip is named after its version, for example `sagot-ai-v1.3.zip`.

| Version | What it adds | Answer generation |
|---|---|---|
| **v1.0** | Hybrid RAG with a cross-vendor verifier; the system evaluated in the thesis (commit `fd35002`, "v1 results") | prompt v1, no dense floor |
| **v1.1** | Conversational recovery (no dead ends) and deterministic tax computation (commit `2dc82f3`) | prompt v1, no dense floor |
| **v1.2** | Live demo page (`/demo`) with the C0 / C1 / C2 comparison; concise generation prompt (v2) | prompt v2, no dense floor |
| **v1.3** | Dense-search floor: the generator always sees the top 4 dense hits as well as the reranker's picks (4–6 excerpts); prompt v3 combines every relevant excerpt (up to about 80 words); version label | prompt v3, dense floor k=4 |

## Reproducing an older version's answers

The older code paths are kept and selected in `.env`:

| To reproduce | Set |
|---|---|
| v1.0 / v1.1 (what `rag_eval/results_v2` measured) | `RAG_PROMPT_VERSION=1` and `DENSE_FLOOR_K=0` |
| v1.2 | `RAG_PROMPT_VERSION=2` and `DENSE_FLOOR_K=0` |
| v1.3 (default) | nothing (or `RAG_PROMPT_VERSION=3`, `DENSE_FLOOR_K=4`) |

When the settings match none of these, the demo and `summary.json` say "v1.3 (custom settings)".

## v1.3 in detail

**Why.** In the demo, C1 (standard dense-only RAG) gave answers with more references, and its Context Relevance score was higher than Sagot AI's. Sagot AI's reranker keeps only 1–3 excerpts, cutting anything below its score thresholds, so a relevant chunk that dense search ranked highly could be dropped before generation.

**What changed:**

- **Dense-search floor** (`app/retrieval.add_dense_floor`). After the reranker's cut, the top `DENSE_FLOOR_K` (4) chunks by embedding similarity are added back if the cut dropped them, up to `EVIDENCE_MAX_DOCS` (6) excerpts in total.
  - The reranker's picks come first, so they keep citation numbers [1], [2], …
  - Duplicates (same chunk, or near-identical text) are skipped.
  - No extra search is made: these chunks are already among the candidates the reranker scored.
  - Sagot AI's evidence therefore always includes what C1 sees (same embedding model, same k), plus the BM25, issuance-number and reranker results.
- **Prompt v3** (`app/prompts.py`): v2 plus "read all excerpts and combine what they say, citing each one", with room for up to about 80 words.

**Unchanged:** the language trigger, intent routing, hybrid search, RRF, the reranker, the cross-vendor verifier (it now checks against the larger evidence set), the retry loop and the fail-closed refusal.

**Effect on the evaluation:**

- **Before quoting v1.3 numbers, re-run T-TED.** `rag_eval/results_v2` measured v1.0/v1.1.
- **The two Context Relevance scores move differently.** The demo's Context Relevance is computed from the answer (the Answer Relevance formula), so a more complete answer raises it. The evaluation tool's Context Relevance counts relevant sentences ÷ all sentences in the retrieved excerpts, so passing more excerpts can lower that ratio even when the answer improves. Report both sides of this trade-off.
- **Don't tune on T-TED.** Choose `DENSE_FLOOR_K` and `EVIDENCE_MAX_DOCS` on your own new questions, not on T-TED, so the evaluation stays independent.
