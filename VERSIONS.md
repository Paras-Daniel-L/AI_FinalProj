# Sagot AI versions

The running version is in `app/version.py`, shown on the live demo page ("Live demo · v1.5") and written into every evaluation run's `summary.json` (`system_version`, `reproduces`). Each delivered zip is named after its version, for example `sagot-ai-v1.3.zip`.

| Version | What it adds | Answer generation |
|---|---|---|
| **v1.0** | Hybrid RAG with a cross-vendor verifier; the system evaluated in the thesis (commit `fd35002`, "v1 results") | prompt v1, no dense floor |
| **v1.1** | Conversational recovery (no dead ends) and deterministic tax computation (commit `2dc82f3`) | prompt v1, no dense floor |
| **v1.2** | Live demo page (`/demo`) with the C0 / C1 / C2 comparison; concise generation prompt (v2) | prompt v2, no dense floor |
| **v1.3** | Dense-search floor: the generator always sees the top 4 dense hits as well as the reranker's picks (4–6 excerpts); prompt v3 combines every relevant excerpt (up to about 80 words); version label | prompt v3, dense floor k=4 |
| **v1.4** | Demo focuses on Sagot AI vs **LLM only (C0)** and vs **REVIE (C2)**; standard RAG (C1) removed from the demo (no meaningful difference after v1.3). Comparison choices: Sagot AI only, vs LLM only, vs REVIE, vs LLM & REVIE | same as v1.3 |
| **v1.5** | Demo compares **Sagot AI with REVIE only** (C0 removed too; baseline code removed, kept in tag `v1.3`). **Team access for a public link:** `TEAM_ACCESS_CODE` lets teammates run the demo and evaluation tool, with demo runs rate-limited and one run per person at a time; `share.bat` / `share.ps1` (Cloudflare quick tunnel), `SHARING.md`, `Dockerfile` for always-on hosting | same as v1.3 |

## Reproducing an older version's answers

The older code paths are kept and selected in `.env`:

| To reproduce | Set |
|---|---|
| v1.0 / v1.1 (what `rag_eval/results_v2` measured) | `RAG_PROMPT_VERSION=1` and `DENSE_FLOOR_K=0` |
| v1.2 | `RAG_PROMPT_VERSION=2` and `DENSE_FLOOR_K=0` |
| v1.3 – v1.5 (default) | nothing (or `RAG_PROMPT_VERSION=3`, `DENSE_FLOOR_K=4`) |

When the settings match none of these, the demo and `summary.json` say "v1.5 (custom settings)".

## v1.5 in detail

**Why:**

- The panel comparison is about the project versus the existing BIR chatbot.
- Standard RAG (C1) showed no meaningful difference after v1.3, and the LLM-only baseline (C0) isn't needed for that story.
- Teammates need to test the system from their own devices.

**What changed:**

- **Demo:** "Compare with REVIE" is the only comparison. The API's compare modes are now `none` and `revie`; `c0`, `c1`, `c2` and `all` are rejected. `app/baselines.py` was removed.
- **Team access** (`app/guard.py`, `static/js/team-access.js`):
  - With `TEAM_ACCESS_CODE` set, a page opened through the public link asks for the code once and sends it as `X-Team-Code`. The code is compared in constant time.
  - 10 wrong codes from one address within 10 minutes lock that address out for the rest of the 10 minutes.
  - Demo runs from the public link share `/query`'s per-minute and daily limits.
  - Each visitor runs one demo question at a time, at most `DEMO_MAX_CONCURRENT` (3) overall. This replaces v1.2–v1.4's single global lock, so testers don't block each other.
  - `/upload` and `/reset` are never opened by the code.
- **Sharing:** `share.bat` / `share.ps1` start the server and a free Cloudflare quick tunnel. `SHARING.md` covers both options (laptop link, or always-on hosting with the `Dockerfile`).

**Unchanged:** how answers are produced (same as v1.3), and the evaluation tool's code (its page only loads the team-access script).

**Upgrading from a v1.2–v1.4 folder:** extract `sagot-ai-v1.5.zip` over the project, then **delete `app/baselines.py`** by hand (a zip can add and replace files but not remove them).

## Commit text for v1.5

Use this when committing v1.5 in GitHub Desktop.

**Summary**

```
v1.5: Sagot AI vs REVIE demo, team access code and public sharing
```

**Description**

```
Demo
- Compare Sagot AI with REVIE only ("Compare with REVIE"); remove the
  LLM-only (C0) baseline and the leftover baseline code (app/baselines.py).
  Standard RAG (C1) was already removed in v1.4. Both remain in tag v1.3.
- Compare modes are now "none" and "revie".

Public sharing for teammates
- TEAM_ACCESS_CODE: teammates on the public link can run the demo and the
  evaluation tool after entering the code once; without it those pages are
  read-only. Wrong codes are rate-limited; /upload and /reset stay host-only.
- Demo runs from the public link share the chatbot's per-minute and daily
  limits; one run per person at a time, DEMO_MAX_CONCURRENT (3) overall.
- share.bat / share.ps1: start the server and a free Cloudflare tunnel.
- SHARING.md: how to share (laptop link or always-on hosting), costs, fixes.
- Dockerfile + .dockerignore for always-on hosting (e.g. Hugging Face Spaces).

Versioning
- app/version.py and VERSION: 1.5.0 (answers unchanged since v1.3).
- VERSIONS.md: v1.5 notes; README sections 13-14 updated.

Tests: 153 passing (team code, lockout, rate limits, Sagot AI vs REVIE).
```

If the repository is still at v1.1 (commit `2dc82f3`) and you commit v1.2 through v1.5 together, use the same summary and add this line at the top of the description:

```
Includes v1.2-v1.4: live demo page, concise prompt (v2/v3), dense-search floor, version label.
```

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
