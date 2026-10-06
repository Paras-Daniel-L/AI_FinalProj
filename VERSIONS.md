# Sagot AI versions

The running version is in `app/version.py`. It shows on the live demo page ("Live demo · v1.5.1") and is saved in every evaluation `summary.json` (`system_version`, `reproduces`). Each zip is named after its version, for example `sagot-ai-v1.5.1.zip`.

| Version | What changed | Answers produced with |
|---|---|---|
| v1.0 | Hybrid RAG with a cross-vendor verifier. The system measured in the thesis (`rag_eval/results_v2`). | prompt 1, no dense floor |
| v1.1 | Conversational recovery (no dead ends) and the deterministic tax calculator. | prompt 1, no dense floor |
| v1.2 | Live demo page (`/demo`). Concise generation prompt. | prompt 2, no dense floor |
| v1.3 | Dense-search floor: the top 4 semantic hits are always added to the reranker's picks (4–6 excerpts). Prompt 3 combines every relevant excerpt. | prompt 3, dense floor 4 |
| v1.4 | Demo compares with LLM only and REVIE (standard RAG removed). | same as v1.3 |
| v1.5 | Demo compares with REVIE only. Team access code so teammates can test through the ngrok link. | same as v1.3 |
| v1.5.1 | Cleanup, no behavior change: one README with a file-by-file guide; Docker, Cloudflare and share scripts removed; unused `app/baselines.py` removed; `EVAL_THESIS_DIR` chooses the folder the Thesis results tab reads. | same as v1.3 |

## Reproducing an older version's answers

| To reproduce | Put in `.env` |
|---|---|
| v1.0 / v1.1 (what `rag_eval/results_v2` measured) | `RAG_PROMPT_VERSION=1` and `DENSE_FLOOR_K=0` |
| v1.2 | `RAG_PROMPT_VERSION=2` and `DENSE_FLOOR_K=0` |
| v1.3 – v1.5.1 | nothing (the defaults) |

If the settings match none of these, the demo and `summary.json` say "v1.5.1 (custom settings)". The LLM-only and standard-RAG demo baselines are in git tag `v1.3` if the ablation is ever needed again.

## Commit text for v1.5.1

**Summary**

```
v1.5.1: one README, file-by-file guide, remove Docker/Cloudflare files and dead code
```

**Description**

```
Documentation and cleanup only. Answers are produced exactly as in v1.3-v1.5.

- README.md rewritten as the single manual: running with uvicorn + ngrok,
  what teammates can do with and without the team code, the pipeline,
  the role of every code file, and a plain-language guide to the live demo
  (what the panel sees, how a run flows through the files, the stream
  events, the three scores).
- VERSIONS.md shortened. CHANGES.md, CONVERSATION_AND_COMPUTATION.md and
  SHARING.md removed; what is still current is in README.md.
- Removed Dockerfile, .dockerignore, share.bat, share.ps1 (the project is
  run with uvicorn and shared with ngrok) and the unused app/baselines.py.
- New optional EVAL_THESIS_DIR setting: the Thesis results tab can show
  rag_eval/results_v2 instead of rag_eval/results.
- Comments in app/guard.py, app/version.py and rag_eval/demo_web.py no
  longer mention Docker, Cloudflare or SHARING.md. Version 1.5.1.
- Tests: 154 pass (new test for EVAL_THESIS_DIR).
```
