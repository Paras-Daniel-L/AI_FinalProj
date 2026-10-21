# Sagot AI — retrieval and evaluation improvements

What changed, why, and the exact order to run things. Every change traces back to
something measured in the first T-TED run (`rag_eval/results/`).

## What the first run showed

| Finding | Number |
|---|---|
| Gold issuance retrieved at rank 1 | 95 / 100 questions. Finding the right document was not the problem. |
| Chunks passed to the generator | Always 5, so about 4 of every 5 were off-topic. Context Relevance was 0.19. |
| Context Relevance with only the right chunks | Capped near 0.43, because chunks mix topics and are cut every 800 characters. |
| Chunks ending mid-sentence | 64% |
| Broken tokens from PDF extraction | About 290 (`Value -Added`, `8 -2026`, `shal l`) |
| Refusals | 12 / 100. In 7 of them the answer was in the context. |
| Answers identical across two runs | 8 / 58, although the retrieved context was identical in 57 / 58 |
| Context Relevance judge outputs cut off | 4 units (`JUDGE_MAX_TOKENS=1200`) |

## Changes

### 1. Ingestion — `app/textproc.py` (new), `app/database.py`
- **PDF text cleaning:** joins soft line wraps, fixes broken hyphens (`Value -Added` → `Value-Added`, `8 -2026` → `8-2026`), and repairs split words (`shal l` → `shall`). The word repair uses only the corpus's own word counts, so no dictionary is needed.
- **Sentence-aligned chunks:** whole sentences are packed into chunks of at most 800 characters, with a 1-sentence overlap. Chunks no longer start or end mid-sentence inside a page. Page ends still split, because chunk IDs are per page.
- **Issuance metadata on every chunk:**
  - `issuance_id` (`RMC-8-2026`)
  - `issuance_ids` (the chunk's own ID plus the IDs it cites)
  - `title`
  - `doc_summary` (the SUBJECT line or first sentence)
  - `context_header`
- **The context header is embedded and BM25-indexed with the chunk, but is not stored as evidence.** This lets page 5 of RMC 19-2022 be found by a question naming RMC 19-2022. The generator never sees the header text, so it can't be cited as a fact.
- `CHUNKING_MODE=legacy` restores the old splitter, for an ablation.
- The chunking settings are recorded in the index config. Changing them makes the app ask for `--reset` instead of mixing two kinds of chunks in one index.

### 2. Retrieval — `app/retrieval.py`, `app/rerank.py` (new)
- **Stage 1 (recall):**
  - semantic top 20 + BM25 top 20
  - **issuance-ID matching:** a question naming "RR 14-2022" also searches inside that issuance
  - all fused by RRF into a pool of 30 candidates
- **Stage 2 (precision):** the `jina-reranker-v2-base-multilingual` cross-encoder scores every candidate. It reuses your `JINA_API_KEY` and adds no new dependency. Then the **dynamic cut**:
  - keep candidates with score ≥ `RERANK_MIN_SCORE` and ≥ `RERANK_RELATIVE` × the best score
  - keep 1 to `RERANK_MAX_DOCS` (default 3) chunks
  - near-duplicates (a Digest plus its full issuance, overlapping chunks) are dropped first
  - chunks of an issuance the user named get a small boost (`ID_MATCH_BOOST`)
- **BM25 query handling:** function words are dropped from the query, which matters for Filipino because "ang", "sa" and "ng" have a high IDF in an English corpus. A small Filipino→English tax glossary adds English terms (buwis→tax, kaltas→withholding, …).
- **Optional sentence-level evidence compression** (`EVIDENCE_COMPRESSION=1`, **off by default**):
  - keeps only the sentences of the kept chunks that the reranker scores as relevant
  - also keeps any list lead-in a kept item depends on
  - the generator, the verifier and the evaluation all see the compressed text, so scores stay honest
- **If the reranker fails, retrieval falls back to the old RRF top 5.** An outage lowers quality but never breaks the app.
- `RERANKER=none` restores the original single-stage retrieval exactly.

### 3. Generation — `app/prompts.py`, `app/llm.py`, `app/llm_client.py`
- **NO_ANSWER only when no excerpt addresses the question's subject.** The prompt explains that the English legal wording of an excerpt answers a Filipino question when the meaning matches, and that partial answers are allowed. This targets the 7 refusals that had the answer in context.
- **Answer-first style:**
  - the first sentence restates the question's subject (and the issuance number, if the user named one)
  - Yes/No or Oo/Hindi first for yes/no questions
  - under about 100 words, with no closing advice
- **Conflicting figures:** the answer gives the revised or current figure and says which one it replaced.
- **Retry after a verifier rejection:** the generator is told to *delete* the flagged claims or restate them literally, instead of rewording them into new claims.
- **Determinism:**
  - `GENERATION_TEMPERATURE` defaults to 0.0 (it was 0.1)
  - optional `LLM_SEED`
  - for evaluation runs, also pin providers and set `LLM_ALLOW_FALLBACKS=0`
- **No test-set leakage:** the prompt's examples are generic patterns. A test asserts that no T-TED question appears in the prompt. Topic words lifted from T-TED questions were also kept out of the glossary.
- **Unchanged:** the verifier and the fail-closed design.

### 4. Jina rate limits — `app/jina_http.py` (new)
The free Jina tier allows 100 requests and 100,000 tokens per minute. Rebuilding the index sends about 900 chunks, which is roughly 270,000 tokens. The old code sent them as fast as it could and gave up after waiting 7 seconds, so `--reset` crashed with `HTTP 429`.
- **Throttle:** every Jina call (embeddings, reranker, evaluation embeddings) goes through one limiter that keeps a 60-second window under 90% of both limits. It corrects its own estimate with the token count Jina reports. Waits are printed (`⏳ pacing for the Jina rate limit …`), so a pause doesn't look like a hang.
- **Retry:** if a 429 or 5xx still happens, it waits (honoring `Retry-After`) and tries again, up to 8 times.
- **Embedding batches** are capped at about 20,000 estimated tokens each.
- **Resumable index build:** the index config is written before embedding starts, and chunks are saved batch by batch. If a build stops partway, run `python -m app.database` (without `--reset`) to continue where it stopped.
- **Live chat:** the reranker waits at most `RERANK_MAX_WAIT` (10 s) for rate-limit budget, then falls back to the old top 5, so users never wait a minute. The evaluation scripts wait as long as needed instead, so no evaluated answer comes from the fallback.
- The limits are per process. If you run the chatbot and an evaluation at the same time, they share one quota, so set `JINA_RPM=50` and `JINA_TPM=50000` in `.env` for that.

### 5. Evaluation — `rag_eval/`
- **Context Relevance v3** (`SCORING_VERSION = 3`):
  - sentences are split in Python with the same splitter as ingestion, then numbered
  - the judge returns only the numbers of the relevant sentences
  - source label lines are not counted
  - this gives a fixed denominator and output that can't be cut off
  - **report v2 and v3 separately; they are different measurements**
- `JUDGE_MAX_TOKENS` default is raised to 2500.
- **Effect-size labels are suppressed when fewer than 6 pairs differ.** The first run reported a "large effect" from 2 non-tied pairs.
- **`run_evaluation --results-dir`:** baseline and improved runs never overwrite each other. Each `summary.json` now records the pipeline settings that produced it.
- **`rag_eval/retrieval_eval.py` (new):** a judge-free retrieval report using the dataset's `source` field as the gold label:
  - hit@1, hit@kept, gold_share, mean chunks kept, MRR
  - `--sweep` grid-searches the cut thresholds offline from cached scores

## Run order

```bash
# 0. (optional) run the unit tests — no network needed
python -m pytest tests/test_improvements.py -q

# 1. Rebuild the index (required: new chunking + metadata). OCR is cached.
#    On the free Jina tier this takes a few minutes because of the rate limit.
#    If it stops partway, run it again WITHOUT --reset to resume.
python -m app.database --reset

# 2. Calibrate the reranker cut on a DEV set — 30-50 questions NOT in T-TED,
#    same JSON shape as ted_dataset.json (qid, query, source).
python -m rag_eval.retrieval_eval --dataset rag_eval/dev_set.json --sweep
#    -> copy the suggested RERANK_MIN_SCORE / RERANK_RELATIVE / RERANK_MAX_DOCS into .env

# 3. Re-score the BASELINE answers with the v3 judge (no chatbot calls, judge only)
python -m rag_eval.run_evaluation --subset all

# 4. Evaluate the improved pipeline in its own folder
python -m rag_eval.run_evaluation --subset all --results-dir rag_eval/results_v2

# 5. (optional) A/B the evidence compression
EVIDENCE_COMPRESSION=1 python -m rag_eval.run_evaluation --subset all --results-dir rag_eval/results_v2_compress
```

Recommended `.env` additions for evaluation runs:

```
GENERATOR_PROVIDERS=<one host>
VERIFIER_PROVIDERS=<one host>
LLM_ALLOW_FALLBACKS=0
LLM_SEED=42
```

## New settings (all optional)

| Variable | Default | Meaning |
|---|---|---|
| `CHUNKING_MODE` | `sentence` | `legacy` = the old 800/80 character splitter |
| `CHUNK_SIZE` / `CHUNK_OVERLAP_SENTENCES` | 800 / 1 | Sentence chunker |
| `RERANKER` | `jina` | `none` = the original single-stage retrieval |
| `RERANK_MODEL` | `jina-reranker-v2-base-multilingual` | |
| `RERANK_MIN_SCORE` / `RERANK_RELATIVE` | 0.15 / 0.4 | Dynamic cut. **Calibrate these.** |
| `RERANK_MIN_DOCS` / `RERANK_MAX_DOCS` | 1 / 3 | Chunks passed to the generator |
| `ID_MATCH_BOOST` / `ID_MATCH_TOP_K` | 0.15 / 8 | Named-issuance handling |
| `DEDUP_OVERLAP` | 0.8 | Near-duplicate threshold |
| `SEMANTIC_TOP_K` / `BM25_TOP_K` / `RRF_FINAL_TOP_K` | 20 / 20 / 30 (5/5/5 when the reranker is off) | Candidate pool |
| `FALLBACK_TOP_K` | 5 | Used if the reranker fails |
| `EVIDENCE_COMPRESSION` | 0 | Sentence-level compression |
| `SENTENCE_MIN_SCORE` / `SENTENCE_RELATIVE` / `SENTENCE_MIN_KEEP` | 0.1 / 0.3 / 2 | Compression thresholds |
| `GENERATION_TEMPERATURE` / `VERIFIER_TEMPERATURE` | 0.0 / 0.0 | |
| `LLM_SEED` | unset | Sampling seed |
| `JUDGE_MAX_TOKENS` | 2500 | |
| `JINA_RPM` / `JINA_TPM` | 100 / 100000 | Your Jina plan's per-minute limits |
| `JINA_SAFETY` | 0.9 | Fraction of each limit actually used |
| `JINA_MAX_RETRIES` / `JINA_BACKOFF_MAX` | 8 / 60 | Retries after a 429 or 5xx |
| `JINA_MAX_TOKENS_PER_REQUEST` | 20000 | Estimated-token cap per embedding request |
| `RERANK_MAX_WAIT` | 10 | Seconds a live query waits for rate-limit budget before falling back |

## What was and wasn't verified

- **Verified here:**
  - 29 unit tests
  - an offline end-to-end run: the real ingestion code, a real Chroma index and the real `retrieve_docs` / `run_query` code, using stand-in embeddings, a stand-in reranker and a mocked LLM
  - context per question shrank from 5 chunks to about 3, the share of chunks from the gold issuance rose from 0.65 to 0.84, and gold recall held
  - "RMC No. 8-2026" (B36) now retrieves its issuance through ID matching
- **Not verified:** real-quality numbers. They need your Jina and OpenRouter keys and your PDFs. The default thresholds are starting points, so calibrate them in step 2.
- **Watch Groundedness after the prompt change.** The prompt answers more and refuses less. The verifier remains the safety net, but check that Groundedness stays near 0.99.
- **Still manual:** auditing the ground truth for the correctness disputes (B24 old vs. revised target, B30 judge error, B58).
