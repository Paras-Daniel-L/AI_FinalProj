# Sagot AI: Philippine BIR tax Q&A that refuses to make things up

**Version 1.5.1** (see `VERSIONS.md` for the history).

Sagot AI answers questions about Philippine BIR tax issuances (RRs, RMCs, RMOs, RAOs, rulings, FAQs) using only a fixed set of official PDFs. Every answer is written by one model (DeepSeek) and then checked by a second model from a different company (Qwen) against the exact excerpts the first one was given. If the check fails three times, Sagot AI gives guidance instead of an answer. Tax computations never go through a model: the numbers come from verified rule files and plain code.

One server gives you three pages:

| Page | Address | Who it is for |
|---|---|---|
| Chatbot | `/` | Anyone with the link |
| Live demo | `/demo` | The panel: the chatbot plus its process and three scores, live |
| Evaluation tool | `/eval` | The team: thesis metrics, package tests, thesis results |

This file is the only manual. The code is the final word if the two disagree.

---

## 1. Running it (uvicorn + ngrok)

### First time only

```powershell
cd C:\Users\paras\Desktop\thesisFinal
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r rag_eval/requirements.txt   # the demo and /eval need these
```

Install the Tesseract program (for scanned PDFs) from <https://github.com/UB-Mannheim/tesseract/wiki>. If it isn't on PATH, add `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe` to `.env`.

Create `.env` in the project folder. Never commit it, zip it or send it to anyone, because it holds your API keys.

```
OPENROUTER_API_KEY=sk-or-...
JINA_API_KEY=jina_...
JUDGE_MODEL=openai/gpt-4o-mini        # scores on /demo and /eval; must differ from the generator and verifier
TEAM_ACCESS_CODE=pick-8-or-more-chars # lets teammates on the ngrok link run the demo and /eval
DAILY_QUERY_CAP=100
```

Put the PDFs in `data/<year>/` (or `data/faq/`) and build the index once:

```powershell
python -m app.database --reset
```

### Every time

```powershell
# Terminal 1: the server, reachable only from this computer
venv\Scripts\Activate.ps1
uvicorn main:app --host 127.0.0.1 --port 8000

# Terminal 2: the public link
ngrok http 8000 --url https://your-subdomain.ngrok-free.dev
```

- On this computer, open <http://localhost:8000>. Teammates open your ngrok address.
- Use `--host 127.0.0.1`, not `0.0.0.0`, so the only way in from outside is through ngrok. (`python main.py` binds `0.0.0.0`, so prefer the uvicorn command.)
- Add `--reload` while editing code. Stop with `Ctrl+C` in each terminal.
- The link only works while both terminals are open and the laptop is awake.
- ngrok's free plan shows visitors a "You are about to visit…" page once. They click **Visit Site**.
- Keep `TRUST_CF_CONNECTING_IP` unset (or `0`). Through ngrok, a visitor could fake that header and slip past the per-person limit.

### What teammates can do through the ngrok link

| Action | Without the team code | With the team code |
|---|---|---|
| Use the chatbot (`/`) | Yes, within 6 questions per minute per person and `DAILY_QUERY_CAP` per day | Same |
| Run questions on `/demo` and `/eval` | No (read only) | Yes; demo runs count toward the same limits |
| `/upload`, `/reset` | Never | Never (only from this computer) |

When the code is needed, `/demo` and `/eval` show a small "Team access code" card. The code is remembered in that browser. A green **Team access ✓ · Forget code** chip removes it. After 10 wrong codes in 10 minutes, that address is locked out for the rest of the 10 minutes. To revoke everyone's access, change `TEAM_ACCESS_CODE` and restart uvicorn.

---

## 2. How a question is answered

```
question
  │ 1  Input check          app/sanitize.py     too long / too many questions → refused
  │ 2  Language trigger     app/language.py     english / filipino / taglish → reply language
  │ 3  Greeting / intent    app/greetings.py, app/intent.py
  │                         greeting or "what can you do" → fixed reply (no model)
  │                         "magkano tax ko…"  → tax calculator (app/computation/, no model)
  │                         too vague          → clarifying question
  │ 4  Year routing         app/classifier.py   "RMC 34-2024" → search only 2024
  │ 5  Answer cache         app/cache.py        exact same question → saved verified answer
  │ 6  Hybrid retrieval     app/retrieval.py    Jina semantic search (20) + BM25 keywords (20)
  │                                             + issuance-number match
  │ 7  RRF fusion           app/retrieval.py    one ranked list of 30 candidates
  │ 8  Reranker             app/rerank.py       Jina reranker keeps the best 1–3
  │    Dense-search floor   app/retrieval.py    the top-4 semantic hits are added back (max 6 excerpts)
  │ 9  Generator            app/llm.py          DeepSeek writes a cited answer [1][2], or NO_ANSWER
  │10  Verifier             app/llm.py          Qwen checks every claim against the same excerpts
  │                                             pass → answer (cached)
  │                                             fail → rewrite with the verifier's notes, up to 3 tries
  ▼                                             still failing → guidance (app/recovery.py), never cached
answer
```

Why two models from different companies: a model checking its own work tends to approve its own mistakes. The verifier checks that each claim is **in the excerpts**. It cannot tell if the PDF itself is wrong (for example an OCR misread). Sources marked `· OCR` are the ones to double-check.

---

## 3. What each file does

### Start-up

| File | Role |
|---|---|
| `main.py` | What `uvicorn main:app` loads. Takes the chatbot app from `app/api.py` and adds the `/eval` and `/demo` pages. If the evaluation extras aren't installed, the chatbot still starts without them. |
| `requirements.txt` | Packages the chatbot needs. |
| `rag_eval/requirements.txt` | Extra packages for `/eval` and `/demo` (scipy, numpy). |
| `VERSION`, `app/version.py` | The version number shown on the demo page and saved in every evaluation result. |
| `.env` | Your keys and settings. Never shared. |

### `app/`: the chatbot

| File | Role |
|---|---|
| `api.py` | The web server's routes: `/` (chat page), `/query` (answer a question), `/status`, `/upload`, `/reset`. `run_query()` runs steps 1–10 above. |
| `guard.py` | Protection for the public link: per-person rate limit, daily cap, host-only `/upload`/`/reset`, the team access code, the bad-code lockout. |
| `sanitize.py` | Step 1: cleans the question and refuses abusive input. |
| `language.py` | Step 2: detects English, Filipino or Taglish. |
| `greetings.py` | Step 3: spots a bare "hi" / "kumusta". |
| `intent.py` | Step 3: chat, tax information, tax computation or out of scope. |
| `classifier.py` | Step 4: year filter from a citation or a single year. |
| `cache.py` | Step 5: the SQLite answer cache (`cache/answers.sqlite3`). |
| `retrieval.py` | Steps 6–8: semantic + BM25 + issuance match, RRF fusion, the reranker cut and the dense-search floor. |
| `rerank.py` | The Jina reranker call used in step 8. |
| `embeddings.py` | Jina embeddings (query vs. passage modes). |
| `jina_http.py` | One rate limiter and retry policy for every Jina call. |
| `llm.py` | Steps 9–10: the generate → verify → retry loop. |
| `llm_client.py` | Low-level OpenRouter calls: retries, cost, thinking off. |
| `prompts.py` | Every prompt and fixed message. `RAG_PROMPT_VERSION` 1/2/3 picks the generation prompt (3 is the default). |
| `recovery.py` | The guidance shown instead of a dead-end refusal. No model call. |
| `schemas.py` | Shapes of requests and responses. |
| `trace.py` | Writes one line per question to `logs/queries.jsonl` (drafts, verdicts, cost). |
| `database.py` | Builds the index from `data/` into `chroma/` (`python -m app.database`). |
| `textproc.py` | PDF text cleanup and sentence-aligned chunking used by ingestion. |
| `ocr.py` | Tesseract OCR for scanned pages, cached in `ocr_cache/`. |
| `progress.py` | **Live demo only.** Lets the pipeline report each step as it happens (section 4). |

### `app/computation/`: the tax calculator (no model)

| File | Role |
|---|---|
| `rules.py` | Loads and validates `tax_rules/*.json`. Rejects broken tables. `python -m app.computation.rules` prints them. |
| `calculator.py` | Decimal arithmetic and the step-by-step breakdown. |
| `extract.py` | Reads amounts, years and pay periods from English, Filipino and Taglish. |
| `dialogue.py` | Asks for missing inputs over several turns, and validates them. |
| `messages.py` | Every sentence the calculator says, in three languages. |

Supported: graduated income tax (2018–2022 and 2023–2026 tables), the 8% option for the purely self-employed, and employees from gross salary (SSS, PhilHealth and Pag-IBIG computed for 2025–2026). Anything else (VAT, estate, corporate…) is declined instead of guessed. The year is never assumed.

### Data and configuration

| Path | Role |
|---|---|
| `tax_rules/*.json` | Tax rates and contribution rules, with legal basis and a verification record. |
| `config/document_metadata.json` | Labels (document type, tax type) attached to chunks at ingestion. |
| `data/` | The PDFs (not in git). |
| `chroma/` | The search index, rebuilt from `data/` (not in git). |
| `cache/`, `logs/`, `ocr_cache/` | Created while running (not in git, not for sharing). `logs/queries.jsonl` holds visitors' questions. |

### `static/`: the pages

| File | Role |
|---|---|
| `index.html`, `js/app.js`, `css/style.css` | The chatbot page. Has the **Live demo** button. The **Evaluation** button is hidden (see section 4). |
| `demo.html`, `js/demo.js`, `css/demo.css` | The live demo page (section 4). |
| `eval.html`, `js/eval.js`, `css/eval.css` | The evaluation tool (section 5). |
| `js/team-access.js` | Asks for the team code on `/demo` and `/eval` and sends it with every request. |

### `rag_eval/`: evaluation and the demo's back end

| File | Role |
|---|---|
| `demo_web.py` | **Live demo back end** (section 4). |
| `demo_metrics.py` | **The demo's three scores** (section 4). |
| `web.py` | Back end of `/eval`: single tests, package tests, saved runs, the Thesis results tab. |
| `service.py` | Scores one question for `/eval`. |
| `run_evaluation.py` | The offline thesis run over T-TED, with checkpoints (section 5). |
| `judge.py` | The judge model's prompts (claims, facts, relevance). |
| `ragas_metrics.py` | Groundedness, Context Relevance, Answer Relevance. |
| `embeddings_eval.py` | Embeddings for Answer Relevance's similarity step. |
| `stats.py` | The statistics from the SOP (paired tests, confusion matrix). |
| `dataset.py`, `ted_dataset.json` | The T-TED questions and answers. |
| `json_util.py` | Reads JSON out of a model's reply. |
| `compare_runs.py` | Command-line: compares two result folders (before/after). |
| `retrieval_eval.py` | Command-line: retrieval-only hit rate and reranker threshold sweep. |
| `results/` | Default results folder (also holds `/eval`'s saved package runs in `web_runs/`). |
| `results_v2/` | The clean baseline run used in the thesis. |

### `tests/` and `tools/`

| File | Role |
|---|---|
| `tests/test_*.py` | 154 offline tests (no keys, no network): `python -m pytest -q` (install `pytest` first). |
| `tools/test_computation_live.py` | 38 computation questions against the running server, checked by an independent calculator. |
| `tools/build_reference_note.py` | Builds the reference PDF for the knowledge base (needs `pip install reportlab`). |
| `check_scans.py` | Lists PDFs with scanned pages. |
| `check_ingestion_coverage.py` | Compares the PDFs on disk with what's in the index. |

---

## 4. The live demo, explained

### What the panel sees

1. A plain chat. Type any tax question (there are no built-in questions) and press send.
2. While Sagot AI works, one plain line says what it is doing: "Searching the BIR documents…", "Double-checking every claim…".
3. The answer appears. Long answers fold behind **Show full answer**.
4. Under the answer are three buttons. Nothing else shows until clicked:
   - **Sources (n)**: the documents used, with links.
   - **Scores**: three scores from 0 to 1, each with **How it is computed**.
   - **How it was made**: the architecture strip (Input check → Language trigger → Intent & routing → Hybrid retrieval → RRF fusion → Reranker → Generator → Verifier → Answer) and a step-by-step log: what each search found, what the reranker kept, every draft, every verifier verdict and retry. Stages a turn didn't use are struck through.
5. **Options** (next to the question box) has:
   - **Compare with REVIE**: ask REVIE the same question on the BIR website, paste its answer here, then send. Both answers appear side by side, and **Compare the scores side by side** shows one bar chart per score.
   - **Reference answer**: type the correct answer if you want the third score.
6. **Guide** in the top bar explains the page for a first-time viewer.

### The three scores

They are computed by a separate judge model (`JUDGE_MODEL`), never by the chatbot's own models.

| Name on the page | How it is computed | When it is N/A |
|---|---|---|
| Groundedness | Claims in the answer supported by the retrieved excerpts ÷ all claims | No excerpts to check against (REVIE, a computation, no documents found) or no factual claims (a polite decline) |
| Context Relevance | The judge writes 3 questions from the answer. Score = average similarity between those and the real question (the evaluation's Answer Relevance formula) | Only if the judge or Jina call fails |
| Answer Relevance | Facts in your reference answer that the answer states correctly ÷ all reference facts (the evaluation's Answer Correctness formula) | No reference answer typed |

"N/A" means the score does not apply. It is never a zero. Greetings, clarifying questions and computation follow-ups are not scored. Be ready to say in the defense that the demo's last two names use the evaluation's Answer Relevance and Answer Correctness formulas. The page shows the real formula.

### What happens when you press send

```
browser (static/js/demo.js)
  │ POST /eval/api/run/demo  {question, compare, reference, REVIE's answer}
  ▼
app/guard.py              host computer or valid team code? within the limits?
  ▼
rag_eval/demo_web.py      one run per person, at most DEMO_MAX_CONCURRENT at once
  │                       starts one worker thread per system (Sagot AI, and REVIE if comparing)
  │
  ├─ Sagot AI worker ── app/api.run_query()  (the SAME pipeline as the chatbot, cache skipped)
  │      every step calls app/progress.emit(...) ──► sent to the browser as a "step" event
  │      then rag_eval/demo_metrics.score_demo() ──► one "metric" event per score
  │
  └─ REVIE worker ───── uses the pasted answer, then scores it the same way
  ▼
the browser receives one JSON line per event, as it happens (NDJSON stream)
demo.js lights up the strip, writes the log, fills in the answer and the scores
```

`app/progress.py` only reports when the demo is listening. For the normal chatbot and `/eval` it does nothing, so they behave exactly as before.

### The events the page receives

| Event | Meaning |
|---|---|
| `start` | Which systems are running, and whether scoring is available |
| `step` | One pipeline step started or finished (e.g. `semantic`, `rerank`, `dense_floor`, `generate`, `verify`), with its details |
| `answer` | The final answer, sources and outcome |
| `metric` | One score: `running`, then `done` with the result |
| `metrics_skipped` | Why this turn isn't scored |
| `error` | Something failed (shown on that system's card) |
| `system_done` | One system finished, with its time |
| `ping` | Keeps the connection alive during long steps |
| `done` | The whole run is finished |

### Demo files, one by one

| File | What it does |
|---|---|
| `rag_eval/demo_web.py` | `GET /demo` serves the page. `GET /demo/api/config` tells the page the version, the scores' definitions and whether this visitor may run. `GET /team/api/status` says whether a team code is needed and valid. `POST /eval/api/run/demo` runs the question and streams the events. The run endpoint is under `/eval/api/run/` on purpose, so the same access rule as `/eval` covers it. |
| `rag_eval/demo_metrics.py` | The three scores above. It reuses the evaluation's judge (`judge.py`) and similarity code (`ragas_metrics.py`) unchanged. |
| `app/progress.py` | `emit()` reports a step. `listening()` is how `demo_web.py` subscribes for one run. |
| `static/demo.html` | The page layout: top bar, chat area, Options panel, question box. |
| `static/js/demo.js` | Sends the question, reads the stream, and draws everything. Useful places: `MODES` (compare options), `PIPES` (the architecture strip), `STEP_STAGE` (which step lights which stage), `liveText()` (the plain status line), `describe()` (the step log text), `CLIP_PX` (fold height). |
| `static/css/demo.css` | The look, including the phone layout. |
| `static/js/team-access.js` | The team code card and chip. Loaded before `demo.js`. |
| `tests/test_demo.py` | Offline tests for the events, scores, compare modes, team code and limits. |

### Showing or hiding the Evaluation button

It is hidden on `/` and `/demo` so the panel isn't drawn to it. `/eval` still works. Open `http://localhost:8000/?eval=1` once to show it in that browser, and `?eval=0` to hide it again.

### Cost

One demo question costs Sagot AI's 2–6 model calls plus 3 judge calls and 1 Jina call per system. The cache is always skipped so every step runs live.

---

## 5. The evaluation tool and the thesis run

`/eval` has four tabs:

| Tab | What it does |
|---|---|
| How it works | Explains the four inputs (question, context, answer, ground truth) and every score in plain language. |
| Test one question | Sagot AI, another chatbot (paste its answer), or both. |
| Test a package | Upload CSV/JSON or load T-TED. Runs in the background and is saved to `rag_eval/results/web_runs/`. |
| Thesis results | RQ1–RQ3 from the offline run's checkpoint. |

The offline run (from the project folder, with the venv active):

```powershell
python -m rag_eval.run_evaluation --subset A --limit 4                            # quick check
python -m rag_eval.run_evaluation --subset all --results-dir rag_eval/results_v3  # full run in its own folder
python -m rag_eval.run_evaluation --rq2-only                                      # language trigger only, free
python -m rag_eval.compare_runs --before rag_eval/results_v2 --after rag_eval/results_v3   # before vs. after
```

Re-running the same command resumes; finished questions are never paid for twice. Every `summary.json` records the system version and settings that produced it.

**Which results the Thesis tab shows.** It reads `rag_eval/results/` by default, which has a mixed run. To show the clean baseline, add this to `.env` and restart uvicorn:

```
EVAL_THESIS_DIR=rag_eval/results_v2
```

---

## 6. Settings (`.env`)

Only the two API keys are required. The rest have working defaults.

| Setting | Default | What it does |
|---|---|---|
| `OPENROUTER_API_KEY`, `JINA_API_KEY` | (required) | Model and embedding access |
| `JUDGE_MODEL` | (none) | Judge for `/demo` and `/eval` scores. Without it, answers still show but aren't scored |
| `TEAM_ACCESS_CODE` | (none) | Team code for the ngrok link (8+ characters). Unset = host computer only |
| `RATE_LIMIT_PER_MIN` | `6` | Questions per person per minute |
| `DAILY_QUERY_CAP` | `100` | Questions per day, all visitors (about $1.10/day worst case) |
| `DEMO_MAX_CONCURRENT` | `3` | Demo questions running at the same time |
| `ADMIN_LOCAL_ONLY`, `EVAL_LOCAL_ONLY` | `1`, `1` | Keep `/upload`/`/reset` and the run buttons host-only (unless the team code is given) |
| `TRUST_CF_CONNECTING_IP` | `0` | Keep at 0 with ngrok |
| `EVAL_THESIS_DIR` | `rag_eval/results` | Folder the Thesis results tab reads |
| `GENERATOR_MODEL` | `deepseek/deepseek-v4.1-flash` | Writes answers |
| `VERIFIER_MODEL` | `qwen/qwen3.8-27b` | Checks answers (must be a different company) |
| `MAX_RAG_RETRIES` | `3` | Write → check attempts before guidance |
| `RAG_PROMPT_VERSION` | `3` | Generation prompt (1 = the one used for `results_v2`) |
| `DENSE_FLOOR_K`, `EVIDENCE_MAX_DOCS` | `4`, `6` | Dense-search floor (0 = off) and the excerpt limit |
| `RERANK_MIN_SCORE`, `RERANK_RELATIVE`, `RERANK_MAX_DOCS` | `0.15`, `0.4`, `3` | Reranker cut |
| `SEMANTIC_TOP_K`, `BM25_TOP_K`, `RRF_FINAL_TOP_K` | `20`, `20`, `30` | Candidate pool sizes |
| `GENERATOR_PROVIDERS`, `VERIFIER_PROVIDERS`, `LLM_ALLOW_FALLBACKS`, `LLM_SEED` | (none), (none), `1`, (none) | Pin hosts and seed for repeatable evaluation runs |
| `CACHE_ENABLED`, `CACHE_TTL_DAYS` | `1`, `7` | Answer cache |
| `USE_HISTORY` | `0` | 1 = use earlier turns as context |
| `RECOVERY_ENABLED` | `1` | 0 = old one-line refusal (ablation) |
| `JINA_RPM`, `JINA_TPM` | `100`, `100000` | Your Jina plan's per-minute limits |
| `TESSERACT_CMD` | (auto) | Path to tesseract.exe |

Every other setting has a comment next to its `os.environ.get(...)` line in the code.

---

## 7. Everyday commands

```powershell
# Corpus
python -m app.database               # add new PDFs only
python -m app.database --reset       # full rebuild (also clears the cache)
python check_scans.py                # which PDFs need OCR
python check_ingestion_coverage.py   # PDFs on disk vs. indexed

# Cache and logs
python -m app.cache --stats
python -m app.cache --clear          # after editing classifier.py or source labels
python -m app.trace                  # outcome counts, attempts, cost
python -m app.trace --last 5

# Checks
python -m pytest -q                              # offline tests
python -m app.computation.rules                  # print the loaded tax rules
python tools/test_computation_live.py            # server must be running
```

To add one PDF while the server runs (from this computer only):

```powershell
curl.exe -X POST http://localhost:8000/upload -F "file=@C:\path\RMC No. 12-2026.pdf" -F "category=2026"
```

---

## 8. Design decisions to keep in mind

- **Fail-closed.** Anything but a clean "SUPPORTED" from the verifier counts as a failure.
- **No rewriter or translation step.** The answer is written directly in the user's language, so nothing unchecked happens after verification.
- **History is off.** Each question stands alone. This keeps the cache useful and stops the generator from reusing facts the verifier never checked.
- **The year filter needs exactly one year.** Two years or a range searches everything, so amendments aren't missed.
- **The cache is exact-match.** "RMC 24-2026" and "RMC 25-2026" never share an answer. Only verified answers are cached.
- **Rates live in `tax_rules/`, not in a model.** A year outside a rule's verified range is refused. To support 2027, check the law first, then extend `tax_year_to`.
