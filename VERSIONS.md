# Sagot AI versions

The running version is in `app/version.py`. It shows on the live demo page ("Live demo · v1.5.2") and is saved in every evaluation `summary.json` (`system_version`, `reproduces`). Each zip is named after its version, for example `sagot-ai-v1.5.2.zip`.

| Version | What changed | Answers produced with |
|---|---|---|
| v1.0 | Hybrid RAG with a cross-vendor verifier. The system measured in the thesis (`rag_eval/results_v2`). | prompt 1, no dense floor |
| v1.1 | Conversational recovery (no dead ends) and the deterministic tax calculator. | prompt 1, no dense floor |
| v1.2 | Live demo page (`/demo`). Concise generation prompt. | prompt 2, no dense floor |
| v1.3 | Dense-search floor: the top 4 semantic hits are always added to the reranker's picks (4–6 excerpts). Prompt 3 combines every relevant excerpt. | prompt 3, dense floor 4 |
| v1.4 | Demo compares with LLM only and REVIE (standard RAG removed). | same as v1.3 |
| v1.5 | Demo compares with REVIE only. Team access code so teammates can test through the ngrok link. | same as v1.3 |
| v1.5.1 | Cleanup, no behavior change: one README with a file-by-file guide; Docker, Cloudflare and share scripts removed; unused `app/baselines.py` removed; `EVAL_THESIS_DIR` chooses the folder the Thesis results tab reads. | same as v1.3 |
| v1.5.2 | Tax computation without rounding: every figure is exact, nothing is rounded to centavos or pesos. A per-period figure that repeats (e.g. ÷ 52) is shown to 8 decimals with "…", never rounded. Amounts finer than a centavo are asked again. Fixed: semi-monthly pay read as monthly; "self-employed" refused as mixed income. | same as v1.3 (RAG unchanged) |

## Reproducing an older version's answers

| To reproduce | Put in `.env` |
|---|---|
| v1.0 / v1.1 (what `rag_eval/results_v2` measured) | `RAG_PROMPT_VERSION=1` and `DENSE_FLOOR_K=0` |
| v1.2 | `RAG_PROMPT_VERSION=2` and `DENSE_FLOOR_K=0` |
| v1.3 – v1.5.2 | nothing (the defaults) |

If the settings match none of these, the demo and `summary.json` say "v1.5.2 (custom settings)". The LLM-only and standard-RAG demo baselines are in git tag `v1.3` if the ablation is ever needed again.

## Commit text for v1.5.2

**Summary**

```
v1.5.2: exact tax computation (no rounding), fix semi-monthly and self-employed parsing
```

**Description**

```
Tax computation only. RAG answers are produced exactly as in v1.3-v1.5.1.

- No rounding anywhere in the calculator (app/computation/calculator.py):
  taxable income, every SSS/PhilHealth/Pag-IBIG share, the tax due and the
  balance after withholding are exact Decimal results. Before, each step was
  rounded to centavos (e.g. PhilHealth 2.5% x P34,749.99 = P868.74975 was
  shown and used as P868.75).
- Per-period tax and take-home are one exact division of the annual figure.
  When the quotient repeats (e.g. / 52), it is shown to 8 decimals followed
  by "...", cut off, never rounded, and a note says so.
- Contributions are computed on exact fractions, so weekly pay (x 52 / 12)
  loses nothing; the annual total is always exact.
- Display (messages.py): computed figures always show centavos and every
  further digit (P3,765.00, P868.74975). Headline no longer says
  "Estimated"/"about"; the closing note says every figure is exact.
- Amounts finer than a centavo (800,000.555) are asked again, not rounded.
- Fixed (extract.py): "semi-monthly salary is P17,500" was read as monthly
  (tax P0 instead of P20,415.00); "self-employed" matched the employee
  pattern and was refused as mixed income.
- tools/test_computation_live.py: the independent oracle is exact too; new
  case C14 (P34,749.99/month). 39/39 pass against a running server.
- Tests: 165 pass (new tests/test_exact_computation.py checks 300 random
  salaries against an independent exact calculation). Version 1.5.2.
```
