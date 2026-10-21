# v1.6: mitigating hallucination, raising groundedness and relevance

This explains what v1.6 changes, which thesis metric each change targets, the evidence behind each default, and how to measure it. Nothing here has been run against T-TED. Every number below comes from `logs/queries.jsonl` (the team's own testing, 2026-10-02 to 10-08) with T-TED look-alike questions excluded.

## The problem in the logs

| What the logs showed | Effect on the metrics |
|---|---|
| 14 of 49 verified answers to non-T-TED questions were refusals in the model's own words ("Hindi direktang sinasagot ng mga excerpt…") followed by loosely related facts. They were badged, cached and scored as answers. | Answer Relevance falls (the answer is about something else). Users see a "verified" answer that answers nothing. |
| Off-topic questions (Netflix tax, "malungkot ako", Rizal) all had a best reranker score of 0.03–0.13. Real BIR questions scored 0.26 and up. The reranker always keeps at least one chunk and the dense floor added more, so the generator nearly always had something to talk about. | Same as above, plus Context Relevance falls (irrelevant excerpts). |
| 9 verified answers named "RA No. 11590". No excerpt they were given contains it. The AI verifier passed all of them. | Groundedness: a real hallucination that the verifier missed. |
| Prompt v3 told every answer to start with Yes/No, so "Paano ko makukuha ang TIN ko?" got "Oo, makukuha mo…". | Answer Relevance. |
| The fresh-grad "how do I get my TIN?" was answered from RMC 37-2024, which is about recovering an existing TIN. Fully grounded, wrong question. | Answer Relevance. The verifier checks support, not relevance. |

## What v1.6 does

Each row is one switch in `.env`. All off reproduces v1.5.2.

| Change | Where | Targets | Switch (default) |
|---|---|---|---|
| **Evidence gate.** Best reranker score below the threshold: no answer is written, guidance is shown. Outcome stays `no_retrieval`. | `app/answerability.py`, `app/api.py` | hallucination, answer relevance | `EVIDENCE_GATE_MIN_SCORE` (0.14) |
| **Non-answer detector.** A draft whose first sentence says the excerpts don't answer counts as `generator_no_answer`. A partial answer that answers first is kept. | `app/answerability.py` | answer relevance | `NON_ANSWER_DETECTION` (1) |
| **Conditional dense floor.** A top dense hit is added back only if the reranker gives it at least half the best score, or it is from the same issuance. | `app/retrieval.py` | context relevance | `DENSE_FLOOR_RELATIVE` (0.5) |
| **Sentence selection.** Only the sentences about the question reach the generator and the verifier (existing code, now on). | `app/retrieval.py` | context relevance, groundedness | `EVIDENCE_COMPRESSION` (1) |
| **Rule-based grounding check.** Every ₱ amount, %, issuance/RA/form number and e-mail must be in the excerpt the sentence cites. Compared by value: ₱3,000,000.00 = P3 million = "Three Million Pesos (P3,000,000.00)". | `app/grounding.py` | groundedness | `GROUNDING_CHECK` (1) |
| **Claim repair.** A flagged sentence that isn't the opening answer is removed, and the rest goes to the verifier. A sentence the verifier rejects is removed the same way, and the rest is verified once more. Nothing unverified is ever shown. | `app/grounding.py`, `app/llm.py` | groundedness (keeps supported content) | `CLAIM_REPAIR` (1) |
| **Relevance-aware verifier.** The verifier also sees the question and says whether the draft answers it. Grounding is judged exactly as before. | `app/prompts.py` (`VERIFY_PROMPT_V2`), `app/llm.py` | answer relevance | `VERIFIER_RELEVANCE` (1) |
| **Prompt v4.** Yes/No only for yes/no questions. The answer's shape follows the question (steps, figure, definition, date). Never open with what the excerpts lack. No arithmetic of its own. | `app/prompts.py` | answer relevance, groundedness | `RAG_PROMPT_VERSION` (4) |
| **Extra Filipino keyword terms** (29 words from the logs). A test fails if any appears in a T-TED question. | `app/retrieval.py` | context relevance | `QUERY_GLOSSARY_V16` (1) |

Routing fixes from the 2026-10-07 log review: "what is eFPS?" now reaches RAG; a capability question with extra text is chat; Taglish compute verbs (magcompute, computein, gaano kalaki) reach the calculator; OFW pay is not computed as resident compensation; SSS/PhilHealth/Pag-IBIG questions explain that these are contributions and show the employee shares. A model error now shows "please try again" (HTTP 200) instead of a server error. The evaluation still sees the error and retries the unit.

## Why these are defensible

- **Rules where rules work.** The gate, the non-answer detector and the grounding check are deterministic, free and repeatable. They catch what an LLM verifier misses: the rule check flagged 14 of 390 logged verified answers, and all 14 were real problems.
- **Fail-closed is unchanged.** Repair only removes sentences, never adds, and the repaired text must pass the verifier itself.
- **No test-set leakage.** Thresholds came from the logs. `tools/dev_tune.py` refuses a DEV file containing a T-TED look-alike. Tests prove the new keyword terms and the DEV questions don't come from T-TED. The "no T-TED question is routed away from RAG" test still passes.
- **Ablation-ready.** Each change has a switch. `python -m tools.dev_tune ablation` runs the ladder v1.5.2 → +gate → +ground → +context → v1.6 on DEV and prints one table.

## Trade-offs to state at the defense

- **The gate can refuse an answerable question** whose best excerpt scores low. In the logs, "may vat na ba sa roblox?" (0.12) would be refused, although a general digital-services VAT excerpt arguably applies. That is a deliberate precision-over-recall choice. Tune it with `gate-sweep` on DEV.
- **Sentence selection raises Context Relevance partly by construction** (fewer sentences, more of them relevant). The real justification is that the generator sees less noise. Report it with Answer Correctness alongside, so a drop in completeness would show.
- **More refusals.** Non-answers and weak evidence now become guidance. The refusal rate on T-TED will likely rise, while the quality of what is answered should rise too. Report both.
- **Cost.** Sentence selection adds one Jina reranker call per question. Claim repair adds at most one verifier call per attempt. The gate saves the whole generation when it fires.
- **The rule check is narrow on purpose.** It checks figures and identifiers, not meaning. Meaning stays the verifier's job.

## How to measure (in order)

```powershell
python -m tools.dev_tune logs                 # free: what v1.6 does to the logged questions
python -m tools.dev_tune gate-sweep           # Jina only: confirm EVIDENCE_GATE_MIN_SCORE on DEV
python -m tools.dev_tune ablation --limit 5   # smoke test (full pipeline + judge)
python -m tools.dev_tune ablation             # the ablation table → dev/results/ablation.csv
python -m rag_eval.run_evaluation --fresh     # only then: one T-TED run of the final v1.6
```

Check and finish the labels in `dev/dev_questions.jsonl` before trusting DEV numbers (set `label_status` to `checked`, add references where you can). Compare the T-TED run against `rag_eval/results_v2`. That run used prompt 1 with no v1.3+ changes, so the comparison covers v1.3–v1.6 together. Run-to-run noise is about ±0.02 per metric.
