# Sagot AI: conversational recovery and tax computation

This document covers two additions requested after the professor's review:

1. **No more dead ends.** When the bot can't answer, it guides the user toward a question it can answer.
2. **Tax computation.** The bot computes individual income tax from the user's figures, using deterministic code and verified rules.

Neither change touches the evaluated RAG path. Retrieval, generation, verification and the cache work exactly as before, and the trace outcome names are unchanged.

---

## 1. Architecture before and after

**Before:** question → sanitize → language → greeting? → year filter → cache → retrieve → generate → verify → answer **or one fixed refusal sentence**.

**After:**

```
question ─► sanitize ─► language ─► greeting? ─► INTENT (app/intent.py)
                                                   │
         ┌─────────────────┬───────────────────────┼────────────────────┬──────────────────────┐
         ▼                 ▼                       ▼                    ▼                      ▼
       CHAT          TAX_COMPUTATION          too vague          TAX_INFORMATION        OUT_OF_SCOPE
   fixed capability  app/computation/         clarifying         (unchanged RAG path)   (also RAG: the knowledge
   reply             (no model call)          question                │                  base decides scope)
                     │                                                ▼
                     ├─ extract inputs (EN/FIL/Taglish)         retrieve → generate → verify
                     ├─ merge with state (newest value wins)          │
                     ├─ validate (numbers, year, eligibility)         ├─ verified ─► answer + sources
                     ├─ ask for missing inputs                        └─ no evidence / not verified
                     ├─ rule = tax_rules/<type>.json, the ONE             ─► GUIDANCE (app/recovery.py):
                     │  verified version for that exact year                 honest status + clarifying
                     └─ Decimal calculator ─► breakdown + legal basis        question + example questions +
                                              + knowledge-base reference     closest documents + capabilities
```

### Existing components reused

| Need | Reused component |
|---|---|
| Language of every reply | `app/language.py` (detected once per turn). A bare "800,000" inherits the language the computation started in. |
| Input cleaning | `app/sanitize.py` (a one-character reply such as "8%" is allowed while a computation is open) |
| Greeting short-circuit | `app/greetings.py` |
| Year-aware retrieval | `app/classifier.get_year_filter`, unchanged |
| Hybrid retrieval + rerank | `app/retrieval.py`, unchanged |
| Grounded generation + verifier | `app/llm.py`, unchanged |
| Trace log | `app/trace.py`. Every computation turn is logged with its input state, output state and calculator result. |
| Knowledge base | Chroma. The computation checks, using metadata only, that the BIR FAQ page it cites is indexed. |

### Files created

| File | Purpose |
|---|---|
| `app/intent.py` | Rule-based routing: CHAT / TAX_INFORMATION / TAX_COMPUTATION / OUT_OF_SCOPE, plus a "too vague" flag and a topic |
| `app/recovery.py` | Guidance messages (English, Filipino, Taglish). No model call, no tax facts. |
| `app/computation/rules.py` | Loads and **validates** `tax_rules/*.json`; looks up the one rule version for (tax type, year) |
| `app/computation/calculator.py` | Deterministic Decimal calculator with a step-by-step breakdown |
| `app/computation/extract.py` | Reads amounts, years, situation, cancel and other-tax mentions from English, Filipino and Taglish |
| `app/computation/dialogue.py` | Multi-turn input collection, validation and computation |
| `app/computation/messages.py` | Every computation sentence, in 3 languages |
| `tax_rules/income_tax_graduated.json` | TRAIN graduated rates, 2018–2022 and 2023–2026 |
| `tax_rules/income_tax_8_percent.json` | 8% option for purely self-employed individuals, 2018–2026 |
| `tax_rules/income_tax_compensation.json` | Employees: tax from gross pay (₱90,000 benefits cap, contributions exclusion), 2018–2026 |
| `tax_rules/employee_contributions.json` | SSS, PhilHealth and Pag-IBIG employee shares, 2025–2026 |
| `tools/build_reference_note.py` | Builds the knowledge-base note `data/reference/Employee_Compensation_Tax_Reference.pdf` |
| `config/document_metadata.json` | Knowledge-base metadata per folder and per file |
| `tests/test_conversation.py` | 84 tests: the 16 scenarios, the employee mode, and safety and regression checks |

### Files modified

| File | Change |
|---|---|
| `app/api.py` | Intent routing, the computation branch, the clarification branch, and guidance instead of the fixed refusal. An open computation survives an unrelated turn. |
| `app/schemas.py` | `ComputationState` (re-validated on every request). Request field `computation_state`; response fields `intent` and `computation`. |
| `app/database.py` | `apply_document_metadata()` stamps `document_type`, `tax_type`, `tax_year`, `jurisdiction`, `computation_related` and `rule_ids` on every chunk |
| `app/language.py` | `language_from_label()` |
| `static/js/app.js` | Sends `computation_state` back each turn, clears it on New Chat, and adds labels for the new modes |
| `static/index.html` | Adds a "Compute my income tax" suggestion chip |
| `README.md` | Pipeline diagram and project layout |

---

## 2. Conversational recovery

**When it runs:** retrieval finds nothing (`no_retrieval`), the generator says NO_ANSWER (`generator_no_answer`), no draft passes the verifier (`verification_failed`), the index is missing (`no_index`), or the message is too vague to retrieve on (`clarification`).

**What the user gets.** There are always three parts, kept apart so the answer and the guidance can't be confused:

1. **An honest status.** For example: "I don't have enough information in my documents to answer that yet." It is never an answer.
2. **A way forward:**
   - a clarifying question for the detected topic (income, computation, withholding, VAT, corporate, filing, registration, invoicing, general)
   - 2–3 example questions phrased the way the system can answer them
   - when retrieval found something, the titles of the closest documents
3. **What the bot can do:** BIR rules and issuances from its documents, plus individual income tax computation.

**Why it can't hallucinate:**
- No model writes it.
- The example questions are questions, not claims.
- Document titles come from chunk metadata.
- The closest documents are never listed as `sources`, so they don't look like support for an answer.

**Evaluation compatibility:**
- Outcome codes are unchanged, so `rag_eval` still counts these replies as refusals.
- Only the text shown to the user changed.
- `RECOVERY_ENABLED=0` restores the old one-line refusal, for a baseline-identical ablation run.

**Why OUT_OF_SCOPE still retrieves.** A word-list scope test is too brittle for this corpus. For example, "Is POGO now banned?" contains no tax word but is answered by a BIR issuance. The first version of the word list also missed 4 of the 100 T-TED questions (natural-gas incentives, ecozone purchases, the landed value of imported vehicles). So the knowledge base is the scope test, and the OUT_OF_SCOPE label only chooses the wording of the guidance if no evidence is found.

---

## 3. Tax computation

### Where the rates come from

The knowledge-base PDFs explain the computation steps (BIR FAQ, Question 10). They do not contain the bracket table; they only say "0% to 35%". So the rates are stored as **structured, versioned rule files** (`tax_rules/*.json`). Each version records:
- its tax-year range
- its legal basis (Sec. 24(A)(2)(a)/(b), NIRC as amended by RA 10963)
- its sources (the RA 10963 text; the Senate Tax Study and Research Office TRAIN primer)
- a verification record

The loader **rejects** a rule file when:
- its brackets are not contiguous, or
- a bracket's base tax differs from the tax computed at its lower bound from the bracket below, or
- two versions of the same tax claim the same year.

That check matters for this thesis. The RA 10963 text as published prints the 2018–2022 32% bracket as "not over ₱5,000,000", while the next bracket starts at ₱8,000,000 with a ₱2,410,000 base. ₱490,000 + 32% × ₱6,000,000 = ₱2,410,000, so the bound must be ₱8,000,000. The validator rejects the misprinted version, and a test asserts that.

Print and check every loaded rule with:

```
python -m app.computation.rules
```

### The flow

1. **Detect intent.** A compute verb (compute, calculate, i-compute, kwentahin…), or "how much tax / magkano … tax" together with "my/ko/kong", "I owe" or an amount. "How is income tax computed?" stays a RAG question, because the FAQ explains the steps.
2. **Determine the tax type:**
   - taxable income, salary, or nothing stated yet → graduated
   - "8%" → the 8% option
   - self-employed without a stated option → ask which option
   - VAT, estate, donor's, corporate, mixed-income, etc. → *no rule loaded*: say so, no calculation
3. **Determine the year.** It is never assumed. A year outside the rule's verified range is refused, and the user is shown the supported range. No neighbouring year's rule is ever used.
4. **Look up the rule.** `Registry.lookup(tax_type, year)` returns the **one** verified version that covers that exact year.
5. **Identify the required inputs** from the rule file's `inputs`.
6. **Check what's missing.** Ask for exactly those inputs: grouped on the first turn, one at a time afterwards.
7. **Validate.** Malformed numbers ("800,00"), negatives, monthly figures, two different values for the same input, two years, and non-numeric replies are all asked again. None of them is used.
8. **Calculate** with Decimal arithmetic, rounded half-up to centavos.
9. **Return the breakdown:** inputs, rule, legal basis, bracket, each arithmetic step, tax withheld and the balance (if given), and any assumption made.
10. **Cite the sources:** the rule's legal basis and references, plus the knowledge-base page if it is indexed.

### Memory across turns

The state (collected values, what is still needed, the language) is returned to the browser with every computation reply and sent back with the next message. The server keeps no sessions.

The state is re-validated on every request:
- unknown keys and malformed values are dropped
- a rate can never come from the client

On each turn:
- the newest value always replaces the old one, and the reply says so ("₱800,000 → ₱850,000")
- a real question asked mid-computation goes to RAG, and the open computation is kept
- "cancel" / "huwag na" clears it

### Supported computations

| Tax type | Years | Inputs |
|---|---|---|
| Individual income tax, graduated rates | 2018–2022, 2023–2026 (separate tables) | tax year, annual taxable income, tax withheld (optional) |
| 8% option, purely self-employed (≤ ₱3M VAT threshold) | 2018–2026 | tax year, gross sales/receipts, other non-operating income (optional), tax withheld (optional) |
| Employees, from gross salary | 2018–2026; contributions auto-computed for 2025–2026 | tax year, gross pay, pay period (monthly, semi-monthly, weekly, annual), 13th-month pay and other benefits (optional), contributions (only when there is no verified schedule for the year), tax withheld (optional) |

### Employee mode (from gross salary)

Triggered when the user gives a salary or pay with a pay period ("₱35,000 a month", "sahod ko 60k kada buwan"), mentions 13th-month pay or contributions, or asks for "the tax on my salary". If they give *taxable* income, the graduated path is used as before.

1. **Annualize:** monthly × 12, semi-monthly × 24, weekly × 52. Daily pay is *not* annualized, because paid days per year differ between employers; the user is asked for monthly or annual pay.
2. **Exclude mandatory contributions** (NIRC Sec. 32(B)(7)(f)). For 2025–2026 they are computed from `tax_rules/employee_contributions.json`:
   - **SSS:** 5% of the Monthly Salary Credit, ₱5,000–₱35,000 in ₱500 steps (Circular 2024-006)
   - **PhilHealth:** 2.5% (half of 5%) of salary between ₱10,000 and ₱100,000 (RA 11223 Sec. 10; Circular 2020-0005; Advisories PA2025-0002 and PA2026-0042)
   - **Pag-IBIG:** 2% of pay up to ₱10,000 (HDMF Circular 460)

   For any other year, the bot asks for the user's own total (from payslips or Form 2316) instead of borrowing 2025 rates.
3. **Add 13th-month pay and other benefits above ₱90,000** (Sec. 32(B)(7)(e), as amended by RA 10963). If none are given, the reply says it assumed ₱90,000 or less.
4. **Apply the graduated table for the same year**, then show the tax per pay period and the estimated take-home pay.

The loader rejects the employee rule if the graduated table it depends on is missing. Cross-check: taxcalculator.com.ph's worked example (₱35,000 a month in 2025: ₱33,900 contributions, ₱386,100 taxable, ₱20,415 tax, ₱30,473.75 take-home) is reproduced exactly by a test.

**Source note.** The SSS circular PDF could not be fetched from sss.gov.ph. Its figures are confirmed by the SSS circulars index and two independent summaries of the circular (KPMG, Grant Thornton); the rule file records this. The Pag-IBIG 1% rate for pay of ₱1,500 and below was not confirmed officially, so contributions aren't auto-computed at that level.

**Knowledge-base note.** `tools/build_reference_note.py` writes a 12-question reference PDF from the same official sources: the exclusions, no personal exemptions, both rate tables, the 2025–2026 contribution rules and a worked example. Its example figures are produced by the calculator. Put it in `data/reference/` and run `python -m app.database` (no `--reset` needed). Its chunks are tagged `document_type: reference_note` and labeled "Reference Notes". None of its questions overlaps the T-TED set.

**To add a tax:** add a JSON file in `tax_rules/`. If its method is new, add one function to `calculator.METHODS` and its templates to `messages.py`.

**To extend a year range** (e.g. to 2027): first confirm the law hasn't changed, then change `tax_year_to`.

---

## 4. Example conversations (actual output)

**Multi-turn computation in English, with a correction:**

> **User:** Can you calculate my income tax?
> **Bot:** Sure, I can calculate that. I just need a few details first:
> - **Tax year** (e.g. 2025)
> - **Annual taxable income** (e.g. ₱800,000)
>
> If you're purely self-employed and chose the 8% option, tell me your gross sales/receipts instead.
>
> **User:** 800,000
> **Bot:** Got it. Which **tax year** should I use? (e.g. 2025) The rates depend on the year, so I won't assume one.
>
> **User:** 2025
> **Bot:** **Estimated income tax due for 2025: ₱102,500**
> - Rule applied: Graduated income tax rates for individuals (TRAIN schedule for tax years 2023 onwards)
> - Legal basis: Sec. 24(A)(2)(a), NIRC of 1997 as amended by Sec. 5 of RA No. 10963 (TRAIN Law)
> - Bracket: over ₱400,000 but not over ₱800,000: ₱22,500 + 20% of the excess over ₱400,000
>
> 1. Excess over ₱400,000: ₱800,000 − ₱400,000 = ₱400,000
> 2. 20% × ₱400,000 = ₱80,000
> 3. Tax due: ₱22,500 + ₱80,000 = **₱102,500**
>
> **User:** Actually, it's 850,000.
> **Bot:** Updated your annual taxable income: ₱800,000 → **₱850,000**.
> **Estimated income tax due for 2025: ₱115,000** …

**Filipino, unsupported year:**

> **User:** Kwentahin mo ang income tax ko para sa 2015, taxable income ko ay 500,000
> **Bot:** Wala akong verified na rule para sa **Graduated na income tax rates para sa mga indibidwal** sa taong **2015**, kaya hindi ko ito ico-compute gamit ang rates ng ibang taon. Kaya kong mag-compute para sa taong **2018 hanggang 2026**. Alin doon ang gagamitin ko?
> **User:** 2024
> **Bot:** **Tinatayang income tax due para sa 2024: ₱42,500** …

**No rule loaded:**

> **User:** Compute my VAT for 2025
> **Bot:** I can't compute **VAT** yet: no validated computation rule for it has been loaded, and I won't make up rates. Right now I can compute **individual income tax** (graduated rates, or the 8% option for purely self-employed individuals). I can still answer questions about VAT from my documents. Just ask.

**Recovery instead of a dead end:**

> **User:** VAT
> **Bot:** I'd like to help. Could you tell me a bit more about what you want to know?
> What about VAT: a specific transaction (like digital services or sales), registration, or a particular issuance?
> You could ask, for example: *Is VAT imposed on digital services like video streaming and online platforms?* …

---

## 5. Tests

```
python -m pytest tests -q        # 121 passed (37 existing + 84 new), offline, about 2 s
```

`tests/test_conversation.py` replaces retrieval, the models and Chroma with fakes.

**Your 16 scenarios:**

| # | Scenario | Test |
|---|---|---|
| 1 | Relevant question → RAG answer | `test_s1_…` |
| 2 | Vague → clarification, no retrieval | `test_s2_…` |
| 3 | Unsupported → guidance toward tax topics | `test_s3_…` |
| 4 | Incomplete → follow-up question | `test_s4_…` |
| 5 | No retrieval → guidance, not a dead end | `test_s5_…`, `test_s5b_…` (closest documents) |
| 6 | Taglish guidance | `test_s6_…`, plus a Filipino variant |
| 7 | Complete computation | `test_s7_…` (also asserts no model or retrieval call) |
| 8 | Missing input → follow-up | `test_s8_…` |
| 9 | Multi-turn memory | `test_s9_…` |
| 10 | Changed value → recompute | `test_s10_…` |
| 11 | Year selects its own rule | `test_s11_…` (2022 → ₱130,000; 2023 → ₱102,500) |
| 12 | Unsupported year → no calculation | `test_s12_…` |
| 13 | Missing rule → no fabrication | `test_s13_…` (VAT), `test_s13b_…` (rule file absent) |
| 14 | Invalid numbers | `test_s14_…` (malformed, negative, monthly, conflicting, words) |
| 15 | Taglish computation | `test_s15_…`, `test_s15b_…` (Filipino kept for bare numbers) |
| 16 | English computation | `test_s16_…` |

**Additional checks:**
- **Routing:** no T-TED question is routed away from RAG; intent labels for the spec's examples.
- **Rule validation:** the RA 10963 misprint, a wrong base tax and overlapping years are all rejected; an unverified rule is never used.
- **Calculation:** bracket boundaries for both tables; no neighbouring-year fallback.
- **Inputs:** amount formats and non-amounts; a tampered client state can't inject values.
- **Conversation:** a mid-computation question goes to RAG and the computation is kept; cancel; 8% eligibility.
- **Ablation and ingestion:** `RECOVERY_ENABLED=0`; metadata stamping at ingestion.

**End to end:** the real server was run against the real Chroma index, and the chat UI was driven in a browser (Chromium). The multi-turn computation completed, the state round-tripped through the page, and the knowledge-base reference was found in the real index. The live RAG path (Jina + OpenRouter) was **not** called during this work, so no API credits were used.

---

## 6. Limitations and risks

- **Scope of computation.** Individual income tax only: employees from gross salary, anyone from annual taxable income, and the 8% option for the purely self-employed. Not computed: minimum wage earners, mixed-income earners, corporate tax/MCIT, VAT, percentage tax, estate/donor's tax, and business deductions (OSD or itemized). When a user says "income" without a pay period, the reply states that it was treated as taxable income.
- **Employee estimates.** PhilHealth is computed on the pay the user gives, but the law bases it on *basic* salary, so pay that includes allowances overstates it slightly (the reply says so). Union dues are excluded by law but not computed; the user can include them in a manual contributions total. Contribution schedules cover 2025–2026 only; extend them per year after checking each agency for a newer circular.
- **Rates are not in the PDF corpus.** They are in `tax_rules/`, checked against RA 10963 and the Senate primer. For the cleanest defense story, add the official RA 10963 / RR 8-2018 PDF to `data/` and tag it in `config/document_metadata.json` as `"document_type": "tax_rule_source"`.
- **Year coverage is manual.** `tax_year_to` is 2026. After 2026 the bot refuses until someone verifies the law and extends it. That is deliberate.
- **Rule-based extraction and intent.** Both are explainable and tested, but phrasing outside the patterns can be missed (e.g. amounts written in words are asked again, never guessed). Watch `logs/queries.jsonl` for `intent` and `computation` entries and extend the patterns.
- **Language per turn.** If a user starts in Taglish and then writes a full English sentence, that reply is in English, as the spec asks. Only marker-less replies ("800,000", "2025") inherit the earlier language.
- **State lives in the browser tab.** Refreshing the page or clicking New Chat ends an open computation.
- **Rate limit.** Each computation turn counts toward `RATE_LIMIT_PER_MIN` (default 6) and `DAILY_QUERY_CAP`, even though it costs no model call.
- **Evaluation.** Refused T-TED items now show guidance text instead of the one-line refusal. If an answer-quality metric scores refusals, use `RECOVERY_ENABLED=0` when comparing against the old baseline.
- **Metadata on existing chunks.** The new knowledge-base metadata applies to chunks indexed from now on. Run `python -m app.database --reset` to stamp the whole index (this re-embeds everything, which takes a few minutes on the free Jina tier).

## 7. Settings

| Variable | Default | Meaning |
|---|---|---|
| `RECOVERY_ENABLED` | 1 | 0 restores the old one-line refusal (ablation) |
| `TAX_RULES_DIR` | `tax_rules` | Where rule files are loaded from |
| `ALLOW_UNVERIFIED_RULES` | 0 | 1 lets an unverified rule version compute (development only) |
