"""
Tests for conversational recovery (no dead ends) and tax computation.
No network: retrieval, the generator/verifier and Chroma are replaced with
fakes, so these run offline in about a second.

Run from the project root:  python -m pytest tests -q

Scenario numbers (S1..S16) match the thesis test plan:
  Recovery:     S1 relevant -> RAG, S2 vague -> clarify, S3 unsupported ->
                guidance, S4 incomplete -> follow-up, S5 no retrieval ->
                guidance, S6 Taglish guidance
  Computation:  S7 complete, S8 missing info, S9 multi-turn memory,
                S10 value update, S11 year-specific rule, S12 unsupported
                year, S13 missing rule, S14 invalid input, S15 Taglish,
                S16 English
"""

import json
import os
from decimal import Decimal

os.environ.setdefault("OPENROUTER_API_KEY", "test")
os.environ.setdefault("JINA_API_KEY", "test")

import pytest  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from app import intent as intent_router  # noqa: E402
from app import recovery  # noqa: E402
from app.computation import dialogue  # noqa: E402
from app.computation.calculator import compute  # noqa: E402
from app.computation.extract import extract  # noqa: E402
from app.computation.rules import Registry, RuleError, get_registry, load_registry, parse_rule  # noqa: E402
from app.llm import RagResult  # noqa: E402
from app.schemas import ComputationState, QueryRequest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_REFUSAL = "I'm sorry, but I couldn't find a relevant answer"


# ── Fake pipeline ─────────────────────────────────────────────────────────

class FakeChroma:
    def __init__(self, *a, **kw):
        pass

    def get(self, include=None, where=None, limit=None):
        if where is not None:  # knowledge-base reference lookup for computations
            return {"ids": ["faq:x:4:0"], "metadatas": [{"title": "BIR_Income_Tax_FAQ", "page": 4, "year": "faq"}]}
        return {"ids": ["chunk-1", "chunk-2"]}


class World:
    """What retrieval and the models will do for the next query."""

    def __init__(self):
        self.docs = []
        self.rag_outcome = "verified"
        self.retrieve_calls = 0
        self.rag_calls = 0

    def retrieve(self, query, year_filter, db):
        self.retrieve_calls += 1
        return list(self.docs)

    def run_rag(self, query, history, context_text, language=None):
        self.rag_calls += 1
        if self.rag_outcome == "verified":
            return RagResult("verified", "Individuals earning purely compensation income not over P250,000 need not file [1].",
                             language.label)
        return RagResult(self.rag_outcome, "refusal", language.label)


def _doc(title="BIR_Income_Tax_FAQ", year="faq", page=1, summary="BIR Income Tax FAQ (Philippines)"):
    return Document(page_content="Some text.", metadata={
        "id": f"{year}:{title}:{page}:0", "title": title, "year": year, "page": page,
        "source": f"data/{year}/{title}.pdf", "doc_summary": summary, "_rrf_score": 0.03,
    })


@pytest.fixture
def world(monkeypatch):
    from app import api
    w = World()
    monkeypatch.setattr(api.trace_log, "write", lambda tr: None)
    monkeypatch.setattr(api.answer_cache, "CACHE_ENABLED", False)
    monkeypatch.setattr(api, "get_embedding_function", lambda: None)
    monkeypatch.setattr(api, "_warn_if_stale_index", lambda: None)
    monkeypatch.setattr(api, "Chroma", FakeChroma)
    monkeypatch.setattr(api, "retrieve_docs", w.retrieve)
    monkeypatch.setattr(api, "run_rag", w.run_rag)
    w.api = api
    return w


def ask(world, text, state=None):
    response, _trace = world.api.run_query(QueryRequest(query=text, computation_state=state))
    return response


def converse(world, *messages):
    state, replies = None, []
    for m in messages:
        r = ask(world, m, state)
        replies.append(r)
        state = r.computation
    return replies


# ── Conversational recovery ──────────────────────────────────────────────

def test_s1_relevant_question_gets_normal_rag_answer(world):
    world.docs = [_doc()]
    r = ask(world, "Who is not required to file an income tax return?")
    assert r.mode == "rag" and r.outcome == "verified"
    assert r.sources and r.sources[0].startswith("[1]")
    assert world.rag_calls == 1
    assert r.intent == "TAX_INFORMATION"


def test_s2_vague_question_gets_clarification_without_retrieval(world):
    r = ask(world, "tax?")
    assert r.mode == "clarify" and r.outcome == "clarification"
    assert world.retrieve_calls == 0 and world.rag_calls == 0
    assert "Could you tell me a bit more" in r.answer
    assert "You could ask, for example:" in r.answer


def test_s3_unsupported_question_gets_guidance_toward_tax_topics(world):
    world.docs = [_doc()]
    world.rag_outcome = "generator_no_answer"
    r = ask(world, "Can you recommend a good adobo recipe for dinner?")
    assert r.intent == "OUT_OF_SCOPE"
    assert r.outcome == "generator_no_answer"           # trace outcome unchanged (rag_eval counts it as a refusal)
    assert r.mode == "guidance"
    assert "outside what I can help with" in r.answer
    assert "What I can help with" in r.answer
    assert OLD_REFUSAL not in r.answer
    assert r.sources == []                               # never shown as supporting evidence


def test_s4_related_but_incomplete_question_gets_follow_up(world):
    r = ask(world, "deadline?")
    assert r.mode == "clarify"
    assert "Which return or form is it, and for which year?" in r.answer


def test_s5_no_retrieval_result_gives_guidance_not_dead_end(world):
    world.docs = []
    r = ask(world, "What is the tax treatment of cryptocurrency mining rewards for individuals?")
    assert r.outcome == "no_retrieval" and r.mode == "guidance"
    assert "I don't have enough information in my documents to answer that yet." in r.answer
    assert "You could ask, for example:" in r.answer
    assert world.rag_calls == 0


def test_s5b_unverified_answer_names_closest_documents_only(world):
    world.docs = [_doc("RMC No. 99-2025", "2025", summary="Clarifies the filing of returns")]
    world.rag_outcome = "verification_failed"
    r = ask(world, "What are the rules for filing returns of estates under trust?")
    assert r.outcome == "verification_failed" and r.mode == "guidance"
    assert "couldn't confirm an answer" in r.answer
    assert "RMC No. 99-2025: Clarifies the filing of returns" in r.answer
    assert r.sources == []


def test_s6_taglish_incomplete_question_gets_taglish_guidance(world):
    r = ask(world, "What about yung tax?")
    assert r.language == "taglish" and r.mode == "clarify"
    assert "Gusto kitang i-help" in r.answer
    world.docs = []
    r = ask(world, "Ano yung tax rules about crypto mining sa bahay ko?")
    assert r.language == "taglish" and r.mode == "guidance"
    assert "Wala pa akong enough na information" in r.answer


def test_filipino_guidance_is_in_filipino(world):
    world.docs = []
    r = ask(world, "Ano ang patakaran sa pagbubuwis ng mga minana na alahas?")
    assert r.language == "filipino"
    assert "Wala pa akong sapat na impormasyon" in r.answer


def test_chat_capability_question(world):
    r = ask(world, "What can you do?")
    assert r.mode == "chat" and world.retrieve_calls == 0
    assert "computing individual income tax" in r.answer


def test_recovery_can_be_turned_off_for_ablation(world, monkeypatch):
    monkeypatch.setattr(recovery, "RECOVERY_ENABLED", False)
    world.docs = []
    r = ask(world, "What is the tax treatment of cryptocurrency mining rewards for individuals?")
    assert r.mode == "no_answer" and r.answer.startswith(OLD_REFUSAL)


# ── Tax computation through the API ──────────────────────────────────────

def test_s7_complete_computation_query(world):
    r = ask(world, "My taxable income is ₱800,000 for 2025. How much income tax do I need to pay?")
    assert r.intent == "TAX_COMPUTATION"
    assert r.mode == "computation" and r.outcome == "computation_done"
    assert "**Estimated income tax due for 2025: ₱102,500**" in r.answer
    assert "₱22,500 + ₱80,000 = **₱102,500**" in r.answer
    assert any("RA No. 10963" in s for s in r.sources)
    assert any(s.startswith("Knowledge base: BIR_Income_Tax_FAQ, p.5") for s in r.sources)
    assert world.rag_calls == 0 and world.retrieve_calls == 0   # no LLM involved in the number


def test_s8_missing_information_asks_follow_up(world):
    r = ask(world, "Calculate my income tax.")
    assert r.outcome == "computation_needs_input" and r.mode == "computation_input"
    assert "**Tax year**" in r.answer and "**Annual taxable income**" in r.answer
    assert "Estimated" not in r.answer
    assert r.computation is not None and r.computation.awaiting == ["tax_year", "taxable_income"]


def test_s9_multi_turn_preserves_values(world):
    r1, r2, r3 = converse(world, "Can you calculate my income tax?", "800,000", "2025")
    assert r2.outcome == "computation_needs_input"
    assert "Which **tax year** should I use?" in r2.answer
    assert r2.computation.values["taxable_income"] == "800000.00"
    assert r3.outcome == "computation_done"
    assert "₱102,500" in r3.answer


def test_s10_changed_value_replaces_old_one(world):
    *_, r4 = converse(world, "Can you calculate my income tax?", "My income is 800,000", "2025",
                      "Actually, it's 850,000.")
    assert r4.outcome == "computation_done"
    assert "₱800,000 → **₱850,000**" in r4.answer
    assert "**Estimated income tax due for 2025: ₱115,000**" in r4.answer
    assert r4.computation.values["taxable_income"] == "850000.00"


def test_s11_year_selects_its_own_rule(world):
    r2022 = ask(world, "Compute my income tax for 2022, taxable income 800,000")
    r2023 = ask(world, "Compute my income tax for 2023, taxable income 800,000")
    assert "₱130,000" in r2022.answer and "2018 to 2022" in r2022.answer
    assert "₱102,500" in r2023.answer and "2023 onwards" in r2023.answer


def test_s12_unsupported_year_is_not_computed_with_another_years_rule(world):
    r = ask(world, "Compute my income tax for 2015, taxable income 800,000")
    assert r.outcome == "computation_unsupported_year"
    assert "Estimated" not in r.answer and "₱102,500" not in r.answer and "₱130,000" not in r.answer
    assert "2018 to 2026" in r.answer
    assert "tax_year" not in r.computation.values and r.computation.awaiting == ["tax_year"]
    r2 = ask(world, "2024", r.computation)        # the user picks a supported year; income is kept
    assert r2.outcome == "computation_done" and "₱102,500" in r2.answer


def test_s13_missing_rule_is_not_fabricated(world):
    r = ask(world, "Compute my VAT for 2025, my sales are 2,000,000")
    assert r.outcome == "computation_no_rule"
    assert "I can't compute **VAT** yet" in r.answer
    assert "12%" not in r.answer and "Estimated" not in r.answer
    assert r.computation is None


def test_s13b_rule_file_missing_means_no_computation(tmp_path):
    empty = load_registry(tmp_path)
    t = dialogue.handle("Compute my income tax for 2025, taxable income 800,000", "english", registry=empty)
    assert t.outcome == "computation_no_rule" and "Estimated" not in t.answer


def test_s14_invalid_numbers_are_rejected(world):
    r1 = ask(world, "Compute my income tax for 2025")
    r2 = ask(world, "800,00", r1.computation)
    assert r2.outcome == "computation_invalid_input" and "**800,00**" in r2.answer
    assert "taxable_income" not in r2.computation.values
    r3 = ask(world, "-50,000", r2.computation)
    assert r3.outcome == "computation_invalid_input"
    r5 = ask(world, "600,000 or 700,000", r3.computation)
    assert r5.outcome == "computation_invalid_input" and "Which one should I use?" in r5.answer
    r6 = ask(world, "eight hundred thousand", r5.computation)
    assert r6.outcome == "computation_needs_input" and "I need a number" in r6.answer
    r7 = ask(world, "600,000", r6.computation)
    assert r7.outcome == "computation_done" and "₱62,500" in r7.answer


def test_s15_taglish_computation_in_taglish(world):
    r = ask(world, "Pa-compute naman ng income tax ko for 2025, taxable income ko is 800k")
    assert r.language == "taglish" and r.outcome == "computation_done"
    assert "ng excess over ₱400,000" in r.answer
    assert "Puwede mong palitan ang kahit anong value" in r.answer


def test_s15b_filipino_multi_turn_keeps_filipino_for_bare_numbers(world):
    r1, r2 = converse(world, "Magkano tax ko kung 800k ang taxable income ko?", "2025")
    assert r1.language == "filipino" and "Taon ng buwis" in r1.answer
    assert r2.language == "filipino"                     # "2025" alone has no language markers
    assert "Tinatayang income tax due para sa 2025: ₱102,500" in r2.answer


def test_s16_english_computation_in_english(world):
    r = ask(world, "Calculate my income tax for 2023: taxable income 1,200,000, my employer withheld 200,000")
    assert r.language == "english"
    assert "Excess over ₱800,000: ₱1,200,000 − ₱800,000 = ₱400,000" in r.answer
    assert "₱202,500 − ₱200,000 = **₱2,500 still payable**" in r.answer


def test_eight_percent_option_and_eligibility(world):
    r1, r2, r3 = converse(world, "freelancer ako, compute mo tax ko sa 2025", "8%", "gross receipts 1,500,000")
    assert "aling option" in r1.answer
    assert r3.outcome == "computation_done" and "₱100,000" in r3.answer
    r = ask(world, "I'm a freelancer on the 8% option, gross receipts 3,500,000 for 2024, compute my tax")
    assert r.outcome == "computation_not_eligible" and "₱3,000,000" in r.answer
    assert r.computation.tax_type == "income_tax_graduated"


def test_info_question_during_computation_goes_to_rag_and_keeps_state(world):
    world.docs = [_doc()]
    r1 = ask(world, "Calculate my income tax")
    r2 = ask(world, "What is the deadline for 2025?", r1.computation)   # contains a year, but is a new question
    assert r2.mode == "rag" and world.rag_calls == 1
    assert r2.computation is not None and r2.computation.status == "collecting"
    assert "computation is still open" in r2.answer
    r3 = ask(world, "800k", r2.computation)
    assert r3.outcome == "computation_needs_input"


def test_cancel_clears_computation(world):
    r1 = ask(world, "Calculate my income tax")
    r2 = ask(world, "never mind", r1.computation)
    assert r2.outcome == "computation_cancelled" and r2.computation is None


def test_tampered_state_cannot_inject_values():
    s = ComputationState(values={"taxable_income": "abc", "rate": "0.01", "tax_year": "1850", "regime": "zero"})
    assert s.values == {}


# ── Routing safety: the evaluated RAG path is unchanged ─────────────────

def test_no_tted_question_is_diverted_from_rag():
    with open(os.path.join(ROOT, "rag_eval", "ted_dataset.json"), encoding="utf-8") as f:
        items = json.load(f)
    for it in items:
        i = intent_router.classify(it["query"])
        assert i.label in (intent_router.TAX_INFORMATION, intent_router.OUT_OF_SCOPE), it["qid"]
        assert not i.vague, it["qid"]


@pytest.mark.parametrize("text,label", [
    ("Magkano tax ko kung 800k ang taxable income ko?", "TAX_COMPUTATION"),
    ("Can you compute my income tax?", "TAX_COMPUTATION"),
    ("How much tax do I owe?", "TAX_COMPUTATION"),
    ("Paano i-compute ang income tax ko?", "TAX_COMPUTATION"),
    ("Calculate this for me.", "TAX_COMPUTATION"),
    ("Magkano ang babayaran kong buwis sa 2025?", "TAX_COMPUTATION"),
    ("How is income tax computed?", "TAX_INFORMATION"),
    ("Paano i-compute ang income tax?", "TAX_INFORMATION"),
    ("How much is the tax on digital services?", "TAX_INFORMATION"),
    ("Is offshore gaming or POGO now completely banned and illegal in the Philippines?", "TAX_INFORMATION"),
    ("What can you do?", "CHAT"),
    ("Recommend a good adobo recipe", "OUT_OF_SCOPE"),
])
def test_intent_labels(text, label):
    assert intent_router.classify(text).label == label


# ── Extraction ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("800k", Decimal("800000")), ("₱800,000", Decimal("800000")), ("P1.2M", Decimal("1200000")),
    ("2 milyon", Decimal("2000000")), ("850,000.50", Decimal("850000.50")), ("Php 500 thousand", Decimal("500000")),
])
def test_amount_formats(text, expected):
    ex = extract(text)
    assert len(ex.amounts) == 1 and ex.amounts[0].value == expected


def test_numbers_that_are_not_amounts():
    ex = extract("Per RMC 34-2024 and BIR Form 2316, 8% applies for 2025; I have 3 kids; 13th month")
    assert ex.amounts == [] and ex.years == [2025] and ex.regime == "8_percent"


def test_amount_roles():
    ex = extract("taxable income 1,200,000 and my employer withheld 200,000 in 2023")
    roles = {a.role: a.value for a in ex.amounts}
    assert roles == {"taxable_income": Decimal("1200000"), "tax_withheld": Decimal("200000")}


# ── Rules and calculator ─────────────────────────────────────────────────

def _graduated_rule_data():
    with open(os.path.join(ROOT, "tax_rules", "income_tax_graduated.json"), encoding="utf-8") as f:
        return json.load(f)


def test_rule_files_load_and_validate():
    reg = load_registry()
    assert reg.errors == []
    assert set(reg.rules) == {"income_tax_graduated", "income_tax_8_percent", "income_tax_compensation",
                              "employee_contributions"}


def test_validator_catches_the_published_5_million_typo():
    data = _graduated_rule_data()
    data["versions"][0]["brackets"][4]["not_over"] = "5000000"    # as printed in RA 10963
    with pytest.raises(RuleError, match="contiguous|base tax"):
        parse_rule(data)


def test_validator_catches_wrong_base_tax():
    data = _graduated_rule_data()
    data["versions"][1]["brackets"][3]["base_tax"] = "105000"
    with pytest.raises(RuleError, match="base tax"):
        parse_rule(data)


def test_validator_rejects_overlapping_years():
    data = _graduated_rule_data()
    data["versions"][1]["tax_year_from"] = 2022
    with pytest.raises(RuleError, match="overlap"):
        parse_rule(data)


def test_unverified_rule_is_not_used(tmp_path):
    data = _graduated_rule_data()
    data["versions"][1]["verification"]["status"] = "unverified"
    (tmp_path / "g.json").write_text(json.dumps(data), encoding="utf-8")
    reg = load_registry(tmp_path)
    assert reg.lookup("income_tax_graduated", 2025)[2] == "unverified"
    t = dialogue.handle("Compute my income tax for 2025, taxable income 800,000", "english", registry=reg)
    assert t.outcome == "computation_rule_unverified" and "Estimated" not in t.answer


@pytest.mark.parametrize("year,income,expected", [
    (2025, "0", "0.00"), (2025, "250000", "0.00"), (2025, "250001", "0.15"), (2025, "400000", "22500.00"),
    (2025, "800000", "102500.00"), (2025, "2000000", "402500.00"), (2025, "8000000", "2202500.00"),
    (2025, "10000000", "2902500.00"), (2020, "400000", "30000.00"), (2020, "8000000", "2410000.00"),
    (2018, "1000000", "190000.00"),
])
def test_graduated_tax_values(year, income, expected):
    rule, version, status = get_registry().lookup("income_tax_graduated", year)
    assert status == "ok"
    assert str(compute(rule, version, year, {"taxable_income": Decimal(income)}).tax_due) == expected


def test_registry_never_borrows_a_neighbouring_year():
    reg: Registry = get_registry()
    for year in (2017, 2027, 1999):
        assert reg.lookup("income_tax_graduated", year)[2] == "unsupported_year"


# ── Knowledge-base metadata ──────────────────────────────────────────────

def test_ingestion_stamps_document_metadata():
    from app.database import apply_document_metadata
    faq = Document(page_content="x", metadata={"source": "data\\faq\\BIR_Income_Tax_FAQ.pdf"})
    issuance = Document(page_content="x", metadata={"source": "data/2024/RMC No. 1-2024.pdf"})
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        apply_document_metadata(faq, "faq")
        apply_document_metadata(issuance, "2024")
    finally:
        os.chdir(cwd)
    assert faq.metadata["document_type"] == "faq" and faq.metadata["computation_related"] is True
    assert faq.metadata["tax_type"] == "income_tax" and "ph-individual-income-tax-graduated" in faq.metadata["rule_ids"]
    assert issuance.metadata["document_type"] == "issuance" and issuance.metadata["computation_related"] is False
    assert issuance.metadata["jurisdiction"] == "PH"


# ── Employees: computing from gross salary ───────────────────────────────

def test_employee_salary_matches_published_example(world):
    # Cross-check: taxcalculator.com.ph's FAQ works ₱35,000 a month through to
    # ₱33,900 contributions, ₱386,100 taxable, ₱20,415 tax (₱1,701.25 a month)
    # and ₱30,473.75 take-home. Same result from the verified rule files.
    r = ask(world, "My salary is ₱35,000 a month. How much is my income tax for 2025?")
    assert r.outcome == "computation_done" and r.computation.tax_type == "income_tax_compensation"
    assert "₱20,415 a year (about ₱1,701.25 per month)" in r.answer
    assert "SSS 5% × ₱35,000 salary credit = ₱1,750" in r.answer
    assert "PhilHealth 2.5% × ₱35,000 = ₱875" in r.answer
    assert "Pag-IBIG 2% × ₱10,000 = ₱200" in r.answer
    assert "₱420,000 − ₱33,900 = **₱386,100**" in r.answer
    assert "**₱30,473.75**" in r.answer
    assert any("Circular No. 2024-006" in s for s in r.sources)
    assert any("PA2026-0042" in s for s in r.sources)
    assert any(s.startswith("Knowledge base: BIR_Income_Tax_FAQ, p.2") for s in r.sources)
    assert world.rag_calls == 0


def test_employee_monthly_income_is_salary_not_an_error(world):
    r = ask(world, "I earn 50k a month, compute my income tax for 2025")
    assert r.outcome == "computation_done" and r.computation.tax_type == "income_tax_compensation"
    assert r.computation.values["pay_period"] == "monthly"


def test_employee_benefits_above_cap_are_added(world):
    r1, r2 = converse(world, "Magkano ang tax ko? Sahod ko 60k kada buwan para sa 2026", "13th month ko ay 120,000")
    assert r1.language == "filipino" and "₱78,220" in r1.answer
    assert "libre sa buwis ang unang ₱90,000, kaya **₱30,000** ang taxable" in r2.answer
    assert "₱720,000 − ₱41,400 + ₱30,000 = **₱708,600**" in r2.answer
    assert "₱84,220" in r2.answer


def test_employee_year_without_contribution_table_asks_for_contributions(world):
    r1, r2, r3, r4 = converse(world, "compute the tax on my salary for 2024", "50,000", "monthly", "3,500 a month")
    assert "Gross pay" in r1.answer and "taxable income" not in r1.answer.lower()
    assert "monthly**, **semi-monthly" in r2.answer
    assert r3.outcome == "computation_needs_input" and "**2025 to 2026**" in r3.answer
    assert r4.outcome == "computation_done"
    assert "Mandatory contributions you gave (for the year): **₱42,000**" in r4.answer   # 3,500 × 12
    assert "₱54,100" in r4.answer


def test_employee_manual_contributions_override_the_schedule(world):
    *_, r = converse(world, "My salary is 35,000 a month, compute my tax for 2025", "my contributions are 30,000 for the year")
    assert "Mandatory contributions you gave (for the year): **₱30,000**" in r.answer
    assert "₱420,000 − ₱30,000 = **₱390,000**" in r.answer


def test_employee_semi_monthly_and_annual_pay(world):
    r = ask(world, "semi-monthly pay of 25,000, tax for 2025?")
    assert "₱25,000 × 24 = ₱600,000" in r.answer and "₱54,820" in r.answer
    r = ask(world, "My gross salary is 900,000 a year for 2025, compute my tax")
    assert "₱116,025" in r.answer and "×" not in r.answer.split("**Calculation**")[1].split("\n")[1][:30]


def test_employee_daily_pay_is_not_guessed(world):
    r = ask(world, "My salary is 1,000 a day, compute my tax for 2025")
    assert r.outcome == "computation_needs_input" and "daily pay" in r.answer and "Estimated" not in r.answer


def test_minimum_wage_earner_is_not_computed(world):
    r = ask(world, "I'm a minimum wage earner, compute my tax for 2025")
    assert r.outcome == "computation_no_rule" and "Estimated" not in r.answer


def test_taxable_income_still_uses_the_graduated_path(world):
    r = ask(world, "I'm an employee, my taxable income is 800,000 for 2025, compute my tax")
    assert r.computation.tax_type == "income_tax_graduated" and "₱102,500" in r.answer


@pytest.mark.parametrize("monthly,msc,sss,ph,pi", [
    ("5249.99", "5000", "250.00", "250.00", "105.00"),      # below the lowest bracket edge; PhilHealth floor
    ("5250", "5500", "275.00", "250.00", "105.00"),
    ("20000", "20000", "1000.00", "500.00", "200.00"),
    ("34749.99", "34500", "1725.00", "868.75", "200.00"),
    ("34750", "35000", "1750.00", "868.75", "200.00"),
    ("150000", "35000", "1750.00", "2500.00", "200.00"),    # SSS and PhilHealth ceilings
])
def test_contribution_schedule_boundaries(monthly, msc, sss, ph, pi):
    from app.computation.calculator import compute_contributions
    _rule, version, status = get_registry().lookup("employee_contributions", 2025)
    assert status == "ok"
    c = compute_contributions(version, Decimal(monthly))
    assert (str(c.sss_msc), str(c.sss), str(c.philhealth), str(c.pagibig)) == (msc, sss, ph, pi)


def test_no_contribution_schedule_before_2025():
    assert get_registry().lookup("employee_contributions", 2024)[2] == "unsupported_year"


def test_contribution_rule_validation():
    with open(os.path.join(ROOT, "tax_rules", "employee_contributions.json"), encoding="utf-8") as f:
        data = json.load(f)
    data["versions"][0]["sss"]["msc_max"] = "35250"          # not a multiple of the ₱500 step
    with pytest.raises(RuleError, match="multiples"):
        parse_rule(data)


def test_compensation_rule_needs_the_graduated_table(tmp_path):
    for name in ("income_tax_compensation.json", "employee_contributions.json"):
        (tmp_path / name).write_text(open(os.path.join(ROOT, "tax_rules", name), encoding="utf-8").read(), encoding="utf-8")
    reg = load_registry(tmp_path)                             # graduated table missing
    assert "income_tax_compensation" not in reg.rules
    assert any("depends on missing" in e for e in reg.errors)


def test_employee_mode_tted_routing_unchanged():
    test_no_tted_question_is_diverted_from_rag()
