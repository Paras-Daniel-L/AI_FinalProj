"""
Tests for system v1.6: hallucination mitigation, groundedness and relevance.

  answerability   evidence gate, non-answer detector
  grounding       deterministic check (figures, issuance/RA/form numbers,
                  citations) and claim repair
  run_rag         the generate -> check -> verify -> repair loop, with a fake
                  generator and verifier
  api             gate stops generation; a model error shows guidance (HTTP 200)
  routing         "what is X?", capability chat, Taglish compute verbs, OFW, SSS
  integrity       the new BM25 words never come from T-TED; version switches

Offline, no API keys: every model and retrieval call is a fake.
"""

import json
import os

os.environ.setdefault("OPENROUTER_API_KEY", "test")
os.environ.setdefault("JINA_API_KEY", "test")

import pytest  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from app import answerability, grounding, llm, retrieval, version  # noqa: E402
from app import intent as intent_router  # noqa: E402
from app.computation.dialogue import handle  # noqa: E402
from app.llm_client import ChatResult  # noqa: E402
from app.schemas import QueryRequest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONTEXT = "\n\n---\n\n".join([
    "[1] RR No. 8-2024, p.2 (2024)\nMicro Taxpayers are those whose gross sales for a taxable year are less than "
    "Three Million Pesos (P3,000,000.00).",
    "[2] RMC No. 37-2024, p.1 (2024)\nTaxpayers who forgot their TIN may send an inquiry to tin.inquiry@bir.gov.ph "
    "within thirty (30) days of the request.",
    "[3] RR No. 16-2023, p.3 (2023)\nThe withholding tax rate is one percent (1%) of one-half of the gross remittances.",
])


TOPICS = ["micro taxpayers sell below three million", "withholding agents remit by the tenth day",
          "digital services are subject to value-added tax", "books of accounts are registered online",
          "penalties apply to late filing of returns"]


def doc(score, n=1, issuance=None, dist=None):
    meta = {"id": f"c{n}", "source": f"data/2024/RMC No. {n}-2024.pdf", "page": 0, "year": "2024",
            "_rerank_score": score}
    if issuance:
        meta["issuance_id"] = issuance
    if dist is not None:
        meta["_semantic_distance"] = dist
    return Document(page_content=TOPICS[(n - 1) % len(TOPICS)], metadata=meta)


# ── Evidence gate ────────────────────────────────────────────────────────

def test_gate_stops_weak_evidence_and_passes_strong_evidence():
    assert answerability.passes_gate([doc(0.09), doc(0.05, 2)], threshold=0.14) == (False, 0.09)
    assert answerability.passes_gate([doc(0.31)], threshold=0.14) == (True, 0.31)


def test_gate_is_open_without_rerank_scores_or_when_off():
    plain = Document(page_content="x", metadata={"id": "a"})
    assert answerability.passes_gate([plain], threshold=0.14) == (True, None)   # reranker failed: no judgement
    assert answerability.passes_gate([doc(0.01)], threshold=0)[0] is True        # EVIDENCE_GATE_MIN_SCORE=0


# ── Non-answer detector ──────────────────────────────────────────────────

@pytest.mark.parametrize("draft", [
    "Hindi direktang sinasagot ng mga excerpt ang tanong tungkol sa penalties. Ang [1] ay tungkol sa...",
    "Hindi saklaw ng excerpts ang buwis sa Netflix subscription.",
    "Hindi masasagot nang buo: ang mga excerpt ay may rate table [1].",
    "Hindi sapat ang mga excerpt na ito para sagutin nang tiyak ang tanong.",
    "**Hindi tinutukoy ng mga excerpt** ang partikular na \"Roblox.\" Ang saklaw ng [1] ay...",
    "The excerpts do not mention the penalty for late filing. They describe [1]...",
    "I can't answer this from the documents provided.",
])
def test_non_answers_are_detected(draft):
    assert answerability.is_non_answer(draft)


@pytest.mark.parametrize("draft", [
    "Hindi. Ang RMC No. 37-2024 ay para sa TIN inquiry [2].",                  # a real "No" answer
    "Hindi na kailangang mag-update ng COR [1].",
    "Oo, may threshold na ₱500,000.00 [3]. Hindi sakop ng mga dokumento ang petsa ng bisa.",  # partial: answer first
    "Yes. Online sellers must display the badge [1].",
    "Ang Micro Taxpayer ay may gross sales na mas mababa sa ₱3,000,000.00 [1].",
])
def test_real_answers_are_not_flagged(draft):
    assert not answerability.is_non_answer(draft)


# ── Deterministic grounding check ───────────────────────────────────────

def test_figures_match_by_value_and_written_numbers_count():
    ok = ("Micro Taxpayers have gross sales below ₱3,000,000 [1]. "
          "An inquiry is answered within 30 days at tin.inquiry@bir.gov.ph [2]. The rate is 1% [3].")
    rep = grounding.check(ok, CONTEXT)
    assert rep.passed, rep.issues
    assert rep.n_tokens >= 4
    assert grounding.check("Micro means below ₱3 million [1].", CONTEXT).passed


def test_a_figure_or_issuance_in_no_excerpt_is_flagged():
    rep = grounding.check("Yes [1]. This is under RA No. 12023 and RMC No. 99-2025 [1].", CONTEXT)
    assert not rep.passed
    assert any("RA No. 12023" in i for i in rep.issues) and any("RMC-99-2025" in i for i in rep.issues)


def test_a_figure_cited_to_the_wrong_excerpt_is_flagged():
    rep = grounding.check("The rate is 1% [1].", CONTEXT)
    assert not rep.passed and "excerpt [3]" in rep.issues[0]


def test_nonexistent_citation_and_question_figures():
    assert "does not exist" in grounding.check("It is less than ₱3,000,000 [7].", CONTEXT).issues[0]
    # A figure the user gave may be repeated without support.
    q = "My gross pay is ₱360,000.00, am I a micro taxpayer?"
    assert grounding.check("Yes. With ₱360,000.00 you are under ₱3,000,000 [1].", CONTEXT, q).passed


def test_citation_on_a_later_sentence_covers_the_line():
    assert grounding.check("Micro Taxpayers sell less than ₱3,000,000. This is per RR 8-2024 [1].", CONTEXT).passed


def test_repair_drops_flagged_sentences_but_never_the_opening_answer():
    draft = "Yes [1]. Micro Taxpayers sell less than ₱3,000,000 [1]. It is under RA No. 12023 [1]."
    rep = grounding.check(draft, CONTEXT)
    assert [f.index for f in rep.flagged] == [2]
    fixed = grounding.repair(draft, {2})
    assert fixed == "Yes [1]. Micro Taxpayers sell less than ₱3,000,000 [1]." and grounding.check(fixed, CONTEXT).passed
    # "Yes." is short, so sentence 1 is part of the opening answer too.
    assert grounding.repair(draft, {1}) is None
    assert grounding.repair("Micro is below ₱9,000 [1]. Also [1].", {0}) is None


def test_repair_keeps_bullets_and_needs_a_citation_left():
    draft = "The conditions are [1]:\n- gross sales below ₱3,000,000 [1]\n- registered under RA 99999 [1]"
    assert grounding.repair(draft, {2}) == "The conditions are [1]:\n- gross sales below ₱3,000,000 [1]"
    assert grounding.repair("Oo, kasama ang online sellers [1]. Wala nang iba.", {1}) == "Oo, kasama ang online sellers [1]."
    assert grounding.repair("Oo, kasama [1]. Ayon ito sa RR 8-2024 [1].", {0}) is None   # the answer itself


def test_verifier_issues_map_to_sentences():
    draft = "Yes [1]. Micro is below ₱3,000,000 [1]. The penalty is ₱25,000 for each violation [1]."
    assert grounding.map_issues_to_sentences(draft, ["The penalty is ₱25,000 for each violation"]) == {2}
    assert grounding.map_issues_to_sentences(draft, ["something entirely unrelated here"]) is None


# ── The generate -> check -> verify -> repair loop ───────────────────────

class Models:
    """Fake llm_client.chat: queued generator drafts and verifier replies."""

    def __init__(self, drafts, verdicts):
        self.drafts, self.verdicts, self.calls = list(drafts), list(verdicts), []

    def __call__(self, system, user, *, model, temperature, max_tokens, provider_order=None, role="llm",
                 thinking_off=None):
        self.calls.append((role, user))
        text = (self.drafts if role == "generator" else self.verdicts).pop(0)
        return ChatResult(text=text, finish_reason="stop", model=model)

    def roles(self):
        return [r for r, _u in self.calls]


SUPPORTED = '{"verdict": "SUPPORTED", "issues": [], "answers_question": true, "relevance_issue": ""}'


def rag(monkeypatch, drafts, verdicts, question="Who is a micro taxpayer?"):
    from app import llm_client
    m = Models(drafts, verdicts)
    monkeypatch.setattr(llm_client, "chat", m)
    return llm.run_rag(question, [], CONTEXT), m


def test_non_answer_draft_becomes_generator_no_answer_without_a_verifier_call(monkeypatch):
    result, m = rag(monkeypatch, ["Hindi direktang sinasagot ng mga excerpt ang tanong. Ang [1] ay tungkol sa..."], [])
    assert result.outcome == "generator_no_answer" and m.roles() == ["generator"]
    assert result.attempts[0].status == "non_answer"


def test_grounding_failure_in_a_detail_is_repaired_then_verified(monkeypatch):
    draft = "Micro Taxpayers sell less than ₱3,000,000 [1]. This comes from RA No. 12023 [1]."
    result, m = rag(monkeypatch, [draft], [SUPPORTED])
    assert result.outcome == "verified"
    assert result.answer == "Micro Taxpayers sell less than ₱3,000,000 [1]."
    assert m.roles() == ["generator", "verifier"]
    assert "RA No. 12023" not in m.calls[1][1]                       # the verifier saw the repaired draft
    assert result.attempts[0].status == "repaired" and result.attempts[0].repair["source"] == "grounding_check"


def test_grounding_failure_in_the_opening_answer_is_regenerated_without_a_verifier_call(monkeypatch):
    bad = "Micro Taxpayers sell less than ₱5,000,000 [1]."
    good = "Micro Taxpayers sell less than ₱3,000,000 [1]."
    result, m = rag(monkeypatch, [bad, good], [SUPPORTED])
    assert result.outcome == "verified" and result.answer == good
    assert m.roles() == ["generator", "generator", "verifier"]
    assert result.attempts[0].status == "grounding_failed"
    assert "₱5,000,000" in m.calls[1][1]                            # the issue went into the audit notice


def test_verifier_rejection_of_one_detail_is_repaired_with_one_more_verifier_call(monkeypatch):
    draft = "Yes [1]. Micro Taxpayers sell less than ₱3,000,000 [1]. They also get a special lane at every office [1]."
    rejected = '{"verdict": "UNSUPPORTED", "issues": ["They also get a special lane at every office"], "answers_question": true}'
    result, m = rag(monkeypatch, [draft], [rejected, SUPPORTED])
    assert result.outcome == "verified"
    assert result.answer == "Yes [1]. Micro Taxpayers sell less than ₱3,000,000 [1]."
    assert m.roles() == ["generator", "verifier", "verifier"]
    assert result.attempts[0].repair["source"] == "verifier" and result.attempts[0].status == "repaired"
    assert len(result.calls) == 3


def test_off_topic_draft_is_retried_and_fails_closed(monkeypatch):
    draft = "You can send a TIN inquiry to tin.inquiry@bir.gov.ph [2]."
    off = '{"verdict": "SUPPORTED", "issues": [], "answers_question": false, "relevance_issue": "answers TIN recovery, not a first TIN"}'
    result, m = rag(monkeypatch, [draft] * 3, [off] * 3, question="How do I get my first TIN?")
    assert result.outcome == "verification_failed"
    assert [a.status for a in result.attempts] == ["off_topic"] * 3
    assert "did not answer the question" in m.calls[2][1]            # 2nd generator call got the relevance notice


def test_relevance_field_missing_counts_as_answering():
    v = llm.parse_verdict('{"verdict": "SUPPORTED", "issues": []}', relevance=True)
    assert v.passed
    v = llm.parse_verdict('{"verdict": "SUPPORTED", "issues": [], "answers_question": "false"}', relevance=True)
    assert not v.passed and v.kind == llm.VERDICT_OFF_TOPIC
    v = llm.parse_verdict('{"verdict": "SUPPORTED", "issues": [], "answers_question": false}', relevance=False)
    assert v.passed                                                 # VERIFIER_RELEVANCE=0: v1 behavior


def test_switches_off_restore_the_old_loop(monkeypatch):
    monkeypatch.setattr(answerability, "NON_ANSWER_DETECTION", False)
    monkeypatch.setattr(grounding, "GROUNDING_CHECK", False)
    draft = "Hindi saklaw ng excerpts ang Netflix. Ang [1] ay tungkol sa micro taxpayers na mas mababa sa ₱3,000,000."
    result, m = rag(monkeypatch, [draft], [SUPPORTED])
    assert result.outcome == "verified" and m.roles() == ["generator", "verifier"]


# ── API: gate and errors ─────────────────────────────────────────────────

class FakeChroma:
    def __init__(self, *a, **kw):
        pass

    def get(self, include=None, where=None, limit=None):
        return {"ids": ["c1", "c2"], "metadatas": []}


@pytest.fixture
def api(monkeypatch):
    from app import api as api_module
    monkeypatch.setattr(api_module.trace_log, "write", lambda tr: None)
    monkeypatch.setattr(api_module.answer_cache, "CACHE_ENABLED", False)
    monkeypatch.setattr(api_module, "get_embedding_function", lambda: None)
    monkeypatch.setattr(api_module, "_warn_if_stale_index", lambda: None)
    monkeypatch.setattr(api_module, "Chroma", FakeChroma)
    return api_module


def test_gate_answers_with_guidance_and_never_calls_a_model(api, monkeypatch):
    calls = []
    monkeypatch.setattr(api, "retrieve_docs", lambda q, y, db: [doc(0.08)])
    monkeypatch.setattr(api, "run_rag", lambda *a, **k: calls.append(1))
    response, trace = api.run_query(QueryRequest(query="Magkano ang tax sa Netflix subscription?"))
    assert response.outcome == "no_retrieval" and response.mode == "guidance" and not calls
    assert trace["evidence_gate"] == {"top_score": 0.08, "threshold": answerability.EVIDENCE_GATE_MIN_SCORE,
                                      "passed": False}


def test_model_error_returns_guidance_not_a_server_error(api, monkeypatch):
    from fastapi.testclient import TestClient

    from app.llm import RagResult
    from main import app as fastapi_app
    monkeypatch.setattr(api, "retrieve_docs", lambda q, y, db: [doc(0.9)])
    monkeypatch.setattr(api, "run_rag", lambda q, h, c, language=None: RagResult("error", "x", language.label,
                                                                                 error="HTTP 401: User not found"))
    with pytest.raises(api.ModelCallFailed):                          # the evaluation still sees an exception
        api.run_query(QueryRequest(query="Who is a micro taxpayer under RR 8-2024?"))
    from app import guard
    monkeypatch.setattr(guard, "RATE_LIMIT_PER_MIN", 1000)
    r = TestClient(fastapi_app).post("/query", json={"query": "Who is a micro taxpayer under RR 8-2024?"})
    assert r.status_code == 200
    body = r.json()
    assert body["outcome"] == "error" and "try again" in body["answer"].lower() and body["sources"] == []


# ── Retrieval: conditional floor ─────────────────────────────────────────

def test_floor_adds_only_dense_hits_the_reranker_also_finds_relevant():
    kept = [doc(0.8, 1, issuance="RR-8-2024", dist=0.3)]
    weak = doc(0.05, 2, dist=0.1)                       # close embedding, but the reranker says unrelated
    related = doc(0.5, 3, dist=0.2)                     # >= 0.5 x 0.8
    same_issuance = doc(0.1, 4, issuance="RR-8-2024", dist=0.25)
    out = retrieval.add_dense_floor(kept, [weak, related, same_issuance], k=4, max_docs=6, relative=0.5)
    assert [d.metadata["id"] for d in out] == ["c1", "c3", "c4"]
    out = retrieval.add_dense_floor(kept, [weak, related, same_issuance], k=4, max_docs=6, relative=0)
    assert [d.metadata["id"] for d in out] == ["c1", "c2", "c3", "c4"]   # 0 = the v1.3 floor


# ── Routing fixes (log review 2026-10-07) ────────────────────────────────

@pytest.mark.parametrize("q", ["what is eFPS?", "ano ang efps", "ano ang TIN?", "what is RMC?", "what is income?",
                               "What is Gross Income?"])
def test_definition_questions_about_a_specific_term_go_to_rag(q):
    i = intent_router.classify(q)
    assert i.label == intent_router.TAX_INFORMATION and not i.vague


@pytest.mark.parametrize("q", ["VAT", "tax?", "deadline?", "income tax", "What about yung tax?"])
def test_bare_topics_are_still_vague(q):
    assert intent_router.classify(q).vague


def test_capability_question_with_extra_text_is_chat():
    assert intent_router.classify("hello sino ka ba ano ba kaya mong gawin pwede mo ba ako suntukin?").label == "CHAT"
    assert intent_router.classify("malungkot ako ngayon pano mo ako matutulungan").label == "CHAT"
    assert intent_router.classify("sino ka at ano ang VAT?").label == "TAX_INFORMATION"   # a tax question stays RAG


@pytest.mark.parametrize("q", ["kaya mo bang magcompute ng tax?", "kaya mo bang computein yung tax para sa akin?",
                               "maari mo ba akong tulungan magcompute ng tax ko?",
                               "may gross salary akong 20,000 gaano kalaki ang aking tax?"])
def test_taglish_compute_requests_reach_the_calculator(q):
    assert intent_router.classify(q).label == intent_router.TAX_COMPUTATION


def test_ofw_pay_is_not_computed_as_resident_compensation():
    turn = handle("Ako ay OFW sa Dubai, sahod ko ay 25,000 kada buwan. Magkano ang tax ko para sa 2025?", "filipino")
    assert turn.outcome == "computation_no_rule" and "OFW" in turn.answer and "₱3,765" not in turn.answer


def test_sss_question_says_it_is_a_contribution_and_shows_the_shares():
    first = handle("magkano ba yung tax ko sa sss?", "filipino")
    assert first.answer.startswith("Ang SSS, PhilHealth at Pag-IBIG ay mga kontribusyon")
    assert first.state.values.get("kind") == "employee"
    done = handle("2025, 20,000 monthly", "filipino", first.state)
    assert done.outcome == "computation_done" and "SSS 5%" in done.answer
    assert "hindi buwis ng BIR" in done.answer


# ── Integrity ────────────────────────────────────────────────────────────

def test_v16_glossary_words_never_come_from_tted():
    import re
    ted = json.load(open(os.path.join(ROOT, "rag_eval", "ted_dataset.json"), encoding="utf-8"))
    words = {w for r in ted for w in re.findall(r"[\w-]+", r["query"].lower())}
    leaked = sorted(set(retrieval.FILIPINO_TAX_GLOSSARY_V16) & words)
    assert not leaked, f"T-TED words in the v1.6 glossary: {leaked}"


def test_dev_set_has_no_tted_lookalikes():
    from tools import dev_tune
    ted = dev_tune._ted_queries()
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "dev", "dev_questions.jsonl"), encoding="utf-8") if l.strip()]
    assert rows and not [r["id"] for r in rows if dev_tune.looks_like_ted(r["query"], ted)]


def test_version_labels_follow_the_switches(monkeypatch):
    assert version.SYSTEM_VERSION == "1.6.0" and version.effective_label() == "v1.6"
    monkeypatch.setattr(version, "current_settings",
                        lambda: {**version.V16_OFF, "RAG_PROMPT_VERSION": "3", "DENSE_FLOOR_K": "4"})
    assert version.effective_label() == "v1.5.2"                    # every v1.6 switch off
    monkeypatch.setattr(version, "current_settings",
                        lambda: {**version.V16_ON, "RAG_PROMPT_VERSION": "4", "DENSE_FLOOR_K": "4",
                                 "CLAIM_REPAIR": "0"})
    assert version.effective_label() == "v1.6 (custom settings)"
