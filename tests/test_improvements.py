"""
Tests for the retrieval / chunking / evaluation improvements. No network:
the reranker, the embeddings and the judge are replaced with fakes.

Run from the project root:  python -m pytest tests -q
"""

import os

os.environ.setdefault("OPENROUTER_API_KEY", "test")
os.environ.setdefault("JINA_API_KEY", "test")
os.environ.setdefault("JUDGE_MODEL", "test/judge")

import pytest  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from app import prompts, retrieval, textproc  # noqa: E402
from app import rerank as reranker  # noqa: E402


# ── textproc ──────────────────────────────────────────────────────────────

def test_clean_pdf_text_fixes_hyphens_and_wraps():
    raw = ("The RES shall not claim any input tax on the pass -\nthrough charges of the\n"
           "Value -Added Tax under RMC No. 8 -2026.\n \n(a) first item;\n(b) second item")
    out = textproc.clean_pdf_text(raw)
    assert "pass-through charges of the Value-Added Tax under RMC No. 8-2026." in out
    assert "\n(a) first item;" in out and "\n(b) second item" in out
    assert "\n\n" in out  # paragraph break kept


def test_clean_pdf_text_keeps_dashes_and_numbers():
    out = textproc.clean_pdf_text("Rate - 12% of ₱1,500.00 for 2024-2025")
    assert out == "Rate - 12% of ₱1,500.00 for 2024-2025"


def test_repair_split_words_uses_corpus_evidence():
    vocab = textproc.build_vocab(["shall"] * 50 + ["creditable"] * 10 + ["in to into"] * 3 + ["in"] * 100 + ["to"] * 100)
    assert textproc.repair_split_words("It shal l be creditab le", vocab) == "It shall be creditable"
    assert textproc.repair_split_words("put in to effect", vocab) == "put in to effect"
    assert textproc.repair_split_words("of the rule", vocab) == "of the rule"


def test_split_sentences_respects_abbreviations_and_semicolons():
    text = ("Under Sec. 4 of RR No. 3-2024, the seller may claim it. The claim is optional; "
            "it is not automatic. R.A. No. 11976 applies.")
    s = textproc.split_sentences(text, min_chars=0)
    assert s == ["Under Sec. 4 of RR No. 3-2024, the seller may claim it.", "The claim is optional;",
                 "it is not automatic.", "R.A. No. 11976 applies."]
    # with the default minimum, short fragments merge into a neighbour
    assert all(len(x) >= 25 for x in textproc.split_sentences(text))


def test_chunk_sentences_never_cuts_mid_sentence():
    sents = [f"Sentence number {i} says something about withholding tax rules." for i in range(40)]
    chunks = textproc.chunk_sentences(sents, max_chars=300, overlap=1)
    assert all(len(c) <= 300 for c in chunks)
    assert all(c.endswith(".") for c in chunks)
    # 1-sentence overlap
    assert chunks[1].split(". ")[0] + "." in chunks[0]


def test_chunk_sentences_hard_splits_monster_sentence():
    chunks = textproc.chunk_sentences(["word, " * 300], max_chars=200)
    assert all(len(c) <= 200 for c in chunks)


@pytest.mark.parametrize("text,expected", [
    ("RMC No. 8-2026", ["RMC-8-2026"]),
    ("Revenue Regulations (RR) No. 7-2021 and RR 14-2022", ["RR-7-2021", "RR-14-2022"]),
    ("3089RMR 2-2002", ["RMR-2-2002"]),
    ("RR No. 014 - 2022", ["RR-14-2022"]),
    ("Revenue Memorandum Circular No. 5-2024", ["RMC-5-2024"]),
    ("sa ilalim ng RR 8-2024?", ["RR-8-2024"]),
    ("RA No. 11976 for 2024-2025", []),
    ("RMC No. 49-2023v3", ["RMC-49-2023"]),
])
def test_parse_issuance_ids(text, expected):
    assert textproc.parse_issuance_ids(text) == expected


def test_document_summary_prefers_subject_line():
    page = "REPUBLIC OF THE PHILIPPINES\nSUBJECT : Clarification on Vapor Products\nTO : All concerned"
    assert textproc.document_summary(page) == "Clarification on Vapor Products"


# ── ingestion ─────────────────────────────────────────────────────────────

def test_sentence_chunking_stamps_issuance_metadata():
    from app import database
    pages = [
        Document(page_content="SUBJECT : Updated list of registered manufacturers\nTO : All\n\n"
                              "This Circular publishes the list. As required under RR No. 7 -2021 the list is updated.",
                 metadata={"source": "data/2024/RMC No. 46-2024.pdf", "page": 0, "year": "2024"}),
        Document(page_content="Page two text about the same list of manufacturers.",
                 metadata={"source": "data/2024/RMC No. 46-2024.pdf", "page": 1, "year": "2024"}),
    ]
    chunks = database.split_documents_by_sentence(pages)
    assert chunks, "no chunks produced"
    for c in chunks:
        assert c.metadata["issuance_id"] == "RMC-46-2024"
        assert c.metadata["title"] == "RMC No. 46-2024"
        assert c.metadata["context_header"].startswith("RMC No. 46-2024 — Updated list")
    assert "RR-7-2021" in chunks[0].metadata["issuance_ids"]
    assert "RR No. 7-2021" in chunks[0].page_content  # broken hyphen repaired
    assert database.embedding_text(chunks[1]).startswith("RMC No. 46-2024 — ")
    assert "RMC No. 46-2024" not in chunks[1].page_content  # header not stored as evidence


# ── retrieval ─────────────────────────────────────────────────────────────

def _doc(text, score=None, issuance="", cid=None):
    meta = {"id": cid or text[:20], "issuance_id": issuance, "source": f"data/x/{issuance or 'X'}.pdf"}
    if score is not None:
        meta["_rerank_score"] = score
    return Document(page_content=text, metadata=meta)


def test_dynamic_cut_thresholds_and_bounds():
    scored = [_doc("alpha beta gamma", 0.9), _doc("delta epsilon", 0.5), _doc("zeta eta", 0.3),
              _doc("theta iota", 0.05)]
    kept = retrieval.dynamic_cut(scored, min_score=0.15, relative=0.4, min_docs=1, max_docs=3)
    assert [d.page_content for d in kept] == ["alpha beta gamma", "delta epsilon"]  # 0.3 < 0.4*0.9
    # min_docs keeps the best even when nothing clears the bar
    low = [_doc("a b c", 0.01), _doc("d e f", 0.009)]
    assert len(retrieval.dynamic_cut(low, min_score=0.15, relative=0.4, min_docs=1, max_docs=3)) == 1


def test_dynamic_cut_drops_near_duplicates():
    text = "the seller shall issue an invoice for every sale of goods"
    scored = [_doc(text, 0.9, cid="a"), _doc(text + " promptly", 0.85, cid="b"), _doc("other topic entirely here", 0.8, cid="c")]
    kept = retrieval.dynamic_cut(scored, min_score=0.1, relative=0.1, min_docs=1, max_docs=2)
    assert [d.metadata["id"] for d in kept] == ["a", "c"]


def test_rerank_and_cut_boosts_named_issuance(monkeypatch):
    cands = [_doc("general text on tax stamps", issuance="RMC-1-2024", cid="1"),
             _doc("vapor products stamps rule", issuance="RR-14-2022", cid="2")]
    monkeypatch.setattr(reranker, "rerank", lambda q, docs: [0.50, 0.40])
    kept = retrieval.rerank_and_cut("ayon sa RR 14-2022", cands, ["RR-14-2022"])
    assert kept[0].metadata["id"] == "2" and kept[0].metadata["_id_match"] is True
    assert kept[0].metadata["_rerank_score"] == pytest.approx(0.40 + retrieval.ID_MATCH_BOOST)


def test_compress_evidence_keeps_relevant_sentences_and_list_leadin(monkeypatch):
    doc = _doc("The platform shall not withhold if: (a) gross remittances have not exceeded ₱500,000; "
               "(b) the seller submits a sworn declaration. Penalties are covered elsewhere in the Code. "
               "The Circular takes effect immediately.", issuance="RMC-8-2024")
    sents = textproc.split_sentences(doc.page_content)

    def fake(q, docs):
        return [0.9 if "₱500,000" in s else 0.01 for s in docs]
    monkeypatch.setattr(reranker, "rerank", fake)
    monkeypatch.setattr(retrieval, "SENTENCE_MIN_KEEP", 1)
    out = retrieval.compress_evidence("threshold?", [doc])
    assert len(out) == 1
    text = out[0].page_content
    assert "₱500,000" in text
    assert "shall not withhold if:" in text            # lead-in pulled in
    assert "takes effect immediately" not in text
    assert out[0].metadata["_full_text"] == doc.page_content
    assert len(sents) >= 3


def test_compress_evidence_fails_open(monkeypatch):
    def boom(q, docs):
        raise reranker.RerankError("down")
    monkeypatch.setattr(reranker, "rerank", boom)
    d = [_doc("Some sentence that is long enough to count. Another sentence that is long enough.")]
    assert retrieval.compress_evidence("q", d) == d


def test_query_tokenizer_drops_function_words_and_expands_glossary():
    toks = retrieval.tokenize("Magkano ang buwis sa RR 8-2024?", query=True)
    assert "ang" not in toks and "sa" not in toks
    assert "tax" in toks and "8-2024" in toks and "2024" in toks


def test_bm25_text_includes_header():
    d = Document(page_content="page five text", metadata={"context_header": "RMC No. 19-2022 — Tax-free exchange"})
    assert retrieval._bm25_text(d).startswith("RMC No. 19-2022")


# ── prompts ───────────────────────────────────────────────────────────────

def test_rag_prompt_formats_and_has_no_test_questions():
    out = prompts.RAG_PROMPT.format(user_language="Filipino", no_answer_sentinel="NO_ANSWER",
                                    context="[1] X\ntext", history="", audit_notice="", question="Q?")
    assert "NO_ANSWER" in out and "USER QUESTION: Q?" in out
    import json
    from pathlib import Path
    ted = json.loads((Path(__file__).parent.parent / "rag_eval" / "ted_dataset.json").read_text(encoding="utf-8"))
    for row in ted:
        assert row["query"][:40] not in prompts.RAG_PROMPT, f"T-TED question {row['qid']} leaked into the prompt"


# ── evaluation ────────────────────────────────────────────────────────────

def test_context_sentences_skip_labels_and_split_consistently():
    from rag_eval import judge
    ctx = ("[1] RMC No. 116-2024 Digest, p.1 (2024)\nGross sales shall exclude VAT. The DUs shall issue\n"
           "an invoice to customers.\n\n---\n\n[2] RR 16-2023, p.1 (2023)\nWithholding applies to platforms.")
    sents = judge.context_sentences(ctx)
    assert sents == ["Gross sales shall exclude VAT.", "The DUs shall issue an invoice to customers.",
                     "Withholding applies to platforms."]


def test_context_relevance_scores_from_numbers(monkeypatch):
    from rag_eval import judge
    monkeypatch.setattr(judge, "_chat", lambda prompt, system, role: '{"relevant": [1, "3", 99]}')
    ctx = "[1] A\nFirst sentence is long enough here. Second sentence is long enough too. Third sentence is also long."
    r = judge.judge_context_relevance("q", ctx)
    assert r.error is None
    assert [s.relevant for s in r.sentences] == [True, False, True]
    assert r.score == pytest.approx(2 / 3)


def test_context_relevance_unparseable_is_error(monkeypatch):
    from rag_eval import judge
    monkeypatch.setattr(judge, "_chat", lambda prompt, system, role: "sorry")
    r = judge.judge_context_relevance("q", "[1] A\nA sentence that is long enough to count.")
    assert r.score is None and r.error


def test_effect_size_not_labeled_with_few_nonzero_pairs():
    from rag_eval.stats import paired_comparison
    r = paired_comparison([1.0] * 18, [1.0] * 16 + [0.8, 0.9])
    assert r.effect_size_label is None and "non-tied" in r.note


def test_retrieval_eval_gold_matching():
    from rag_eval import retrieval_eval as re_
    assert re_.is_gold({"source": "data/2024/RMC No. 116-2024 Digest.pdf"}, ["RMC-116-2024"])
    assert re_.is_gold({"issuance_id": "RR-14-2022"}, ["RR-14-2022"])
    assert not re_.is_gold({"source": "data/2024/RMC No. 46-2024.pdf"}, ["RMC-8-2025"])


# ── Jina rate limiting (100 RPM / 100k TPM free tier) ─────────────────────

class _FakeClock:
    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def test_limiter_paces_by_tokens_per_minute():
    from app import jina_http
    clock = _FakeClock()
    lim = jina_http.RateLimiter(rpm=100, tpm=100_000, safety=0.9, clock=clock, sleep=clock.sleep)
    for _ in range(4):
        lim.acquire(20_000)            # 80k of the 90k budget, no waiting
    assert clock.slept == []
    lim.acquire(20_000)                # would make 100k > 90k -> waits for the window
    assert clock.t - 1000.0 >= 59.9


def test_limiter_paces_by_requests_per_minute():
    from app import jina_http
    clock = _FakeClock()
    lim = jina_http.RateLimiter(rpm=10, tpm=10**9, safety=1.0, clock=clock, sleep=clock.sleep)
    for _ in range(10):
        lim.acquire(1)
    assert clock.slept == []
    lim.acquire(1)
    assert clock.t - 1000.0 >= 59.9


def test_limiter_settle_uses_reported_tokens():
    from app import jina_http
    clock = _FakeClock()
    lim = jina_http.RateLimiter(rpm=100, tpm=100_000, safety=1.0, clock=clock, sleep=clock.sleep)
    h = lim.acquire(90_000)            # estimate was high...
    lim.settle(h, 10_000)              # ...Jina reported 10k
    lim.acquire(80_000)                # fits now without waiting
    assert clock.slept == []


def test_limiter_max_wait_raises_instead_of_blocking():
    from app import jina_http
    clock = _FakeClock()
    lim = jina_http.RateLimiter(rpm=1, tpm=10**9, safety=1.0, clock=clock, sleep=clock.sleep)
    lim.acquire(1)
    with pytest.raises(jina_http.JinaRetryError):
        lim.acquire(1, max_wait=10)


class _Resp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body or {}, headers or {}

    def json(self):
        return self._body


def test_post_retries_429_with_retry_after(monkeypatch):
    from app import jina_http
    replies = [_Resp(429, headers={"Retry-After": "3"}),
               _Resp(200, {"data": [], "usage": {"total_tokens": 5}})]
    waits = []
    monkeypatch.setattr(jina_http.time, "sleep", lambda s: waits.append(s))
    clock = _FakeClock()
    lim = jina_http.RateLimiter(100, 100_000, clock=clock, sleep=clock.sleep)
    resp = jina_http.post("u", {"input": ["x"]}, None, limiter=lim,
                          session=type("S", (), {"post": lambda self, *a, **k: replies.pop(0)})())
    assert resp.status_code == 200 and waits == [3.0]


def test_rate_limit_estimates():
    from app import jina_http
    assert jina_http.estimate_payload_tokens({"input": ["a" * 300, "b" * 30]}) == 110
    # rerank: the query is billed once per document
    assert jina_http.estimate_payload_tokens({"query": "q" * 30, "documents": ["d" * 60] * 3}) == 3 * (10 + 20)


def test_embedding_batches_are_token_bounded(monkeypatch):
    from app import embeddings
    sent = []
    emb = embeddings.JinaTaskEmbeddings(api_key="k")
    monkeypatch.setattr(emb, "_post", lambda batch, task: sent.append(len(batch)) or [[0.0]] * len(batch))
    monkeypatch.setattr(embeddings, "MAX_TOKENS_PER_REQUEST", 1000)
    out = emb.embed_documents(["x" * 900] * 10)       # 300 tokens each -> 3 per request
    assert len(out) == 10 and max(sent) <= 3


def test_index_config_written_before_embedding(monkeypatch, tmp_path):
    """A rate-limit crash mid-way must leave a resumable index (config present)."""
    from app import database
    order = []
    monkeypatch.setattr(database, "CHROMA_PATH", str(tmp_path))
    monkeypatch.setattr(database, "write_index_config", lambda path: order.append("config"))
    monkeypatch.setattr(database, "_add_with_context_headers",
                        lambda db, chunks: order.append("embed"))
    monkeypatch.setattr(database, "clear_cache", lambda: 0)

    class _DB:
        def __init__(self, **kw):
            pass

        def get(self, include=None):
            return {"ids": []}
    monkeypatch.setattr(database, "Chroma", _DB)
    monkeypatch.setattr(database, "get_embedding_function", lambda: None)
    chunk = Document(page_content="text", metadata={"source": "a.pdf", "page": 0, "year": "2024",
                                                    "context_header": "A"})
    database.add_to_chroma([chunk])
    assert order[:2] == ["config", "embed"]
