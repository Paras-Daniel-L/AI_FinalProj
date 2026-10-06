"""
Tests for the live demo page (/demo): the process events, the three demo
metrics, the Sagot AI vs REVIE comparison (v1.5) and the streaming run endpoint.

No network: Chroma, BM25, the reranker, every model call (generator,
verifier, judge) and the Jina matching embeddings are replaced
with fakes. The REAL pipeline code (app/api.py run_query, retrieval fusion
and reranking cut, app/llm.py run_rag) runs, so the events tested here are
the ones the panel will see.

Run from the project root:  python -m pytest tests -q
"""

import json
import os

os.environ.setdefault("OPENROUTER_API_KEY", "test")
os.environ.setdefault("JINA_API_KEY", "test")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from app import progress  # noqa: E402
from app.schemas import QueryRequest  # noqa: E402

ANSWER = "Yes. Online sellers must display the BIR Registration Seal Badge [1]."


def _doc(n, **meta):
    texts = {1: "Online sellers must display the BIR Registration Seal Badge on their shop page.",
             2: "The badge replaces posting a printed Certificate of Registration in a physical store.",
             3: "Revenue District Offices will monitor compliance through regular online checks."}
    return Document(page_content=texts[n], metadata={
        "id": f"2026:RMC No. 38-2026.pdf:{n}:0", "title": "RMC No. 38-2026", "year": "2026", "page": n,
        "source": "data/2026/RMC No. 38-2026.pdf", **meta})


class FakeChroma:
    def __init__(self, *a, **kw):
        pass

    def get(self, include=None, where=None, limit=None):
        return {"ids": ["c1", "c2", "c3"]}


class FakeBM25:
    def search(self, query, k, year=None):
        return [_doc(2, _bm25_score=7.1), _doc(3, _bm25_score=5.0)]


def fake_semantic(db, query, k, year_filter):
    return [_doc(n, _semantic_distance=0.2 + n / 10) for n in range(1, min(k, 3) + 1)]


class FakeLLM:
    """Answers every model call by role, and records which roles were called."""

    def __init__(self):
        self.roles = []
        self.verdicts = ['{"verdict": "SUPPORTED", "issues": []}']

    def __call__(self, system, user, *, model, temperature, max_tokens, provider_order=None,
                 role="llm", thinking_off=None):
        from app.llm_client import ChatResult
        self.roles.append(role)
        text = {
            "generator": ANSWER,
            "ragas:groundedness": '{"claims": [{"claim": "Sellers must display the badge", "supported": true},'
                                  ' {"claim": "The fine is P1,000", "supported": false}]}',
            "ragas:answer_relevance": '{"questions": ["Must online sellers display the badge?", "Q2?", "Q3?"]}',
            "ragas:answer_correctness": '{"facts": [{"fact": "must display the badge", "verdict": "correct",'
                                        ' "evidence": "must display"}, {"fact": "on the shop page",'
                                        ' "verdict": "missing", "evidence": ""}]}',
        }.get(role)
        if role == "verifier":
            text = self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]
        return ChatResult(text=text or "", finish_reason="stop", model=model)


@pytest.fixture
def fakes(monkeypatch):
    from app import api, guard, llm_client, retrieval
    from rag_eval import embeddings_eval, judge
    llm = FakeLLM()
    monkeypatch.setattr(api.trace_log, "write", lambda tr: None)
    monkeypatch.setattr(api.answer_cache, "CACHE_ENABLED", False)
    monkeypatch.setattr(api, "get_embedding_function", lambda: None)
    monkeypatch.setattr(api, "_warn_if_stale_index", lambda: None)
    monkeypatch.setattr(api, "Chroma", FakeChroma)
    monkeypatch.setattr(retrieval, "_semantic_search", fake_semantic)
    monkeypatch.setattr(retrieval, "_get_bm25_index", lambda db: FakeBM25())
    monkeypatch.setattr(retrieval, "RERANK_ON", True)
    monkeypatch.setattr(retrieval.reranker, "rerank", lambda q, docs: [0.9, 0.5, 0.05][: len(docs)])
    monkeypatch.setattr(llm_client, "chat", llm)
    monkeypatch.setattr(judge, "JUDGE_MODEL", "judge/independent-model")
    monkeypatch.setattr(embeddings_eval, "embed_for_matching",
                        lambda texts: [[1.0, 0.0]] + [[0.8, 0.6]] * (len(texts) - 1))
    monkeypatch.setattr(guard, "EVAL_LOCAL_ONLY", False)
    # The test client counts as a visitor from the public link, so demo runs
    # share /query's limits (v1.5); give the tests room and a clean slate.
    monkeypatch.setattr(guard, "RATE_LIMIT_PER_MIN", 1000)
    monkeypatch.setattr(guard, "DAILY_QUERY_CAP", 100000)
    guard._hits.clear()
    guard._bad_codes.clear()
    return llm


def collect(fn):
    events = []
    with progress.listening(events.append):
        result = fn()
    return result, events


# ── Process events ────────────────────────────────────────────────────────

def test_emit_is_a_noop_without_a_listener():
    assert not progress.active()
    progress.emit("anything", data=1)  # must not raise or print anything


def test_sagot_pipeline_reports_every_architecture_step_in_order(fakes):
    from app.api import run_query
    (response, _trace), events = collect(lambda: run_query(QueryRequest(
        query="Do online sellers need to display a BIR Registration Seal Badge?", bypass_cache=True)))
    assert response.outcome == "verified" and response.answer == ANSWER
    done = [e["step"] for e in events if e["status"] == "done"]
    expected = ["sanitize", "language", "intent", "route", "year_filter", "index", "cache",
                "semantic", "bm25", "fusion", "rerank", "dense_floor", "evidence", "generate", "verify", "final"]
    assert done == expected
    rerank = next(e for e in events if e["step"] == "rerank" and e["status"] == "done")
    assert rerank["data"]["kept"] == 2 and rerank["data"]["n_candidates"] == 3   # 0.05 is cut
    floor = next(e for e in events if e["step"] == "dense_floor")["data"]
    assert floor["added"] == 1 and floor["total"] == 3                          # the cut chunk is put back
    evidence = next(e for e in events if e["step"] == "evidence")["data"]["excerpts"]
    assert len(evidence) == 3
    verify = next(e for e in events if e["step"] == "verify" and e["status"] == "done")
    assert verify["data"]["passed"] is True
    assert next(e for e in events if e["step"] == "final")["data"]["outcome"] == "verified"


def test_a_rejected_draft_shows_the_retry_loop(fakes):
    from app.api import run_query
    fakes.verdicts = ['{"verdict": "UNSUPPORTED", "issues": ["The fine is not in the excerpts"]}',
                      '{"verdict": "SUPPORTED", "issues": []}']
    (response, _), events = collect(lambda: run_query(QueryRequest(
        query="Do online sellers need to display a BIR Registration Seal Badge?", bypass_cache=True)))
    verifies = [e["data"] for e in events if e["step"] == "verify" and e["status"] == "done"]
    assert [v["passed"] for v in verifies] == [False, True]
    assert verifies[0]["will_retry"] and verifies[0]["issues"] == ["The fine is not in the excerpts"]
    second = [e for e in events if e["step"] == "generate" and e["status"] == "running"][1]
    assert "AUDIT NOTICE" in second["data"]["retry_reason"]
    assert response.outcome == "verified" and response.attempts == 2


def test_events_do_not_change_the_answer(fakes):
    from app.api import run_query
    q = QueryRequest(query="Do online sellers need to display a BIR Registration Seal Badge?", bypass_cache=True)
    quiet, _ = run_query(q)
    (loud, _), _events = collect(lambda: run_query(q))
    assert (quiet.answer, quiet.sources, quiet.outcome) == (loud.answer, loud.sources, loud.outcome)


# ── Demo metrics ──────────────────────────────────────────────────────────

def test_demo_metrics_formulas(fakes):
    from rag_eval import demo_metrics
    m = demo_metrics.score_demo("Must sellers display the badge?", "[1] RMC\ntext", ANSWER, reference="They must.")
    assert m["groundedness"]["score"] == 0.5                               # 1 of 2 claims supported
    assert m["context_relevance"]["score"] == pytest.approx(0.8)           # cosine([1,0],[0.8,0.6])
    assert m["answer_relevance"]["score"] == 0.5                           # 1 of 2 reference facts correct
    assert "reference" in m["answer_relevance"]["formula"]
    assert "questions generated from the answer" in m["context_relevance"]["formula"]


def test_demo_metrics_not_applicable_is_none_with_a_reason(fakes):
    from rag_eval import demo_metrics
    m = demo_metrics.score_demo("q?", "", "some answer", reference="")
    assert m["groundedness"]["score"] is None and m["groundedness"]["note"]
    assert m["answer_relevance"]["score"] is None and "reference" in m["answer_relevance"]["note"]
    assert m["context_relevance"]["score"] is not None


# ── Streaming endpoint ────────────────────────────────────────────────────

def _run(client, **body):
    r = client.post("/eval/api/run/demo", json=body)
    return r, [json.loads(line) for line in r.text.splitlines() if line.strip()]


@pytest.fixture
def client(fakes):
    import main
    return TestClient(main.app)


def test_demo_compare_with_revie_streams_both_systems(client, fakes):
    r, events = _run(client, query="Do online sellers need to display a BIR Registration Seal Badge?",
                     compare="revie", reference="Yes, they must display it.", revie_answer="I am not sure.")
    assert r.status_code == 200
    assert events[0]["type"] == "start" and events[0]["systems"] == ["sagot", "revie"]
    assert events[-1]["type"] == "done"
    assert not [e for e in events if e["type"] == "error"]
    answers = {e["system"]: e for e in events if e["type"] == "answer"}
    assert set(answers) == {"sagot", "revie"}
    assert answers["sagot"]["outcome"] == "verified" and answers["revie"]["answer"] == "I am not sure."
    done_metrics = {(e["system"], e["metric"]): e["result"] for e in events
                    if e["type"] == "metric" and e["status"] == "done"}
    assert len(done_metrics) == 6                                         # 2 systems x 3 metrics
    assert done_metrics[("revie", "groundedness")]["score"] is None       # REVIE's context is not visible
    assert done_metrics[("sagot", "groundedness")]["score"] == 0.5
    assert {e["system"] for e in events if e["type"] == "step"} == {"sagot", "revie"}
    assert not [role for role in fakes.roles if role.startswith("baseline")]   # no baseline model is called


def test_demo_compare_requires_the_pasted_revie_answer(client):
    r = client.post("/eval/api/run/demo", json={"query": "Is POGO banned?", "compare": "revie"})
    assert r.status_code == 422 and "REVIE" in r.json()["detail"]


def test_demo_greeting_is_not_scored(client):
    _r, events = _run(client, query="hi", compare="none")
    assert [e["type"] for e in events if e["type"] in ("metric", "metrics_skipped")] == ["metrics_skipped"]


def test_demo_page_and_config_are_served(client):
    assert client.get("/demo").status_code == 200
    cfg = client.get("/demo/api/config").json()
    assert set(cfg["systems"]) == {"sagot", "revie"}                     # v1.5: Sagot AI vs REVIE only
    assert cfg["compare_modes"] == {"none": ["sagot"], "revie": ["sagot", "revie"]}
    assert cfg["version"] == "v1.5"
    for gone in ("c0", "c1", "c2", "all"):                               # removed comparison modes
        assert client.post("/eval/api/run/demo", json={"query": "q?", "compare": gone}).status_code == 422
    assert set(cfg["metrics"]) == {"groundedness", "context_relevance", "answer_relevance"}


def test_demo_run_is_locked_to_the_host_computer(client, monkeypatch):
    from app import guard
    monkeypatch.setattr(guard, "EVAL_LOCAL_ONLY", True)
    r = client.post("/eval/api/run/demo", json={"query": "Is POGO banned?", "compare": "none"},
                    headers={"x-forwarded-for": "8.8.8.8"})
    assert r.status_code == 403


# ── Team access for the public link (v1.5) ────────────────────────────────

PUBLIC = {"x-forwarded-for": "8.8.8.8"}


def test_team_code_unlocks_the_demo_on_the_public_link(client, monkeypatch):
    from app import guard
    monkeypatch.setattr(guard, "EVAL_LOCAL_ONLY", True)
    monkeypatch.setattr(guard, "TEAM_ACCESS_CODE", "kalabaw-2026")
    body = {"query": "hi", "compare": "none"}
    no_code = client.post("/eval/api/run/demo", json=body, headers=PUBLIC)
    assert no_code.status_code == 403 and "team access code" in no_code.json()["detail"]
    ok = client.post("/eval/api/run/demo", json=body, headers={**PUBLIC, "x-team-code": "kalabaw-2026"})
    assert ok.status_code == 200 and '"type": "done"' in ok.text
    assert client.get("/team/api/status", headers={**PUBLIC, "x-team-code": "kalabaw-2026"}).json() == \
        {"enabled": True, "valid": True, "host": False}
    cfg = client.get("/demo/api/config", headers=PUBLIC).json()
    assert cfg["can_run"] is False and cfg["team_code_enabled"] is True and "team access code" in cfg["lock_reason"]


def test_wrong_team_codes_are_locked_out(client, monkeypatch):
    from app import guard
    monkeypatch.setattr(guard, "EVAL_LOCAL_ONLY", True)
    monkeypatch.setattr(guard, "TEAM_ACCESS_CODE", "kalabaw-2026")
    wrong = {**PUBLIC, "x-team-code": "guess"}
    for _ in range(10):
        assert client.post("/eval/api/run/demo", json={"query": "hi"}, headers=wrong).status_code == 403
    assert client.post("/eval/api/run/demo", json={"query": "hi"}, headers=wrong).status_code == 429
    # the right code from the same address is also refused until the window passes
    right = {**PUBLIC, "x-team-code": "kalabaw-2026"}
    assert client.post("/eval/api/run/demo", json={"query": "hi"}, headers=right).status_code == 429


def test_public_demo_runs_share_the_query_rate_limit(client, monkeypatch):
    from app import guard
    monkeypatch.setattr(guard, "RATE_LIMIT_PER_MIN", 1)
    assert client.post("/eval/api/run/demo", json={"query": "hi"}, headers=PUBLIC).status_code == 200
    assert client.post("/eval/api/run/demo", json={"query": "hi"}, headers=PUBLIC).status_code == 429


def test_team_code_never_opens_admin_routes(client, monkeypatch):
    from app import guard
    monkeypatch.setattr(guard, "TEAM_ACCESS_CODE", "kalabaw-2026")
    r = client.delete("/reset", headers={**PUBLIC, "x-team-code": "kalabaw-2026"})
    assert r.status_code == 403
