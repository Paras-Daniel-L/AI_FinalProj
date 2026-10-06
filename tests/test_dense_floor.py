"""
Tests for the dense-search floor (system v1.3, app/retrieval.add_dense_floor)
and the version label (app/version.py).

The floor guarantees the generator sees at least the top-k dense hits a
standard RAG would, in addition to the reranker's picks.
"""

from langchain_core.documents import Document

from app import retrieval, version

TEXTS = [
    "Online sellers must display the registration badge on their shop page.",
    "Withholding agents remit taxes on or before the tenth day of the following month.",
    "The percentage tax rate for small businesses returned to three percent.",
    "Digital service providers register for value-added tax through the portal.",
    "Penalties apply for late filing of the annual information return.",
    "Revenue district offices monitor compliance through regular checks.",
]


def doc(n, dist=None, rerank=None):
    meta = {"id": f"c{n}", "source": f"data/2026/RMC No. {n}-2026.pdf", "page": 0, "year": "2026"}
    if dist is not None:
        meta["_semantic_distance"] = dist
    if rerank is not None:
        meta["_rerank_score"] = rerank
    return Document(page_content=TEXTS[n], metadata=meta)


def test_floor_adds_dropped_dense_hits_after_the_reranker_picks():
    candidates = [doc(i, dist=0.1 * (i + 1)) for i in range(6)]
    kept = [doc(3, dist=0.4, rerank=0.9)]                     # the reranker kept only one
    out = retrieval.add_dense_floor(kept, candidates, k=4, max_docs=6)
    assert [d.metadata["id"] for d in out] == ["c3", "c0", "c1", "c2"]   # reranker first, then dense order
    assert [bool(d.metadata.get("_dense_floor")) for d in out] == [False, True, True, True]


def test_floor_respects_the_cap_and_skips_duplicates():
    candidates = [doc(i, dist=0.1 * (i + 1)) for i in range(6)]
    kept = [doc(0, dist=0.1), doc(1, dist=0.2), doc(4)]
    out = retrieval.add_dense_floor(kept, candidates, k=4, max_docs=4)
    assert [d.metadata["id"] for d in out] == ["c0", "c1", "c4", "c2"]   # c0/c1 not duplicated; capped at 4


def test_floor_never_drops_reranker_picks_and_can_be_turned_off():
    candidates = [doc(i, dist=0.1 * (i + 1)) for i in range(6)]
    kept = [doc(i) for i in range(5)]
    assert len(retrieval.add_dense_floor(kept, candidates, k=4, max_docs=3)) == 5   # cap never removes picks
    assert retrieval.add_dense_floor(kept[:1], candidates, k=0) == kept[:1]          # DENSE_FLOOR_K=0: off


def test_floor_skips_near_identical_text():
    a = doc(0, dist=0.1)
    twin = Document(page_content=TEXTS[0], metadata={"id": "other", "_semantic_distance": 0.05})
    out = retrieval.add_dense_floor([a], [twin, doc(1, dist=0.2)], k=2, max_docs=6)
    assert [d.metadata["id"] for d in out] == ["c0", "c1"]


def test_settings_are_recorded_for_the_cache_and_evaluation():
    s = retrieval.retrieval_settings()
    assert s["dense_floor_k"] == retrieval.DENSE_FLOOR_K and s["evidence_max_docs"] == retrieval.EVIDENCE_MAX_DOCS


def test_version_label_matches_the_default_settings():
    assert version.SYSTEM_VERSION.startswith("1.5")
    assert version.effective_label() == "v1.5.1"
