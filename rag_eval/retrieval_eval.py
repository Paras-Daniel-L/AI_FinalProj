"""
Retrieval-only evaluation and threshold calibration — no LLM calls, no judge.

Why: Context Relevance (an LLM judge) can't tell you WHY a context is poor,
and it costs a judge call per question. Every T-TED row already names the
issuance that answers it (`source`, e.g. "RMC No. 116-2024"), which is a
free gold label for retrieval. This script measures, per pipeline setting:

    hit@1          the top-ranked chunk comes from the gold issuance
    hit@kept       at least one chunk passed to the generator is from it
                   (the number that must stay high when chunks are cut)
    gold_share     fraction of passed chunks that are from the gold issuance
                   (a judge-free proxy for context precision)
    mean_kept      chunks passed to the generator per question
    MRR@pool       reciprocal rank of the first gold chunk in the candidate
                   pool (stage 1 quality, before the reranker)

and, with --sweep, re-applies the reranker's dynamic cut over a grid of
thresholds OFFLINE (the candidates and their rerank scores are fetched once
and cached), so choosing RERANK_MIN_SCORE / RERANK_RELATIVE / RERANK_MAX_DOCS
costs one pass of API calls, not one per setting.

TUNE ON A DEV SET, NOT ON T-TED. Picking thresholds on the same questions
you report on makes the reported numbers optimistic. Write 30-50 questions
that are NOT in T-TED (same JSON shape: qid, query, source) and pass them with
--dataset. Running on T-TED is fine for a final, untuned report.

Usage (project root, after `python -m app.database --reset`):
    python -m rag_eval.retrieval_eval --dataset rag_eval/dev_set.json --sweep
    python -m rag_eval.retrieval_eval                       # T-TED, current settings
    python -m rag_eval.retrieval_eval --refresh             # ignore the score cache
"""

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document  # noqa: E402

from app.textproc import issuance_title, parse_issuance_ids  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_DATASET = Path(__file__).parent / "ted_dataset.json"


def chunk_ids(meta: Dict[str, Any]) -> List[str]:
    """The issuance(s) a chunk belongs to: its stamped issuance_id, else the
    ids parsed from its PDF filename (works on a legacy index too)."""
    own = meta.get("issuance_id")
    if own:
        return [own]
    return parse_issuance_ids(issuance_title(meta.get("source", "")))


def is_gold(meta: Dict[str, Any], gold: List[str]) -> bool:
    return bool(set(chunk_ids(meta)) & set(gold))


def collect(rows: List[Dict[str, Any]], cache_path: Path, refresh: bool) -> List[Dict[str, Any]]:
    """Stage-1 candidates + rerank scores per question, cached as JSON."""
    if cache_path.exists() and not refresh:
        print(f"Using cached candidates/scores: {cache_path} (pass --refresh to recompute)")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    from langchain_chroma import Chroma

    from app import rerank as reranker
    from app import retrieval
    from app.classifier import get_year_filter
    from app.embeddings import get_embedding_function

    db = Chroma(persist_directory="chroma", embedding_function=get_embedding_function())
    out = []
    for r in rows:
        q = r["query"]
        cands = retrieval.retrieve_candidates(q, get_year_filter(q), db)
        scores: Optional[List[float]] = None
        if reranker.enabled() and cands:
            try:
                scored = retrieval.score_candidates(q, cands, parse_issuance_ids(q))
                cands = scored
                scores = [d.metadata["_rerank_score"] for d in scored]
            except reranker.RerankError as e:
                print(f"  ⚠️ {r['qid']}: rerank failed ({e}); stage-1 order only")
        out.append({
            "qid": r["qid"], "query": q, "source": r.get("source", ""),
            "candidates": [{"text": d.page_content, "meta": {k: v for k, v in d.metadata.items()
                                                             if isinstance(v, (str, int, float, bool)) or v is None}}
                           for d in cands],
            "scores": scores,
        })
        print(f"  {r['qid']}: {len(cands)} candidates")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def evaluate(items: List[Dict[str, Any]], cut: Optional[Dict[str, Any]] = None,
             legacy_k: Optional[int] = None) -> Dict[str, Any]:
    """Metrics for one setting. cut=None + legacy_k -> plain top-k (no reranker)."""
    from app import retrieval

    hit1, hitk, share, kept_n, rr = [], [], [], [], []
    for it in items:
        gold = parse_issuance_ids(it["source"])
        if not gold:
            continue
        docs = [Document(page_content=c["text"], metadata=dict(c["meta"])) for c in it["candidates"]]
        if not docs:
            hit1.append(0); hitk.append(0); share.append(0.0); kept_n.append(0); rr.append(0.0)
            continue
        first = next((i for i, d in enumerate(docs, 1) if is_gold(d.metadata, gold)), None)
        rr.append(1.0 / first if first else 0.0)
        if cut is not None and it.get("scores") is not None:
            kept = retrieval.dynamic_cut(docs, **cut)
        else:
            kept = docs[: (legacy_k or retrieval.FALLBACK_TOP_K)]
        flags = [is_gold(d.metadata, gold) for d in kept]
        hit1.append(int(bool(flags) and flags[0]))
        hitk.append(int(any(flags)))
        share.append(sum(flags) / len(flags) if flags else 0.0)
        kept_n.append(len(kept))
    n = len(hitk)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731
    return {"n": n, "hit@1": mean(hit1), "hit@kept": mean(hitk), "gold_share": mean(share),
            "mean_kept": mean(kept_n), "MRR@pool": mean(rr)}


def fmt(m: Dict[str, Any]) -> str:
    return (f"hit@1={m['hit@1']:.3f}  hit@kept={m['hit@kept']:.3f}  gold_share={m['gold_share']:.3f}  "
            f"mean_kept={m['mean_kept']:.2f}  MRR@pool={m['MRR@pool']:.3f}  (n={m['n']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="JSON list with qid, query, source")
    parser.add_argument("--sweep", action="store_true", help="grid-search the dynamic-cut thresholds offline")
    parser.add_argument("--refresh", action="store_true", help="recompute candidates/scores (ignore cache)")
    parser.add_argument("--min-hit", type=float, default=0.95,
                        help="--sweep: only recommend settings keeping hit@kept >= this fraction of the "
                             "candidate pool's own hit rate (default 0.95)")
    args = parser.parse_args()

    from app import rerank as _rerank
    _rerank.use_evaluation_retries()  # wait out rate limits; never score the fallback path

    dataset = Path(args.dataset)
    rows = json.loads(dataset.read_text(encoding="utf-8"))
    if dataset.resolve() == DEFAULT_DATASET.resolve() and args.sweep:
        print("⚠️  Sweeping thresholds on T-TED itself: numbers chosen this way are optimistic if you then\n"
              "   report T-TED results. Prefer --dataset <a dev set that is not T-TED>.\n")

    cache = RESULTS_DIR / f"retrieval_cache_{dataset.stem}.json"
    items = collect(rows, cache, args.refresh)

    from app import retrieval
    print("\nCurrent settings:")
    print("  legacy top-5 (no reranker): " + fmt(evaluate(items, None, legacy_k=5)))
    if any(it.get("scores") is not None for it in items):
        print("  reranker + dynamic cut:     " + fmt(evaluate(items, {})))
        pool = evaluate(items, {"min_score": -1.0, "relative": 0.0, "min_docs": 999, "max_docs": 999,
                                "dedup_overlap": 2.0})
        print("  whole candidate pool:       " + fmt(pool))
    else:
        print("  (reranker off or failed — no rerank scores to evaluate)")
        return

    if not args.sweep:
        return
    grid = itertools.product([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
                             [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                             [1, 2, 3, 4])
    floor = args.min_hit * pool["hit@kept"]
    results = []
    for min_score, rel, max_docs in grid:
        m = evaluate(items, {"min_score": min_score, "relative": rel, "max_docs": max_docs})
        results.append(((min_score, rel, max_docs), m))
    ok = [r for r in results if r[1]["hit@kept"] >= floor]
    ok.sort(key=lambda r: (-r[1]["gold_share"], r[1]["mean_kept"]))
    print(f"\nSettings with hit@kept >= {floor:.3f}, ranked by gold_share (precision) then fewer chunks:")
    print("  min_score  relative  max_docs | metrics")
    for (ms, rel, mx), m in ok[:15]:
        print(f"  {ms:9.2f}  {rel:8.2f}  {mx:8d} | {fmt(m)}")
    if ok:
        (ms, rel, mx), _m = ok[0]
        print(f"\nSuggested .env:  RERANK_MIN_SCORE={ms}  RERANK_RELATIVE={rel}  RERANK_MAX_DOCS={mx}")
    else:
        print("\nNo setting kept the recall floor — lower --min-hit or check the reranker.")


if __name__ == "__main__":
    main()
