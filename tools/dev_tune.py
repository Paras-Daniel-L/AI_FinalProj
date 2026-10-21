"""
DEV-set tuning and ablation for the v1.6 hallucination-mitigation changes.

Every threshold in v1.6 must be chosen on DEV questions, never on T-TED (the
thesis test set). This tool is how. It has four commands; the first costs
nothing, the others spend API credit (shown below).

  python -m tools.dev_tune logs
      FREE, offline. Reads logs/queries.jsonl and shows what v1.6 would have
      done to the real questions people asked: the best rerank score of each
      question vs. the evidence-gate threshold, which verified answers the
      non-answer detector catches, and which ones the deterministic grounding
      check flags. Questions that look like T-TED questions (word overlap
      >= 0.4) are skipped, so the test set never informs a threshold.

  python -m tools.dev_tune gate-sweep [--dev dev/dev_questions.jsonl]
      Jina only (embeddings + reranker, no LLM). Retrieves every DEV
      question and prints, for each candidate threshold, how many answerable
      questions would still be answered and how many unanswerable ones would
      be stopped before generation. Pick the threshold, then put
      EVIDENCE_GATE_MIN_SCORE=<value> in .env.

  python -m tools.dev_tune ablation [--dev ...] [--configs v152,gate,...] [--limit N]
      Jina + OpenRouter + judge. Runs the full pipeline once per
      configuration (each in its own process, so .env-style switches apply
      cleanly) and scores every answer with the thesis metrics
      (rag_eval/ragas_metrics.py: Groundedness, Context Relevance, Answer
      Relevance, plus Answer Correctness when a DEV row has a reference).
      Prints one table, writes dev/results/ablation.csv. The ladder adds one
      group of changes at a time:

        v152       every v1.6 switch off (= v1.5.2 answers)
        +gate      + evidence gate, non-answer detector, conditional floor
        +ground    + deterministic grounding check and claim repair
        +context   + sentence-level evidence selection, extra BM25 terms
        v16        + prompt v4 and verifier v2 (= the v1.6 defaults)

  python -m tools.dev_tune run --config NAME ...
      Internal: one configuration (what `ablation` starts per config).

The DEV file
------------
dev/dev_questions.jsonl, one JSON object per line:
    {"id": "D01", "query": "...", "label": "answerable", "label_status": "proposed",
     "reference": "", "note": "..."}
label is one of: answerable | not_in_corpus | out_of_scope | unknown.
The starter file was drafted from real questions in the logs (none of them
T-TED-like); its labels are PROPOSED. Check each one against the documents,
set label_status to "checked", and add a reference answer where you can.
Rows labelled "unknown" are run but left out of the answer/refusal rates.

Results go to dev/results/ (not to rag_eval/results*, which hold the
thesis runs).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEV_DEFAULT = ROOT / "dev" / "dev_questions.jsonl"
RESULTS = ROOT / "dev" / "results"
LOGS = ROOT / "logs" / "queries.jsonl"
TED = ROOT / "rag_eval" / "ted_dataset.json"

REFUSALS = {"no_retrieval", "generator_no_answer", "verification_failed", "no_index"}
ANSWERABLE = "answerable"
UNANSWERABLE = {"not_in_corpus", "out_of_scope"}

OFF = {"DENSE_FLOOR_RELATIVE": "0", "EVIDENCE_COMPRESSION": "0", "EVIDENCE_GATE_MIN_SCORE": "0",
       "NON_ANSWER_DETECTION": "0", "GROUNDING_CHECK": "0", "CLAIM_REPAIR": "0", "VERIFIER_RELEVANCE": "0",
       "QUERY_GLOSSARY_V16": "0", "RAG_PROMPT_VERSION": "3"}
CONFIGS: Dict[str, Dict[str, str]] = {
    "v152": dict(OFF),
    "+gate": {**OFF, "EVIDENCE_GATE_MIN_SCORE": "0.14", "NON_ANSWER_DETECTION": "1", "DENSE_FLOOR_RELATIVE": "0.5"},
    "+ground": {**OFF, "EVIDENCE_GATE_MIN_SCORE": "0.14", "NON_ANSWER_DETECTION": "1", "DENSE_FLOOR_RELATIVE": "0.5",
                "GROUNDING_CHECK": "1", "CLAIM_REPAIR": "1"},
    "+context": {**OFF, "EVIDENCE_GATE_MIN_SCORE": "0.14", "NON_ANSWER_DETECTION": "1", "DENSE_FLOOR_RELATIVE": "0.5",
                 "GROUNDING_CHECK": "1", "CLAIM_REPAIR": "1", "EVIDENCE_COMPRESSION": "1", "QUERY_GLOSSARY_V16": "1"},
    "v16": {},  # the defaults
}


# ── Shared helpers ──────────────────────────────────────────────────────

def _words(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _ted_queries() -> List[set]:
    """T-TED question word sets, used ONLY to skip look-alike questions."""
    try:
        return [_words(r["query"]) for r in json.loads(TED.read_text(encoding="utf-8"))]
    except Exception:
        return []


def looks_like_ted(query: str, ted: List[set], limit: float = 0.4) -> bool:
    w = _words(query)
    return any(len(w & t) / max(1, len(w | t)) >= limit for t in ted)


def load_dev(path: Path) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ted = _ted_queries()
    clash = [r["id"] for r in rows if looks_like_ted(r["query"], ted)]
    if clash:
        sys.exit(f"These DEV rows look like T-TED questions; remove or reword them first: {', '.join(clash)}")
    proposed = sum(1 for r in rows if r.get("label_status") != "checked")
    if proposed:
        print(f"ℹ️  {proposed} of {len(rows)} DEV labels are still 'proposed'. Check them before trusting the numbers.\n")
    return rows


def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.mean(xs), 3) if xs else None


def _fmt(x) -> str:
    return "  –  " if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)


# ── logs (free) ─────────────────────────────────────────────────────────

def cmd_logs(args) -> None:
    from app import answerability, grounding

    if not LOGS.exists():
        sys.exit(f"{LOGS} not found.")
    ted = _ted_queries()
    rows = [json.loads(l) for l in LOGS.read_text(encoding="utf-8").splitlines() if l.strip()]
    seen, scored, skipped = set(), [], 0
    for r in rows:
        q = (r.get("query") or "").strip()
        if not q or q.lower() in seen or not r.get("retrieved"):
            continue
        seen.add(q.lower())
        if looks_like_ted(q, ted):
            skipped += 1
            continue
        top = max((x.get("rerank_score") or -1) for x in r["retrieved"])
        scored.append((top, r.get("outcome"), q, r))
    scored.sort(key=lambda t: t[0])
    thr = answerability.EVIDENCE_GATE_MIN_SCORE
    print(f"Evidence gate (threshold {thr}) on {len(scored)} logged RAG questions "
          f"({skipped} T-TED-like questions skipped):\n")
    for top, outcome, q, _r in scored:
        mark = "STOP " if top < thr else "     "
        print(f"  {mark}{top:6.3f}  {outcome:21s} {q[:90]}")

    print("\nNon-answer detector and grounding check on logged verified answers:")
    n = non = flagged = 0
    for top, outcome, q, r in scored:
        if outcome != "verified":
            continue
        n += 1
        answer = r.get("answer") or ""
        if answerability.is_non_answer(answer):
            non += 1
            print(f"  NON-ANSWER  {answerability.first_sentence(answer)[:100]}")
            continue
        ctx = "\n\n---\n\n".join(f"{x['label']}\n{x['text'].strip()}" for x in r["retrieved"])
        rep = grounding.check(answer, ctx, q)
        if not rep.passed:
            flagged += 1
            print(f"  GROUNDING   {'; '.join(rep.issues)[:120]}   ← {q[:50]}")
    print(f"\n{n} verified answers: {non} non-answers, {flagged} flagged by the grounding check.")


# ── gate-sweep (Jina only) ──────────────────────────────────────────────

def cmd_gate_sweep(args) -> None:
    from langchain_chroma import Chroma

    from app import answerability, intent as intent_router
    from app.api import CHROMA_PATH
    from app.classifier import get_year_filter
    from app.embeddings import get_embedding_function
    from app.retrieval import retrieve_docs

    rows = load_dev(Path(args.dev))
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    results = []
    for r in rows:
        label = intent_router.classify(r["query"]).label
        if label in (intent_router.CHAT, intent_router.TAX_COMPUTATION):
            print(f"  {r['id']}  routed to {label}, no retrieval: {r['query'][:70]}")
            continue
        docs = retrieve_docs(r["query"], get_year_filter(r["query"]), db)
        top = answerability.top_rerank_score(docs)
        results.append((r, top if top is not None else -1.0))
        print(f"  {r['id']}  {r['label']:13s} top={results[-1][1]:.3f}  {r['query'][:70]}")

    answerable = [t for r, t in results if r["label"] == ANSWERABLE]
    unanswerable = [t for r, t in results if r["label"] in UNANSWERABLE]
    print(f"\n{'threshold':>9}  {'answerable kept':>16}  {'unanswerable stopped':>21}")
    for thr in [0, 0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30]:
        kept = sum(1 for t in answerable if t >= thr)
        stopped = sum(1 for t in unanswerable if t < thr)
        print(f"{thr:9.2f}  {kept:>7}/{len(answerable):<8}  {stopped:>10}/{len(unanswerable):<10}")
    print("\nPick the highest threshold that keeps every answerable question you care about, "
          "then set EVIDENCE_GATE_MIN_SCORE in .env.")


# ── run / ablation (full pipeline + judge) ──────────────────────────────

def cmd_run(args) -> None:
    """One configuration, in this process (its env was set by the caller)."""
    from app.api import run_query
    from app.schemas import QueryRequest
    from app.version import current_settings
    from rag_eval import ragas_metrics
    from rag_eval.run_evaluation import _rebuild_context

    rows = load_dev(Path(args.dev))[: args.limit or None]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            rec: Dict[str, Any] = {"id": r["id"], "label": r["label"], "query": r["query"], "config": args.config,
                                   "settings": current_settings()}
            try:
                response, trace = run_query(QueryRequest(query=r["query"], bypass_cache=True))
                rec.update(outcome=response.outcome, mode=response.mode, answer=response.answer,
                           attempts=response.attempts, cost=trace.get("cost"))
                if response.outcome in ("verified", "cache_hit"):
                    ctx = _rebuild_context(trace.get("retrieved") or [])
                    rec["scores"] = ragas_metrics.score_response(r["query"], ctx, response.answer,
                                                                 ground_truth=r.get("reference") or "").to_row()
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  [{args.config}] {r['id']} {rec.get('outcome', 'ERROR')}")


def _summarize(name: str, path: Path) -> Dict[str, Any]:
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    ans = [r for r in recs if r["label"] == ANSWERABLE]
    una = [r for r in recs if r["label"] in UNANSWERABLE]
    scored = [r["scores"] for r in recs if r.get("scores") and r["label"] == ANSWERABLE]
    return {
        "config": name,
        "n": len(recs),
        "answered (answerable)": f"{sum(1 for r in ans if r.get('outcome') == 'verified')}/{len(ans)}",
        "refused (unanswerable)": f"{sum(1 for r in una if r.get('outcome') in REFUSALS or r.get('mode') in ('chat', 'clarify'))}/{len(una)}",
        "groundedness": _mean(s.get("groundedness") for s in scored),
        "context_relevance": _mean(s.get("context_relevance") for s in scored),
        "answer_relevance": _mean(s.get("answer_relevance") for s in scored),
        "answer_correctness": _mean(s.get("answer_correctness") for s in scored),
        "errors": sum(1 for r in recs if r.get("error")),
        "cost": round(sum(r.get("cost") or 0 for r in recs), 4),
    }


def cmd_ablation(args) -> None:
    names = [c.strip() for c in args.configs.split(",")] if args.configs else list(CONFIGS)
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        sys.exit(f"Unknown config(s): {unknown}. Choose from {list(CONFIGS)}.")
    load_dev(Path(args.dev))  # validate once, before spending anything
    RESULTS.mkdir(parents=True, exist_ok=True)
    table = []
    for name in names:
        out = RESULTS / f"ablation_{name.replace('+', 'plus_')}.jsonl"
        if not (args.reuse and out.exists()):
            env = {**os.environ, **CONFIGS[name], "CACHE_ENABLED": "0", "PYTHONIOENCODING": "utf-8"}
            cmd = [sys.executable, "-m", "tools.dev_tune", "run", "--config", name, "--dev", str(args.dev),
                   "--out", str(out)] + (["--limit", str(args.limit)] if args.limit else [])
            print(f"\n▶ {name}: {' '.join(f'{k}={v}' for k, v in CONFIGS[name].items()) or '(v1.6 defaults)'}")
            subprocess.run(cmd, cwd=ROOT, env=env, check=False)
        table.append(_summarize(name, out))

    cols = list(table[0].keys())
    print("\n" + "  ".join(f"{c:>22}" for c in cols))
    for row in table:
        print("  ".join(f"{_fmt(row[c]):>22}" for c in cols))
    with open(RESULTS / "ablation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)
    print(f"\nWrote {RESULTS / 'ablation.csv'}. Means are over answered, answerable DEV questions. "
          "With a small DEV set, differences under about ±0.05 are noise.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("logs", help="free: what v1.6 would do to the logged questions")
    g = sub.add_parser("gate-sweep", help="Jina only: choose EVIDENCE_GATE_MIN_SCORE on DEV")
    g.add_argument("--dev", default=str(DEV_DEFAULT))
    a = sub.add_parser("ablation", help="full pipeline + judge, one run per configuration")
    a.add_argument("--dev", default=str(DEV_DEFAULT))
    a.add_argument("--configs", default="", help=f"comma-separated, from {list(CONFIGS)} (default: all)")
    a.add_argument("--limit", type=int, default=0, help="only the first N DEV rows (smoke test)")
    a.add_argument("--reuse", action="store_true", help="summarize existing result files instead of re-running")
    r = sub.add_parser("run", help="internal: one configuration")
    r.add_argument("--config", required=True)
    r.add_argument("--dev", default=str(DEV_DEFAULT))
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    {"logs": cmd_logs, "gate-sweep": cmd_gate_sweep, "ablation": cmd_ablation, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
