"""
The offline evaluation runner (thesis Task #16): the one script that turns
the T-TED dataset into the numbers RQ1/RQ2/RQ3 need, and the file both the
Chapter 4 report and the live dashboard read from.

What it does, per row of ted_dataset.json:
  - Sagot AI side (every row, Subset A and B): calls app.api.run_query()
    IN-PROCESS (no HTTP, no rate limit, no daily cap — see api.py's own
    docstring on run_query) with bypass_cache=True, so every query is
    answered fresh, exactly like a real user would get, but not served
    from a previously-cached verified answer. The retrieved context is
    rebuilt from the trace's `retrieved` list in the exact block format
    format_context() uses, so RAGAS scores the answer against precisely
    what the generator/verifier actually saw.
  - REVIE side (Subset B only): no call at all — REVIE's answer was
    already collected by hand into the dataset. Scored with context=""
    (see ragas_metrics.score_response()'s docstring: Groundedness and
    Context Relevance correctly come back as None — undefined, not 0 —
    since REVIE has no retrieved context to check against; Answer
    Relevance still runs).
  - RQ2 (trigger/language detection) needs no LLM call at all: it's
    computed straight from app.language.detect_language(), which is
    pure Python. It runs immediately, before any paid API call, so a
    `--dry-run` gives partial results for free.

Runs blind: nothing here is tuned to produce a particular conclusion, and
every score that can't be meaningfully computed stays None rather than
being coerced to 0.0/1.0 — see dataset.py's LANGUAGE NOTE and every
Result class's own docstring in judge.py/ragas_metrics.py/stats.py for
why that matters here specifically.

Checkpointing: this run makes real, billed API calls (generator, verifier,
judge, and Jina embeddings) and can legitimately take a long time or hit a
transient failure partway through. Every row's outcome is appended to
results/checkpoint.jsonl AS SOON AS it's computed, and simply re-running
the same command skips any (qid, side) pair already present there (pass
--fresh to ignore the checkpoint and start clean) — a crash or a Ctrl-C
never means starting over and re-paying for work already done. A row that
raises is recorded with its error and the run continues; the run only
ever fails its own logic, never the whole evaluation.

Usage (run from the project root, so `app` and `rag_eval` both import):
    python -m rag_eval.run_evaluation --subset all
    python -m rag_eval.run_evaluation --subset A --limit 4   # smoke test first
    python -m rag_eval.run_evaluation --subset all           # re-run: resumes automatically
    python -m rag_eval.run_evaluation --rq2-only             # free, no API calls at all

Needs, in .env (see README.md's env-var reference for the rest): the usual
GENERATOR/VERIFIER OpenRouter setup Sagot AI already needs, JINA_API_KEY,
and JUDGE_MODEL (see judge.py — must be a third model, not the generator
or verifier).
"""

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.language import detect_language  # noqa: E402
from app.schemas import QueryRequest  # noqa: E402

from . import dataset, ragas_metrics, stats  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.jsonl"

METRICS = ("groundedness", "context_relevance", "answer_relevance", "answer_correctness")

# Bump whenever HOW an answer is scored changes (a judge prompt, a new
# metric). A checkpointed unit scored under an older version is re-scored
# automatically on the next run — reusing the answer it already has, so the
# chatbot is never called again and the scored answer stays the same one.
#   1: Groundedness, Context Relevance, Answer Relevance
#   2: + Answer Correctness (vs. ground truth); reverse-engineered questions
#      now written in the answer's own language
#   3: Context Relevance over Python-split, numbered sentences (fixed
#      denominator, label lines excluded, no output truncation) — see
#      judge.py. Report v2 and v3 Context Relevance separately; they are not
#      the same measurement.
SCORING_VERSION = 3


def _rebuild_context(retrieved: List[Dict[str, Any]]) -> str:
    """Exactly app.retrieval.format_context()'s block format, rebuilt from a
    trace's `retrieved` list instead of live Document objects, so RAGAS
    scores the answer against what the generator/verifier actually saw."""
    blocks = [f"{item['label']}\n{item['text'].strip()}" for item in retrieved]
    return "\n\n---\n\n".join(blocks)


def run_sagot(query: str) -> Dict[str, Any]:
    """One in-process call to Sagot AI's real pipeline. Returns a plain dict
    (not the pydantic/dataclass objects) so it round-trips through the JSONL
    checkpoint file untouched."""
    from app.api import run_query  # imported lazily: touches Chroma/embeddings at import time

    response, trace = run_query(QueryRequest(query=query, bypass_cache=True))
    context_text = _rebuild_context(trace.get("retrieved") or [])
    return {
        "answer": response.answer,
        "context": context_text,
        "outcome": response.outcome,
        "attempts": response.attempts,
        "mode": response.mode,
        "language": response.language,
        "classification": response.classification,
        "cost": trace.get("cost"),
        "request_id": response.request_id,
    }


def score_unit(qid: str, subset: str, side: str, query: str, context: str, answer: str,
               ground_truth: str = "") -> Dict[str, Any]:
    scored = ragas_metrics.score_response(query, context, answer, ground_truth=ground_truth)
    return {"qid": qid, "subset": subset, "side": side, "scored": scored.to_dict(), "row": scored.to_row(),
            "scoring_version": SCORING_VERSION}


def _stored_sagot(previous: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The Sagot AI answer a checkpointed unit already has, if any. Newer
    units carry it under "sagot"; the earliest ones only inside "scored"."""
    if not previous:
        return None
    if previous.get("sagot"):
        return previous["sagot"]
    scored = previous.get("scored")
    if scored and "answer" in scored:
        return {"answer": scored["answer"], "context": scored.get("context", ""),
                **(previous.get("sagot_meta") or {})}
    return None


def _load_checkpoint() -> Dict[Tuple[str, str], Dict[str, Any]]:
    done: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not CHECKPOINT_PATH.exists():
        return done
    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            done[(record["qid"], record["side"])] = record
    return done


def _append_checkpoint(record: Dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _process_row(row: dataset.TedRow, side: str, done: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    """side is 'sagot' (always) or 'revie' (Subset B only). Returns the
    checkpoint record.

    Resume rules: a unit that SUCCEEDED earlier is reused as-is. A unit that
    ERRORED earlier is retried — and if Sagot AI's answer was already
    produced before the failure (e.g. only the judge call failed), that
    exact answer and context are reused and only re-scored. That saves the
    generator/verifier cost AND keeps the run blind: the answer being scored
    is the one the pipeline gave the first time, not a second roll of the
    dice. (The checkpoint is append-only; the newest line for a unit wins.)"""
    key = (row.qid, side)
    previous = done.get(key)
    if (previous is not None and "error" not in previous
            and previous.get("scoring_version", 1) >= SCORING_VERSION):
        return previous

    started = time.monotonic()
    sagot: Optional[Dict[str, Any]] = _stored_sagot(previous)
    if previous is not None and "error" not in previous:
        print(f"(scored under v{previous.get('scoring_version', 1)}, re-scoring as v{SCORING_VERSION})",
              end=" ", flush=True)
    try:
        if side == "sagot":
            if sagot is None:
                sagot = run_sagot(row.query)
            else:
                print("(reusing earlier Sagot AI answer, re-scoring only)", end=" ", flush=True)
            unit = score_unit(row.qid, row.subset, "sagot", row.query, sagot["context"], sagot["answer"],
                              ground_truth=row.expected_answer)
            unit["sagot"] = sagot
            unit["sagot_meta"] = {k: v for k, v in sagot.items() if k not in ("answer", "context")}
        else:  # "revie" — no call, REVIE's answer was collected by hand
            unit = score_unit(row.qid, row.subset, "revie", row.query, "", row.revie_response,
                              ground_truth=row.expected_answer)
        unit["elapsed_s"] = round(time.monotonic() - started, 2)
    except Exception as e:  # a bad row must not end the whole run
        unit = {
            "qid": row.qid, "subset": row.subset, "side": side,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "elapsed_s": round(time.monotonic() - started, 2),
        }
        if sagot is not None:  # the paid pipeline answer survives a scoring failure
            unit["sagot"] = sagot
        print(f"  ✗ {row.qid} ({side}): {unit['error']}")
    _append_checkpoint(unit)
    return unit


def preflight() -> None:
    """Prove the judge and Jina are reachable BEFORE any paid pipeline call.
    Costs a fraction of a cent; exits with a readable message on failure."""
    from . import embeddings_eval, judge

    print("Preflight: checking the judge model and Jina embeddings ...")
    try:
        model = judge.preflight()
        print(f"  ✓ judge: {model}")
        embeddings_eval.embed_for_matching(["preflight check"])
        print("  ✓ Jina text-matching embeddings")
    except Exception as e:
        print(f"\n✗ Preflight failed — nothing was run, nothing else was billed.\n  {type(e).__name__}: {e}\n")
        if "No endpoints found that can handle the requested parameters" in str(e):
            print("  Hint: this usually means JUDGE_DISABLE_THINKING=1 is set for a NON-reasoning judge "
                  "model (or the reverse). See rag_eval/judge.py's docstring.\n")
        sys.exit(1)


# ── RQ2: free, no API calls ────────────────────────────────────────────────

def compute_rq2(rows: List[dataset.TedRow]) -> Dict[str, Any]:
    """Ground truth from the dataset's own subset labels vs. what
    app.language.detect_language() actually says right now, over every
    Subset A row (the only rows with known-English negatives — Subset B is
    all expected-non-English, so it can't inform a false-positive rate)."""
    a_rows = [r for r in rows if r.subset in (dataset.SUBSET_A_EN, dataset.SUBSET_A_TL)]
    y_true = [dataset.language_ground_truth(r) for r in a_rows]
    y_pred = [0 if detect_language(r.query).label == "english" else 1 for r in a_rows]
    metrics = stats.confusion_metrics(y_true, y_pred)
    per_row = [
        {
            "qid": r.qid, "query": r.query,
            "expected_non_english": bool(t),
            "detected_label": detect_language(r.query).label,
            "predicted_non_english": bool(p),
            "correct": t == p,
        }
        for r, t, p in zip(a_rows, y_true, y_pred)
    ]
    return {"confusion_metrics": metrics.__dict__, "rows": per_row}


# ── RQ1 / RQ3: paired comparisons per metric ───────────────────────────────

def _metric_arrays(units_a: List[Dict[str, Any]], units_b: List[Dict[str, Any]], metric: str):
    a = [u.get("row", {}).get(metric) if "error" not in u else None for u in units_a]
    b = [u.get("row", {}).get(metric) if "error" not in u else None for u in units_b]
    return a, b


def compute_rq1(rows: List[dataset.TedRow], done: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    """English vs. Taglish, Sagot AI only, 20 matched pairs by pair index."""
    pairs = dataset.subset_a_pairs(rows)
    en_units = [done.get((en.qid, "sagot"), {"error": "not run"}) for en, _tl in pairs]
    tl_units = [done.get((tl.qid, "sagot"), {"error": "not run"}) for _en, tl in pairs]
    result = {"n_pairs_available": len(pairs), "metrics": {}}
    for metric in METRICS:
        a, b = _metric_arrays(en_units, tl_units, metric)
        comparison = stats.paired_comparison(a, b, metric_name=f"{metric} (EN vs TL)")
        result["metrics"][metric] = comparison.__dict__
    return result


def compute_rq3(rows: List[dataset.TedRow], done: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    """Sagot AI vs. REVIE, 60 matched pairs (Subset B), by qid."""
    b_rows = dataset.subset_b(rows)
    sagot_units = [done.get((r.qid, "sagot"), {"error": "not run"}) for r in b_rows]
    revie_units = [done.get((r.qid, "revie"), {"error": "not run"}) for r in b_rows]
    result = {"n_pairs_available": len(b_rows), "metrics": {}}
    for metric in METRICS:
        a, b = _metric_arrays(sagot_units, revie_units, metric)
        comparison = stats.paired_comparison(a, b, metric_name=f"{metric} (Sagot AI vs REVIE)")
        result["metrics"][metric] = comparison.__dict__
    result["contradictions"] = {
        side: _contradiction_count(units)
        for side, units in (("sagot", sagot_units), ("revie", revie_units))
    }
    return result


def _contradiction_count(units: List[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    """Answers that state at least one ground-truth fact WRONGLY — the direct,
    reference-based count of hallucinated answers. `judged` excludes units with
    no correctness score, so the rate is never computed over missing data."""
    judged = [u for u in units if "error" not in u and u.get("row", {}).get("answer_correctness") is not None]
    wrong = sum(1 for u in judged if (u["row"].get("n_facts_contradicted") or 0) > 0)
    return {"judged": len(judged), "with_contradiction": wrong}


# ── Output ──────────────────────────────────────────────────────────────────

def write_outputs(rows: List[dataset.TedRow], done: Dict[Tuple[str, str], Dict[str, Any]],
                   rq1: Optional[Dict], rq2: Dict, rq3: Optional[Dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    flat_rows = []
    for (qid, side), unit in sorted(done.items()):
        base = {"qid": qid, "side": side, "subset": unit.get("subset")}
        if "error" in unit:
            base["error"] = unit["error"]
        else:
            base.update(unit.get("row", {}))
            base["elapsed_s"] = unit.get("elapsed_s")
            if side == "sagot":
                base.update({f"sagot_{k}": v for k, v in (unit.get("sagot_meta") or {}).items()})
        flat_rows.append(base)

    if flat_rows:
        fieldnames = sorted({k for r in flat_rows for k in r.keys()})
        with open(RESULTS_DIR / "scores_flat.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)

    with open(RESULTS_DIR / "raw_scores.json", "w", encoding="utf-8") as f:
        json.dump([unit for _key, unit in sorted(done.items())], f, indent=2, ensure_ascii=False)

    summary = {
        "pipeline": _pipeline_settings(),
        "scoring_version": SCORING_VERSION,
        "n_rows_total": len(rows),
        "n_units_scored": len(done),
        "n_errors": sum(1 for u in done.values() if "error" in u),
        "rq1_english_vs_taglish": rq1,
        "rq2_language_detection": rq2,
        "rq3_sagot_vs_revie": rq3,
    }
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {RESULTS_DIR}/raw_scores.json, scores_flat.csv, summary.json")


def _pipeline_settings() -> Dict[str, Any]:
    """What produced these answers — so a results folder says which pipeline
    version (baseline vs. improved, reranker on/off, ...) it belongs to."""
    try:
        from app import llm, prompts, retrieval, version
        from app.embeddings import EMBEDDING_CONFIG
        return {
            "system_version": version.SYSTEM_VERSION,
            "reproduces": version.effective_label(),
            "generator": llm.GENERATOR_MODEL, "verifier": llm.VERIFIER_MODEL,
            "generation_temperature": llm.GENERATION_TEMPERATURE,
            "rag_prompt_version": prompts.RAG_PROMPT_VERSION,
            "retrieval": retrieval.retrieval_settings(), "index": EMBEDDING_CONFIG,
        }
    except Exception as e:  # never let bookkeeping break the write
        return {"error": f"{type(e).__name__}: {e}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", choices=["A", "B", "all"], default="all",
                         help="which rows to run Sagot AI (+ REVIE, for B) scoring on")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N rows (smoke test)")
    parser.add_argument("--fresh", action="store_true",
                         help="ignore any existing results/checkpoint.jsonl and start clean (default: resume, "
                              "skipping (qid, side) units already checkpointed)")
    parser.add_argument("--rq2-only", action="store_true", help="compute only RQ2 (no API calls, no cost)")
    parser.add_argument("--results-dir", default=None,
                         help="write checkpoint/results here instead of rag_eval/results (e.g. "
                              "rag_eval/results_v2) so a run of the improved pipeline never mixes with, "
                              "or overwrites, the baseline run")
    args = parser.parse_args()

    global RESULTS_DIR, CHECKPOINT_PATH
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir).resolve()
        CHECKPOINT_PATH = RESULTS_DIR / "checkpoint.jsonl"
        print(f"Results directory: {RESULTS_DIR}")

    # Wait out Jina rate limits instead of letting the reranker fall back to
    # plain top-k mid-run (an evaluated answer must come from the pipeline
    # being evaluated, not its outage fallback).
    from app import rerank as _rerank
    _rerank.use_evaluation_retries()

    rows = dataset.load()
    print(f"Loaded {len(rows)} T-TED rows.")

    if args.fresh and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    done = _load_checkpoint()
    print(f"{len(done)} (qid, side) units already checkpointed." if done else "No existing checkpoint.")

    if args.rq2_only:
        rq2 = compute_rq2(rows)
        write_outputs(rows, done, rq1=None, rq2=rq2, rq3=None)
        return

    preflight()

    if args.subset in ("A", "all"):
        a_rows = [r for r in rows if r.subset in (dataset.SUBSET_A_EN, dataset.SUBSET_A_TL)]
        if args.limit:
            a_rows = a_rows[: args.limit]
        print(f"Running Subset A (Sagot AI only): {len(a_rows)} rows.")
        for row in a_rows:
            print(f"  {row.qid} ...", end=" ", flush=True)
            unit = _process_row(row, "sagot", done)
            done[(row.qid, "sagot")] = unit
            print("ok" if "error" not in unit else "ERROR")

    if args.subset in ("B", "all"):
        b_rows = dataset.subset_b(rows)
        if args.limit:
            b_rows = b_rows[: args.limit]
        print(f"Running Subset B (Sagot AI + scoring REVIE's collected answers): {len(b_rows)} rows.")
        for row in b_rows:
            print(f"  {row.qid} (sagot) ...", end=" ", flush=True)
            unit = _process_row(row, "sagot", done)
            done[(row.qid, "sagot")] = unit
            print("ok" if "error" not in unit else "ERROR")
            print(f"  {row.qid} (revie) ...", end=" ", flush=True)
            unit = _process_row(row, "revie", done)
            done[(row.qid, "revie")] = unit
            print("ok" if "error" not in unit else "ERROR")

    rq2 = compute_rq2(rows)
    rq1 = compute_rq1(rows, done) if args.subset in ("A", "all") and not args.limit else None
    rq3 = compute_rq3(rows, done) if args.subset in ("B", "all") and not args.limit else None
    write_outputs(rows, done, rq1, rq2, rq3)


if __name__ == "__main__":
    main()
