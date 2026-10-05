"""
Before/after comparison of two evaluation runs of Sagot AI (e.g. the
baseline pipeline vs. the improved one) on the SAME T-TED questions.

Both runs must be scored with the same judge and the same SCORING_VERSION,
otherwise a difference could come from the measurement instead of the
chatbot (Context Relevance v2 and v3 are different measurements — see
judge.py). The script refuses to compare mismatched versions.

What it reports, per metric (Groundedness, Context Relevance, Answer
Relevance, Answer Correctness):
  * mean before / after, and the paired difference (after - before)
  * the same test the thesis already uses (stats.paired_comparison:
    Shapiro-Wilk on the differences -> paired t-test or Wilcoxon, with
    Cohen's d or rank-biserial r), over questions scored in BOTH runs
plus refusal counts, answers contradicting the ground truth, average
excerpts passed to the generator, and the per-subset means (A-EN, A-TL, B).

Writes compare.json and compare_per_question.csv into the AFTER folder; the
CSV lists every question's before/after scores so the biggest gains and
losses can be read one by one (sort it by a *_diff column).

Usage (project root):
    python -m rag_eval.compare_runs --before rag_eval/results --after rag_eval/results_v2
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from . import stats  # noqa: E402

METRICS = ("groundedness", "context_relevance", "answer_relevance", "answer_correctness")
SHORT = {"groundedness": "G", "context_relevance": "CR", "answer_relevance": "AR", "answer_correctness": "AC"}
REFUSAL_OUTCOMES = {"verification_failed", "generator_no_answer", "no_retrieval", "no_index"}
SUBSETS = {"A: Matched pairs (EN)": "A-EN", "A: Matched pairs (TL)": "A-TL", "B: Taglish queries": "B"}


def load_units(folder: Path) -> Dict[str, Dict[str, Any]]:
    """Sagot AI units from a run's checkpoint.jsonl, newest line per qid wins."""
    path = folder / "checkpoint.jsonl"
    if not path.exists():
        raise SystemExit(f"No checkpoint.jsonl in {folder} — run rag_eval.run_evaluation there first.")
    units: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("side") == "sagot":
                units[rec["qid"]] = rec
    return units


def _metric(unit: Optional[Dict[str, Any]], m: str) -> Optional[float]:
    if not unit or "error" in unit:
        return None
    return (unit.get("row") or {}).get(m)


def _outcome(unit: Optional[Dict[str, Any]]) -> Optional[str]:
    if not unit or "error" in unit:
        return None
    return (unit.get("sagot_meta") or {}).get("outcome") or (unit.get("sagot") or {}).get("outcome")


def _n_excerpts(unit: Optional[Dict[str, Any]]) -> Optional[int]:
    if not unit or "error" in unit:
        return None
    ctx = (unit.get("sagot") or {}).get("context") or (unit.get("scored") or {}).get("context") or ""
    return len([b for b in ctx.split("\n\n---\n\n") if b.strip()]) if ctx.strip() else 0


def _mean(xs: List[Optional[float]]) -> Optional[float]:
    v = [x for x in xs if x is not None]
    return sum(v) / len(v) if v else None


def _versions(units: Dict[str, Dict[str, Any]]) -> set:
    return {u.get("scoring_version", 1) for u in units.values() if "error" not in u}


def compare(before: Dict[str, Dict[str, Any]], after: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    qids = sorted(set(before) & set(after), key=lambda q: (q[0], int("".join(c for c in q.split("-")[0] if c.isdigit()) or 0), q))
    result: Dict[str, Any] = {"n_questions": len(qids), "metrics": {}, "by_subset": {}}

    for m in METRICS:
        a = [_metric(after.get(q), m) for q in qids]
        b = [_metric(before.get(q), m) for q in qids]
        result["metrics"][m] = stats.paired_comparison(a, b, metric_name=f"{m} (after vs before)").__dict__

    def refusals(units):
        outs = [_outcome(units.get(q)) for q in qids]
        known = [o for o in outs if o]
        return {"refused": sum(1 for o in known if o in REFUSAL_OUTCOMES), "of": len(known)}

    def contradictions(units):
        judged = [units[q] for q in qids if _metric(units.get(q), "answer_correctness") is not None]
        return {"with_contradiction": sum(1 for u in judged if (u["row"].get("n_facts_contradicted") or 0) > 0),
                "of": len(judged)}

    result["refusals"] = {"before": refusals(before), "after": refusals(after)}
    result["contradictions"] = {"before": contradictions(before), "after": contradictions(after)}
    result["mean_excerpts"] = {"before": _mean([_n_excerpts(before.get(q)) for q in qids]),
                               "after": _mean([_n_excerpts(after.get(q)) for q in qids])}

    for subset, short in SUBSETS.items():
        sq = [q for q in qids if (after[q].get("subset") or before[q].get("subset")) == subset]
        result["by_subset"][short] = {
            m: {"before": _mean([_metric(before.get(q), m) for q in sq]),
                "after": _mean([_metric(after.get(q), m) for q in sq]), "n": len(sq)}
            for m in METRICS
        }

    rows = []
    for q in qids:
        row: Dict[str, Any] = {"qid": q, "query": (after[q].get("row") or {}).get("query", ""),
                               "outcome_before": _outcome(before.get(q)), "outcome_after": _outcome(after.get(q)),
                               "excerpts_before": _n_excerpts(before.get(q)),
                               "excerpts_after": _n_excerpts(after.get(q))}
        for m in METRICS:
            x, y = _metric(before.get(q), m), _metric(after.get(q), m)
            row[f"{m}_before"], row[f"{m}_after"] = x, y
            row[f"{m}_diff"] = (y - x) if x is not None and y is not None else None
        rows.append(row)
    return result, rows


def _f(x: Optional[float], nd: int = 3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def print_report(result: Dict[str, Any]) -> None:
    print(f"\nQuestions compared: {result['n_questions']}\n")
    print(f"{'metric':20s} {'before':>8s} {'after':>8s} {'diff':>8s}   test                   p        effect")
    for m, c in result["metrics"].items():
        sig = "" if c["p_value"] is None else ("  *" if c["significant"] else "")
        eff = "—" if c["effect_size"] is None else f"{c['effect_size']:+.2f} {c['effect_size_label'] or ''}".strip()
        print(f"{m:20s} {_f(c['mean_b']):>8s} {_f(c['mean_a']):>8s} {_f(c['mean_diff'], 3):>8s}   "
              f"{(c['test_used'] or '—'):22s} {_f(c['p_value'], 4):8s}{sig:3s} {eff}  (n={c['n_pairs']})")
        if c.get("note"):
            print(f"{'':20s} note: {c['note']}")
    r, k, e = result["refusals"], result["contradictions"], result["mean_excerpts"]
    print(f"\nRefusals:                 before {r['before']['refused']}/{r['before']['of']}   "
          f"after {r['after']['refused']}/{r['after']['of']}")
    print(f"Contradicts ground truth: before {k['before']['with_contradiction']}/{k['before']['of']}   "
          f"after {k['after']['with_contradiction']}/{k['after']['of']}")
    print(f"Excerpts per answer:      before {_f(e['before'], 1)}   after {_f(e['after'], 1)}")
    print("\nBy subset (before → after)  G=Groundedness CR=Context Relevance AR=Answer Relevance AC=Answer Correctness:")
    for short, ms in result["by_subset"].items():
        parts = [f"{SHORT[m]} {_f(v['before'], 2)}→{_f(v['after'], 2)}" for m, v in ms.items()]
        print(f"  {short:5s} (n={list(ms.values())[0]['n']:2d})  " + "   ".join(parts))
    print("\n* = significant at alpha 0.05. diff = after - before (positive = improved).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", default="rag_eval/results", help="baseline run folder")
    parser.add_argument("--after", default="rag_eval/results_v2", help="improved run folder")
    args = parser.parse_args()
    before_dir, after_dir = Path(args.before), Path(args.after)
    before, after = load_units(before_dir), load_units(after_dir)

    vb, va = _versions(before), _versions(after)
    if vb != va or len(vb) != 1:
        raise SystemExit(
            f"Scoring versions differ (before: {sorted(vb)}, after: {sorted(va)}). Re-score the older run "
            f"first so both use the same judge:\n  python -m rag_eval.run_evaluation --subset all "
            f"--results-dir {before_dir}\n(this reuses its stored answers; the chatbot is not called again)"
        )

    result, rows = compare(before, after)
    result["before_dir"], result["after_dir"] = str(before_dir), str(after_dir)
    result["scoring_version"] = next(iter(va))
    print_report(result)

    (after_dir / "compare.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(after_dir / "compare_per_question.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["qid"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {after_dir / 'compare.json'} and {after_dir / 'compare_per_question.csv'}")


if __name__ == "__main__":
    main()
