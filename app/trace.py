"""
Per-question trace log (JSON Lines) — the raw data for the thesis evaluation.

Every /query writes ONE line to logs/queries.jsonl describing what happened:
the cleaned question, language, year filter, the retrieved excerpts (with
their text, since chunk text can change after a re-index), every generator
draft, every verifier verdict and critique, which host served each model
call, cost, latency, and the final outcome.

Why: the user-facing answer alone can't tell "the generator said NO_ANSWER"
from "the verifier rejected 3 drafts" — both show the same canned refusal.
And "hallucinations caught" can only be measured by a person judging the
REJECTED drafts against the excerpts, which requires keeping those drafts.

Privacy: this file contains visitors' questions (no IP addresses) and
unverified drafts. It stays on this computer; logs/ is in .gitignore. Never
publish it as-is.

Settings (env vars, optional):
    TRACE_ENABLED=1                 set 0 to stop writing traces
    TRACE_PATH=logs/queries.jsonl

Look at recent traces (project root):
    python -m app.trace                 # outcome counts, attempts, cost
    python -m app.trace --last 3        # the last 3 traces in full
"""

import argparse
import json
import os
import threading
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

TRACE_ENABLED: bool = os.environ.get("TRACE_ENABLED", "1").strip() not in ("0", "false", "False", "")
TRACE_PATH: str = os.environ.get("TRACE_PATH", os.path.join("logs", "queries.jsonl"))
TRACE_VERSION = 1  # bump if the record layout changes

_lock = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write(record: Dict[str, Any]) -> None:
    """Append one trace. Never raises: a logging problem must not break a query."""
    if not TRACE_ENABLED:
        return
    try:
        record = {"trace_version": TRACE_VERSION, **record}
        line = json.dumps(record, ensure_ascii=False, default=str)
        directory = os.path.dirname(TRACE_PATH)
        with _lock:
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(TRACE_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        print(f"⚠️  [Trace] could not write trace: {e}")


def read_all(path: str = TRACE_PATH) -> List[Dict[str, Any]]:
    """Every trace in the file (bad lines are skipped)."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    return records


def summarize(records: List[Dict[str, Any]]) -> str:
    if not records:
        return f"No traces yet in {TRACE_PATH}."
    outcomes = Counter(r.get("outcome", "?") for r in records)
    verified_attempts = Counter(
        r["rag"]["n_attempts"] for r in records
        if r.get("outcome") == "verified" and r.get("rag")
    )
    statuses = Counter(
        a.get("status", "?") for r in records if r.get("rag") for a in r["rag"].get("attempts", [])
    )
    cost = sum(r.get("cost") or 0 for r in records)
    lines = [f"{len(records)} trace(s) in {TRACE_PATH}", "", "Outcomes:"]
    lines += [f"  {k:22} {v}" for k, v in outcomes.most_common()]
    if verified_attempts:
        lines += ["", "Verified answers by attempt number:"]
        lines += [f"  attempt {k}: {v}" for k, v in sorted(verified_attempts.items())]
    if statuses:
        lines += ["", "All attempts by status:"]
        lines += [f"  {k:22} {v}" for k, v in statuses.most_common()]
    lines += ["", f"Reported model cost: ${cost:.5f}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize or show query traces.")
    parser.add_argument("--last", type=int, default=0, help="Print the last N traces in full.")
    parser.add_argument("--path", default=TRACE_PATH, help="Trace file (default: %(default)s).")
    args = parser.parse_args()
    records = read_all(args.path)
    if args.last:
        for r in records[-args.last:]:
            print(json.dumps(r, ensure_ascii=False, indent=2))
            print("-" * 60)
    else:
        print(summarize(records))


if __name__ == "__main__":
    main()
