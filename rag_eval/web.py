"""
The evaluation tool's web API and page — served by the SAME server as the
chatbot (main.py includes this router), at http://localhost:8000/eval.

Read-only endpoints (the page, the guide data, the T-TED dataset, the
thesis results, finished package runs) are open to anyone who can see the
chatbot. Endpoints under /eval/api/run... make PAID model calls (Sagot AI's
generator and verifier, the RAGAS judge, Jina) and are locked to the host
computer by app/guard.py (EVAL_LOCAL_ONLY, default on). The page asks
/eval/api/config first and explains the lock instead of offering a form a
visitor can't use.

Package runs execute in ONE background thread (only one at a time, so a
double-click can't double the bill); the page polls for progress. Each
finished question is saved to rag_eval/results/web_runs/<id>.json right
away, so a server restart never loses a run and old runs can be reopened.
"""

import csv
import io
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from app import guard

from . import dataset, judge, ragas_metrics, run_evaluation, service

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
WEB_RUNS_DIR = run_evaluation.RESULTS_DIR / "web_runs"
MAX_BATCH_ROWS = int(os.environ.get("EVAL_MAX_BATCH_ROWS", "200"))
MAX_UPLOAD_CHARS = 5_000_000
_JOB_ID_RE = re.compile(r"[0-9a-f]{8,32}")

# Rough per-question timings measured on this project (for the "estimated
# time" shown before a package run). Sagot AI: pipeline + 4 judge metrics.
SECONDS_PER_SAGOT = 25
SECONDS_PER_EXTERNAL = 10

router = APIRouter()


# ── Request models ──────────────────────────────────────────────────────────

class ItemIn(BaseModel):
    id: str = Field("", max_length=100)
    question: str = Field(..., min_length=1, max_length=2000)
    ground_truth: str = Field("", max_length=5000)
    expected_language: str = Field("", max_length=30)
    other_answer: str = Field("", max_length=10000)
    other_context: str = Field("", max_length=50000)


class RunRequest(BaseModel):
    mode: str
    other_name: str = Field("Other chatbot", max_length=60)
    items: List[ItemIn] = Field(..., min_length=1)


class ParseRequest(BaseModel):
    text: str = Field(..., max_length=MAX_UPLOAD_CHARS)
    filename: str = Field("", max_length=300)


def _inputs(req: RunRequest) -> List[service.EvalInput]:
    """Validate a run request into EvalInputs, reporting every problem at once
    (row numbers match the table the user sees) rather than the first one."""
    if req.mode not in service.MODES:
        raise HTTPException(422, f"Unknown mode {req.mode!r}.")
    problems, out = [], []
    for n, item in enumerate(req.items, start=1):
        label = item.id or f"row {n}"
        if not item.question.strip():
            problems.append(f"{label}: the question is empty.")
        try:
            lang = service.parse_expected_language(item.expected_language)
        except ValueError as e:
            problems.append(f"{label}: {e}")
            lang = None
        if req.mode != service.MODE_SAGOT and not item.other_answer.strip():
            problems.append(f"{label}: no answer from {req.other_name or 'the other chatbot'} was given.")
        out.append(service.EvalInput(
            question=item.question.strip(), ground_truth=item.ground_truth.strip(), expected_language=lang,
            other_answer=item.other_answer.strip(), other_context=item.other_context.strip(), id=item.id or str(n),
        ))
    if problems:
        raise HTTPException(422, {"message": "Please fix these rows first.", "problems": problems[:50]})
    return out


def _judge_status() -> Dict[str, Any]:
    try:
        return {"judge_model": judge.require_independent_judge(), "judge_error": None}
    except judge.JudgeConfigError as e:
        return {"judge_model": judge.JUDGE_MODEL or None, "judge_error": str(e)}


# ── Page + read-only endpoints ──────────────────────────────────────────────

@router.get("/eval", include_in_schema=False)
def eval_page():
    page = STATIC_DIR / "eval.html"
    if not page.exists():
        raise HTTPException(404, "static/eval.html is missing.")
    return FileResponse(page, media_type="text/html")


@router.get("/eval/api/config")
def config(request: Request):
    can_run = guard.can_run_evaluations(request)
    return {
        "can_run": can_run,
        "lock_reason": None if can_run else (
            "You are viewing this page through the public link. Running tests makes paid model calls, "
            "so it only works on the host computer at http://localhost:8000/eval. You can still read "
            "the guide and the thesis results."),
        **_judge_status(),
        "max_batch_rows": MAX_BATCH_ROWS,
        "seconds_per_sagot": SECONDS_PER_SAGOT,
        "seconds_per_external": SECONDS_PER_EXTERNAL,
        "min_trigger_recall": 0.90,
        "max_false_positive_rate": 0.10,
        "running_job": _running_job_id(),
    }


@router.get("/eval/api/dataset")
def ted_dataset():
    """The thesis T-TED dataset, in the tool's input format."""
    rows = dataset.load()
    return [{
        "id": r.qid,
        "subset": r.subset,
        "question": r.query,
        "ground_truth": r.expected_answer,
        "expected_language": "english" if dataset.language_ground_truth(r) == 0 else "taglish",
        "other_answer": r.revie_response,
        "other_context": "",
        "source": r.source,
    } for r in rows]


TEMPLATE_COLUMNS = ["id", "question", "ground_truth", "expected_language", "other_answer", "other_context"]


@router.get("/eval/api/template.csv")
def template_csv():
    rows = {r["id"]: r for r in ted_dataset()}
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=TEMPLATE_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for qid in ("A2-EN", "A2-TL", "B21"):
        if qid in rows:
            writer.writerow(rows[qid])
    # UTF-8 with BOM: Excel otherwise mangles Filipino characters and the ₱ sign.
    return Response("﻿" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="evaluation_template.csv"'})


# Header spellings a non-developer's spreadsheet might use -> our field names.
_HEADER_ALIASES = {
    "id": "id", "q#": "id", "qid": "id", "no": "id", "no.": "id", "number": "id",
    "question": "question", "query": "question", "questions": "question", "user_question": "question",
    "ground_truth": "ground_truth", "groundtruth": "ground_truth", "ground_truths": "ground_truth",
    "possible_correct_answer": "ground_truth", "correct_answer": "ground_truth",
    "expected_answer": "ground_truth", "reference": "ground_truth", "reference_answer": "ground_truth",
    "expected_language": "expected_language", "language": "expected_language", "lang": "expected_language",
    "other_answer": "other_answer", "answer": "other_answer", "response": "other_answer",
    "chatbot_answer": "other_answer", "chatbot_response": "other_answer", "revie_response": "other_answer",
    "revie_answer": "other_answer", "revie_response*": "other_answer",
    "other_context": "other_context", "context": "other_context", "contexts": "other_context",
}


def _norm_header(h: str) -> str:
    return re.sub(r"[\s\-]+", "_", (h or "").strip().lower().lstrip("﻿"))


@router.post("/eval/api/parse")
def parse_upload(req: ParseRequest):
    """Turn an uploaded CSV (or JSON) file into rows, with warnings a
    non-developer can act on. Free: no model calls."""
    text = req.text.lstrip("﻿")
    records: List[Dict[str, Any]]
    if req.filename.lower().endswith(".json") or text.lstrip().startswith("["):
        try:
            data = json.loads(text)
        except ValueError as e:
            raise HTTPException(422, f"That file is not valid JSON ({e}).")
        if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
            raise HTTPException(422, "A JSON file must be a list of objects, one per question.")
        records = data
    else:
        try:
            dialect = csv.Sniffer().sniff(text[:5000], delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        records = list(csv.DictReader(io.StringIO(text), dialect=dialect))

    if not records:
        raise HTTPException(422, "The file has no rows.")
    mapping, ignored = {}, []
    for header in records[0].keys():
        field = _HEADER_ALIASES.get(_norm_header(str(header)))
        if field and field not in mapping.values():
            mapping[header] = field
        elif header:
            ignored.append(str(header))
    if "question" not in mapping.values():
        raise HTTPException(422, "No question column found. The file needs a column named "
                                 "'question' — download the template to see the expected columns.")

    rows, warnings = [], []
    for n, rec in enumerate(records, start=1):
        row = {f: "" for f in TEMPLATE_COLUMNS}
        for header, field in mapping.items():
            value = rec.get(header)
            row[field] = "" if value is None else str(value).strip()
        if not row["question"]:
            continue  # blank spreadsheet line
        if not row["id"]:
            row["id"] = str(n)
        try:
            service.parse_expected_language(row["expected_language"])
        except ValueError:
            warnings.append(f"{row['id']}: expected_language {row['expected_language']!r} is not "
                            "'english' or 'taglish' — it will be ignored.")
            row["expected_language"] = ""
        rows.append(row)

    if len(rows) > MAX_BATCH_ROWS:
        warnings.append(f"Only the first {MAX_BATCH_ROWS} of {len(rows)} rows will be used "
                        "(EVAL_MAX_BATCH_ROWS).")
        rows = rows[:MAX_BATCH_ROWS]
    n_gt = sum(1 for r in rows if r["ground_truth"])
    if n_gt < len(rows):
        warnings.append(f"{len(rows) - n_gt} row(s) have no ground truth — Answer Correctness will be "
                        "blank for those.")
    if ignored:
        warnings.append("Ignored column(s): " + ", ".join(ignored[:10]))
    return {"rows": rows, "columns": sorted(set(mapping.values())), "warnings": warnings}


# ── Single question ─────────────────────────────────────────────────────────

@router.post("/eval/api/run/single")
def run_single(req: RunRequest):
    if len(req.items) != 1:
        raise HTTPException(422, "A single test takes exactly one question.")
    inp = _inputs(req)[0]
    if not req.items[0].id:
        inp.id = ""  # a one-off question has no row number to show
    status = _judge_status()
    if status["judge_error"]:
        raise HTTPException(400, status["judge_error"])
    return service.evaluate_item(inp, req.mode, req.other_name)


# ── Package runs (background jobs) ──────────────────────────────────────────

class _Job:
    def __init__(self, mode: str, other_name: str, inputs: List[service.EvalInput]):
        self.id = uuid.uuid4().hex[:12]
        self.mode, self.other_name, self.inputs = mode, other_name, inputs
        self.items: List[Dict[str, Any]] = []
        self.status = "running"            # running | done | cancelled | error
        self.error: Optional[str] = None
        self.current: Optional[str] = None
        self.created, self.finished = time.time(), None
        self.cancel = threading.Event()
        self.lock = threading.Lock()

    def header(self) -> Dict[str, Any]:
        return {"id": self.id, "mode": self.mode, "other_name": self.other_name, "status": self.status,
                "error": self.error, "total": len(self.inputs), "done": len(self.items),
                "current": self.current, "created": self.created, "finished": self.finished}

    def save(self) -> None:
        WEB_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        with self.lock:
            record = {**self.header(), "items": list(self.items),
                      "inputs": [vars(i) for i in self.inputs]}
        tmp = WEB_RUNS_DIR / f"{self.id}.json.tmp"
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        tmp.replace(WEB_RUNS_DIR / f"{self.id}.json")


_jobs: Dict[str, _Job] = {}
_jobs_lock = threading.Lock()


def _running_job_id() -> Optional[str]:
    with _jobs_lock:
        return next((j.id for j in _jobs.values() if j.status == "running"), None)


def _worker(job: _Job) -> None:
    try:
        for idx, inp in enumerate(job.inputs):
            if job.cancel.is_set():
                job.status = "cancelled"
                break
            job.current = inp.question
            result = service.evaluate_item(inp, job.mode, job.other_name)
            result["index"] = idx
            with job.lock:
                job.items.append(result)
            job.save()
        else:
            job.status = "done"
    except Exception as e:  # never leave a job stuck on "running"
        job.status, job.error = "error", f"{type(e).__name__}: {e}"
    job.current, job.finished = None, time.time()
    job.save()


@router.post("/eval/api/run/batch")
def run_batch(req: RunRequest):
    inputs = _inputs(req)
    if len(inputs) > MAX_BATCH_ROWS:
        raise HTTPException(422, f"A package can have at most {MAX_BATCH_ROWS} questions.")
    running = _running_job_id()
    if running:
        raise HTTPException(409, {"message": "A package test is already running. Wait for it to finish "
                                             "or cancel it first.", "job_id": running})
    # Preflight: one tiny judge call + one embedding, BEFORE any chatbot call,
    # so a misconfiguration costs a fraction of a cent, not a whole package.
    try:
        judge.preflight()
        from . import embeddings_eval
        embeddings_eval.embed_for_matching(["preflight check"])
    except Exception as e:
        raise HTTPException(400, f"Preflight failed, nothing was run: {type(e).__name__}: {e}")
    job = _Job(req.mode, req.other_name, inputs)
    with _jobs_lock:
        _jobs[job.id] = job
    job.save()
    threading.Thread(target=_worker, args=(job,), daemon=True, name=f"eval-{job.id}").start()
    return job.header()


@router.post("/eval/api/run/cancel/{job_id}")
def cancel_batch(job_id: str):
    job = _jobs.get(job_id)
    if not job or job.status != "running":
        raise HTTPException(404, "No running package test with that id.")
    job.cancel.set()
    return {"message": "Stopping after the current question."}


def _load_saved(job_id: str) -> Dict[str, Any]:
    if not _JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(404, "Unknown run.")
    path = WEB_RUNS_DIR / f"{job_id}.json"
    if not path.exists():
        raise HTTPException(404, "Unknown run.")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") == "running":  # the server stopped mid-run
        record["status"] = "interrupted"
    return record


@router.get("/eval/api/jobs/{job_id}")
def job_status(job_id: str, since: int = 0):
    """Progress + only the questions finished since `since` (the page polls,
    so resending every finished question each time would grow quadratically).
    The summary always covers everything finished so far."""
    job = _jobs.get(job_id)
    if job:
        with job.lock:
            items = list(job.items)
        header = job.header()
    else:
        record = _load_saved(job_id)
        items = record.pop("items", [])
        record.pop("inputs", None)
        header = {**record, "done": len(items), "current": None}
    return {**header, "items": items[max(0, since):], "summary": service.summarize(items, header["mode"])}


@router.get("/eval/api/runs")
def list_runs():
    runs = []
    if WEB_RUNS_DIR.exists():
        for path in sorted(WEB_RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
            try:
                record = _load_saved(path.stem)
            except Exception:
                continue
            live = _jobs.get(path.stem)
            runs.append({"id": record["id"], "mode": record["mode"], "other_name": record.get("other_name"),
                         "status": live.status if live else record["status"], "total": record["total"],
                         "done": len(record.get("items", [])), "created": record["created"]})
    return runs


# ── Thesis results (the offline run's checkpoint, computed live) ───────────

def _thesis_units():
    rows = dataset.load()
    done = run_evaluation._load_checkpoint()
    return rows, done


def _unit_metrics(unit: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if unit is None:
        return None
    if "error" in unit:
        return {"error": unit["error"]}
    row = unit.get("row", {})
    return {"metrics": {m: row.get(m) for m in service.METRICS},
            "contradicted": row.get("n_facts_contradicted"),
            "outcome": (unit.get("sagot_meta") or {}).get("outcome"),
            "scoring_version": unit.get("scoring_version", 1)}


@router.get("/eval/api/thesis")
def thesis_results():
    rows, done = _thesis_units()
    table = [{
        "qid": r.qid, "subset": r.subset, "question": r.query, "source": r.source,
        "sagot": _unit_metrics(done.get((r.qid, "sagot"))),
        "revie": _unit_metrics(done.get((r.qid, "revie"))),
    } for r in rows]
    ok = lambda side: sum(1 for (q, s), u in done.items() if s == side and "error" not in u)
    outdated = sum(1 for u in done.values() if "error" not in u
                   and u.get("scoring_version", 1) < run_evaluation.SCORING_VERSION)
    return {
        "progress": {"sagot_scored": ok("sagot"), "sagot_total": len(rows),
                     "revie_scored": ok("revie"), "revie_total": len(dataset.subset_b(rows)),
                     "outdated_scoring": outdated},
        "rq1": run_evaluation.compute_rq1(rows, done),
        "rq2": run_evaluation.compute_rq2(rows),
        "rq3": run_evaluation.compute_rq3(rows, done),
        "table": table,
    }


@router.get("/eval/api/thesis/{qid}")
def thesis_detail(qid: str):
    rows, done = _thesis_units()
    row = next((r for r in rows if r.qid == qid), None)
    if row is None:
        raise HTTPException(404, "Unknown question id.")
    sides = []
    for side, name in (("sagot", service.SAGOT_NAME), ("revie", "REVIE")):
        unit = done.get((qid, side))
        if unit is None:
            continue
        if "error" in unit:
            sides.append({"system": side, "system_name": name, "error": unit["error"]})
            continue
        scored = ragas_metrics.scored_from_dict(unit["scored"])
        system = service.MODE_SAGOT if side == "sagot" else service.MODE_EXTERNAL
        sides.append(service.side_from_scored(system, name, scored, unit.get("sagot_meta"), unit.get("elapsed_s")))
    inp = service.EvalInput(question=row.query, expected_language=dataset.language_ground_truth(row))
    return {"id": row.qid, "subset": row.subset, "question": row.query, "ground_truth": row.expected_answer,
            "source": row.source, "expected_language": inp.expected_language,
            "sides": sides, "language": service.language_check(inp)}
