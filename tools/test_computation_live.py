"""
Live test of Sagot AI's tax computation against an INDEPENDENT reference calculator.

Why this exists
---------------
tests/test_conversation.py checks the computation code with its own expected values.
This script is a second, independent check for the thesis: the expected figures are
computed by a separate implementation written from the law (below, `Oracle`), not by
app/computation, and the chatbot is tested end to end through the real /query API,
including the multi-turn questions it asks.

How to run (no API keys or credits needed: computation never calls Jina or OpenRouter)
--------------------------------------------------------------------------------------
  1. In one terminal:   set RATE_LIMIT_PER_MIN=1000          (Windows; Linux: export ...)
                        uvicorn main:app --port 8000
  2. In another:        python tools/test_computation_live.py
     options:           --url http://localhost:8000   --only C1,C6   --verbose

Output
------
  * a PASS/FAIL table in the console
  * computation_test_results.csv   (one row per case, for the thesis appendix)
  * computation_test_transcripts.json (every turn sent and received, for debugging)

A case passes when the final outcome is one of the expected outcomes AND every
expected peso figure appears in the bot's reply. Figures are compared as numbers,
so "₱20,415", "₱20,415.00" and "20415" all match 20,415.

No rounding (v1.5.2): the oracle computes every figure exactly, like the bot.
A figure such as ₱868.74975 must appear with all its digits; a rounded
₱868.75 in the reply would fail the case.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.error
import urllib.request
from decimal import Decimal as D

# ---------------------------------------------------------------------------
# Independent reference calculator ("oracle")
# Sources: NIRC Sec. 24(A)(2)(a) as amended by RA 10963 (TRAIN); Sec. 24(A)(2)(b) 8% option;
# Sec. 32(B)(7)(e) ₱90,000 benefits cap; Sec. 32(B)(7)(f) contributions excluded;
# SSS Circular 2024-006; RA 11223 / PhilHealth Advisories 2025-0002, 2026-0042; HDMF Circ. 460.
# ---------------------------------------------------------------------------


class Oracle:
    # (over, base_tax, rate)
    TABLE_2018_2022 = [(0, 0, "0"), (250000, 0, "0.20"), (400000, 30000, "0.25"),
                       (800000, 130000, "0.30"), (2000000, 490000, "0.32"), (8000000, 2410000, "0.35")]
    TABLE_2023_ON = [(0, 0, "0"), (250000, 0, "0.15"), (400000, 22500, "0.20"),
                     (800000, 102500, "0.25"), (2000000, 402500, "0.30"), (8000000, 2202500, "0.35")]
    PERIODS = {"monthly": 12, "semi_monthly": 24, "weekly": 52, "annual": 1}

    @classmethod
    def graduated(cls, taxable, year: int) -> D:
        table = cls.TABLE_2018_2022 if year <= 2022 else cls.TABLE_2023_ON
        ti, tax = D(str(taxable)), D(0)
        for over, base, rate in table:
            if ti > over:
                tax = D(base) + (ti - over) * D(rate)
        return tax

    @staticmethod
    def eight_percent(gross_sales, non_operating=0) -> D | None:
        if D(str(gross_sales)) > 3000000:
            return None  # not eligible (above VAT threshold)
        base = D(str(gross_sales)) + D(str(non_operating)) - 250000
        return max(D(0), base * D("0.08"))

    @staticmethod
    def sss_msc(monthly: D) -> D:
        if monthly < 5250:
            return D(5000)
        if monthly >= 34750:
            return D(35000)
        return ((monthly - 4750) // 500) * 500 + 5000  # 5,250-5,749.99 -> 5,500, etc.

    @classmethod
    def contributions_month(cls, monthly) -> tuple[D, D, D]:
        m = D(str(monthly))
        sss = cls.sss_msc(m) * D("0.05")
        philhealth = min(max(m, D(10000)), D(100000)) * D("0.025")
        pagibig = min(m, D(10000)) * D("0.02")
        return sss, philhealth, pagibig

    @classmethod
    def employee(cls, pay, period="monthly", year=2025, benefits=0, withheld=None) -> dict:
        annual = D(str(pay)) * cls.PERIODS[period]
        sss, ph, pi = cls.contributions_month(annual / 12)
        contrib = (sss + ph + pi) * 12
        taxable = annual - contrib + max(D(0), D(str(benefits)) - 90000)
        tax = cls.graduated(taxable, year)
        out = {"annual": annual, "contributions": contrib, "taxable": taxable, "tax": tax}
        if withheld is not None:
            out["balance"] = abs(tax - D(str(withheld)))  # payable or refundable
        return out


# Sanity checks on the oracle itself (published example + bracket base taxes).
_e = Oracle.employee(35000)
assert (_e["contributions"], _e["taxable"], _e["tax"]) == (D(33900), D(386100), D(20415)), _e
for _tbl in (Oracle.TABLE_2018_2022, Oracle.TABLE_2023_ON):
    for (o1, b1, r1), (o2, b2, _) in zip(_tbl, _tbl[1:]):
        assert D(b1) + (o2 - o1) * D(r1) == b2, (o2, b2)

# ---------------------------------------------------------------------------
# Test cases
#   msg      : first message typed by the user
#   answers  : what to reply if the bot asks for a value (keyed by ComputationState.awaiting)
#   follow   : extra messages sent regardless (e.g. "cancel")
#   outcomes : acceptable final outcomes
#   numbers  : peso figures that must appear in the final reply
#   first_outcomes : (optional) acceptable outcomes for the bot's FIRST reply
# ---------------------------------------------------------------------------
DONE = {"computation_done"}
O = Oracle


def emp(*a, **k):
    e = O.employee(*a, **k)
    return [e["contributions"], e["taxable"]] + ([e["tax"]] if e["tax"] else [])


CASES = [
    # ---- A. Graduated table, taxable income given ---------------------------------
    dict(id="A1", group="Graduated", desc="2025, ₱250,000 (top of 0% bracket)",
         msg="Compute my income tax for 2025. My taxable income is ₱250,000.",
         outcomes=DONE, numbers=[], note="tax must be 0; check manually"),
    dict(id="A2", group="Graduated", desc="2025, ₱250,001 (first peso taxed)",
         msg="Compute my income tax for 2025. My taxable income is ₱250,001.",
         outcomes=DONE, numbers=[O.graduated(250001, 2025)]),
    dict(id="A3", group="Graduated", desc="2025, ₱400,000 (bracket boundary)",
         msg="Compute my income tax for 2025. My taxable income is ₱400,000.",
         outcomes=DONE, numbers=[O.graduated(400000, 2025)]),
    dict(id="A4", group="Graduated", desc="2025, ₱600,000",
         msg="Compute my income tax for 2025. My taxable income is ₱600,000.",
         outcomes=DONE, numbers=[O.graduated(600000, 2025)]),
    dict(id="A5", group="Graduated", desc="2025, ₱800,000 (bracket boundary)",
         msg="Compute my income tax for 2025. My taxable income is ₱800,000.",
         outcomes=DONE, numbers=[O.graduated(800000, 2025)]),
    dict(id="A6", group="Graduated", desc="2025, ₱2,000,000 (bracket boundary)",
         msg="Compute my income tax for 2025. My taxable income is ₱2,000,000.",
         outcomes=DONE, numbers=[O.graduated(2000000, 2025)]),
    dict(id="A7", group="Graduated", desc="2025, ₱10,000,000 (35% bracket)",
         msg="Compute my income tax for 2025. My taxable income is ₱10,000,000.",
         outcomes=DONE, numbers=[O.graduated(10000000, 2025)]),
    dict(id="A8", group="Graduated", desc="2022 table, ₱400,000",
         msg="Compute my income tax for 2022. My taxable income is ₱400,000.",
         outcomes=DONE, numbers=[O.graduated(400000, 2022)]),
    dict(id="A9", group="Graduated", desc="2022 table, ₱600,000 (compare A4)",
         msg="Compute my income tax for 2022. My taxable income is ₱600,000.",
         outcomes=DONE, numbers=[O.graduated(600000, 2022)]),
    dict(id="A10", group="Graduated", desc="2022 table, ₱10,000,000",
         msg="Compute my income tax for 2022. My taxable income is ₱10,000,000.",
         outcomes=DONE, numbers=[O.graduated(10000000, 2022)]),
    dict(id="A11", group="Graduated", desc="2018 (first TRAIN year), ₱386,100",
         msg="Compute my income tax for 2018. My taxable income is ₱386,100.",
         outcomes=DONE, numbers=[O.graduated(386100, 2018)]),
    dict(id="A12", group="Graduated", desc="2026 (last supported year), ₱386,100",
         msg="Compute my income tax for 2026. My taxable income is ₱386,100.",
         outcomes=DONE, numbers=[O.graduated(386100, 2026)]),

    # ---- B. 8% option ---------------------------------------------------------------
    dict(id="B1", group="8% option", desc="Gross sales ₱1,000,000",
         msg="I am self-employed. Compute my income tax for 2025 under the 8% option. My gross sales are ₱1,000,000.",
         outcomes=DONE, numbers=[O.eight_percent(1000000)]),
    dict(id="B2", group="8% option", desc="Sales ₱900,000 + other income ₱100,000",
         msg="I am self-employed. Compute my income tax for 2025 under the 8% option. My gross sales are ₱900,000 and my non-operating income is ₱100,000.",
         outcomes=DONE, numbers=[O.eight_percent(900000, 100000)]),
    dict(id="B3", group="8% option", desc="Gross sales exactly ₱3,000,000 (still eligible)",
         msg="I am self-employed. Compute my income tax for 2025 under the 8% option. My gross sales are ₱3,000,000.",
         outcomes=DONE, numbers=[O.eight_percent(3000000)]),
    dict(id="B4", group="8% option", desc="Gross sales ₱3,500,000 (not eligible)",
         msg="I am self-employed. Compute my income tax for 2025 under the 8% option. My gross sales are ₱3,500,000.",
         outcomes={"computation_not_eligible"}, numbers=[]),
    dict(id="B5", group="8% option", desc="Taglish phrasing, same as B1",
         msg="Self-employed ako. Pa-compute ng income tax ko for 2025, 8% option, gross sales ko ay ₱1,000,000.",
         outcomes=DONE, numbers=[O.eight_percent(1000000)]),

    # ---- C. Employee, from gross pay -----------------------------------------------
    dict(id="C1", group="Employee", desc="₱35,000/month 2025 (published example)",
         msg="Compute my income tax for 2025. I am an employee and my monthly salary is ₱35,000.",
         outcomes=DONE, numbers=emp(35000)),
    dict(id="C2", group="Employee", desc="₱20,000/month (tax = 0)",
         msg="Compute my income tax for 2025. I am an employee and my monthly salary is ₱20,000.",
         outcomes=DONE, numbers=emp(20000), note="tax must be 0; check manually"),
    dict(id="C3", group="Employee", desc="₱50,000/month (SSS at cap)",
         msg="Compute my income tax for 2025. I am an employee and my monthly salary is ₱50,000.",
         outcomes=DONE, numbers=emp(50000)),
    dict(id="C4", group="Employee", desc="₱100,000/month (PhilHealth at ceiling)",
         msg="Compute my income tax for 2025. I am an employee and my monthly salary is ₱100,000.",
         outcomes=DONE, numbers=emp(100000)),
    dict(id="C5", group="Employee", desc="₱150,000/month (above PhilHealth ceiling)",
         msg="Compute my income tax for 2025. I am an employee and my monthly salary is ₱150,000.",
         outcomes=DONE, numbers=emp(150000)),
    dict(id="C6", group="Employee", desc="₱25,240/month (SSS MSC ₱25,000)",
         msg="Compute my income tax for 2025. I am an employee and my monthly salary is ₱25,240.",
         outcomes=DONE, numbers=emp(25240)),
    dict(id="C7", group="Employee", desc="₱25,250/month (SSS MSC steps to ₱25,500)",
         msg="Compute my income tax for 2025. I am an employee and my monthly salary is ₱25,250.",
         outcomes=DONE, numbers=emp(25250)),
    dict(id="C8", group="Employee", desc="₱17,500 semi-monthly (must equal C1)",
         msg="Compute my income tax for 2025. I am an employee and my semi-monthly salary is ₱17,500.",
         outcomes=DONE, numbers=emp(17500, "semi_monthly")),
    dict(id="C9", group="Employee", desc="C1 + ₱120,000 13th month/benefits (P30,000 taxable)",
         msg="Compute my income tax for 2025. I am an employee, my monthly salary is ₱35,000 and my 13th month pay and other benefits total ₱120,000.",
         outcomes=DONE, numbers=emp(35000, benefits=120000)),
    dict(id="C10", group="Employee", desc="C1 with ₱18,000 withheld (still payable)",
         msg="Compute my income tax for 2025. I am an employee, my monthly salary is ₱35,000 and my employer withheld ₱18,000.",
         outcomes=DONE, numbers=[O.employee(35000, withheld=18000)["tax"], O.employee(35000, withheld=18000)["balance"]]),
    dict(id="C11", group="Employee", desc="C1 with ₱25,000 withheld (over-withheld)",
         msg="Compute my income tax for 2025. I am an employee, my monthly salary is ₱35,000 and my employer withheld ₱25,000.",
         outcomes=DONE, numbers=[O.employee(35000, withheld=25000)["tax"], O.employee(35000, withheld=25000)["balance"]]),
    dict(id="C12", group="Employee", desc="₱35,000/month in 2026 (2026 contribution rules)",
         msg="Compute my income tax for 2026. I am an employee and my monthly salary is ₱35,000.",
         outcomes=DONE, numbers=emp(35000, year=2026)),
    dict(id="C14", group="Employee", desc="₱34,749.99/month (PhilHealth ₱868.74975, not rounded)",
         msg="Compute my income tax for 2025. I am an employee and my monthly salary is ₱34,749.99.",
         outcomes=DONE, numbers=emp(D("34749.99"))),
    dict(id="C13", group="Employee", desc="Filipino phrasing, same as C1",
         msg="Pakikalkula ang income tax ko para sa 2025. Empleyado ako at ang buwanang sahod ko ay ₱35,000.",
         outcomes=DONE, numbers=emp(35000)),

    # ---- D. Safety behaviour (must refuse or ask, never guess) -----------------------
    dict(id="D1", group="Safety", desc="2017: unsupported year, must not borrow 2018",
         msg="Compute my income tax for 2017. My taxable income is ₱500,000.",
         outcomes={"computation_unsupported_year"}, numbers=[], answers=None),
    dict(id="D2", group="Safety", desc="2027: unsupported year",
         msg="Compute my income tax for 2027. My taxable income is ₱500,000.",
         outcomes={"computation_unsupported_year"}, numbers=[], answers=None),
    dict(id="D3", group="Safety", desc="VAT: no rule, must not compute",
         msg="Compute my VAT for 2025. My sales are ₱1,000,000.",
         outcomes={"computation_no_rule"}, numbers=[], answers=None),
    dict(id="D4", group="Safety", desc="No year given: must ask",
         msg="Compute my income tax. My taxable income is ₱600,000.",
         outcomes={"computation_needs_input"}, numbers=[], answers=None),
    dict(id="D5", group="Safety", desc="Employee 2024: asks for contributions, then computes",
         msg="Compute my income tax for 2024. I am an employee and my monthly salary is ₱35,000.",
         answers={"contributions": "My total contributions for the year are ₱33,900."},
         first_outcomes={"computation_needs_input"},
         outcomes=DONE, numbers=[D(386100), O.graduated(386100, 2024)]),
    dict(id="D6", group="Safety", desc="₱1,200/month: must ask for contributions",
         msg="Compute my income tax for 2025. I am an employee and my monthly salary is ₱1,200.",
         outcomes={"computation_needs_input"}, numbers=[], answers=None),
    dict(id="D7", group="Safety", desc="Amount in words: must ask again",
         msg="Compute my income tax for 2025. My taxable income is six hundred thousand pesos.",
         outcomes={"computation_needs_input", "computation_invalid_input"}, numbers=[], answers=None),
    dict(id="D8", group="Safety", desc="User cancels mid-computation",
         msg="Compute my income tax for 2025.", answers=None, follow=["cancel"],
         outcomes={"computation_cancelled"}, numbers=[]),
]

# Default replies when the bot asks for a value the first message did not contain.
DEFAULT_ANSWERS = {
    "tax_year": "2025",
    "kind": "employee",
    "regime": "graduated",
    "pay_period": "monthly",
    "benefits": "₱0",
    "tax_withheld": "₱0",
    "non_operating_income": "₱0",
}

# ---------------------------------------------------------------------------
NUM_RE = re.compile(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\d.,])\d+(?:\.\d+)?")


def numbers_in(text: str) -> set[D]:
    return {D(m.replace(",", "")) for m in NUM_RE.findall(text or "")}


def reply_text(resp: dict) -> str:
    for key in ("answer", "response", "message", "text", "reply"):
        if isinstance(resp.get(key), str):
            return resp[key]
    return json.dumps(resp, ensure_ascii=False)


def outcome_of(resp: dict) -> str:
    for key in ("outcome", "status"):
        if isinstance(resp.get(key), str):
            return resp[key]
    trace = resp.get("trace") or {}
    return trace.get("outcome") or resp.get("mode") or "?"


def post(url: str, query: str, state, _retries: int = 3):
    body = json.dumps({"query": query, "history": [], "bypass_cache": True,
                       "computation_state": state}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/query", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 429 and _retries > 0:
            # The server's per-minute limit (RATE_LIMIT_PER_MIN, default 6). Start the
            # server with RATE_LIMIT_PER_MIN=1000 to avoid this; otherwise wait it out.
            wait = int(e.headers.get("Retry-After") or 30) + 1
            print(f"      (rate limit: waiting {wait}s — start the server with RATE_LIMIT_PER_MIN=1000 to skip this)")
            import time
            time.sleep(wait)
            return post(url, query, state, _retries - 1)
        return {"answer": f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}", "outcome": "http_error"}
    except (urllib.error.URLError, OSError) as e:  # server down, crashed mid-request, timeout
        return {"answer": f"connection error: {e}", "outcome": "http_error"}


def run_case(url: str, case: dict, max_turns: int = 6):
    turns, state = [], None
    answers = DEFAULT_ANSWERS | (case.get("answers") or {}) if case.get("answers", {}) is not None else {}
    queue = [case["msg"]] + list(case.get("follow", []))
    asked_unknown = None
    while queue and len(turns) < max_turns:
        msg = queue.pop(0)
        resp = post(url, msg, state)
        state = resp.get("computation")
        turns.append({"sent": msg, "outcome": outcome_of(resp), "mode": resp.get("mode"),
                      "reply": reply_text(resp), "computation": state})
        if queue or not isinstance(state, dict) or state.get("status") != "collecting":
            continue
        awaiting = state.get("awaiting") or []
        if not answers or not awaiting:
            break
        key = awaiting[0]
        if key not in answers:
            asked_unknown = key
            break
        queue.append(answers[key])

    final = turns[-1]
    got = numbers_in(final["reply"])
    missing = [n for n in case["numbers"] if D(n) not in got]
    ok_outcome = final["outcome"] in case["outcomes"]
    first_ok = "first_outcomes" not in case or turns[0]["outcome"] in case["first_outcomes"]
    passed = ok_outcome and first_ok and not missing and turns[0]["outcome"] != "http_error"
    why = []
    if not ok_outcome:
        why.append(f"outcome {final['outcome']} (expected {'/'.join(sorted(case['outcomes']))})")
    if not first_ok:
        why.append(f"first reply was {turns[0]['outcome']} (expected it to ask: "
                   f"{'/'.join(sorted(case['first_outcomes']))})")
    if missing:
        why.append("missing " + ", ".join(f"{D(n):,.2f}" for n in missing))
    if asked_unknown:
        why.append(f"bot asked for '{asked_unknown}' (no scripted answer)")
    return passed, "; ".join(why), turns


def main():
    try:
        sys.stdout.reconfigure(errors="replace")  # Windows consoles without UTF-8
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--only", help="comma-separated case ids, e.g. C1,C6")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cases = CASES
    if args.only:
        wanted = {x.strip().upper() for x in args.only.split(",")}
        cases = [c for c in CASES if c["id"] in wanted]

    rows, transcripts, passed_n = [], {}, 0
    for c in cases:
        ok, why, turns = run_case(args.url, c)
        passed_n += ok
        transcripts[c["id"]] = turns
        expected = ", ".join(f"{D(n):,.2f}" for n in c["numbers"]) or "-"
        print(f"{'PASS' if ok else 'FAIL'}  {c['id']:<4} {c['desc']:<55} turns={len(turns)}  {why}")
        if c.get("note"):
            print(f"      note: {c['note']}")
        if args.verbose or not ok:
            for t in turns:
                print(f"      > {t['sent']}\n      < [{t['outcome']}] {t['reply'][:400]!r}")
        rows.append({"id": c["id"], "group": c["group"], "case": c["desc"], "input": c["msg"],
                     "expected_outcome": "/".join(sorted(c["outcomes"])), "expected_figures": expected,
                     "actual_outcome": turns[-1]["outcome"], "turns": len(turns),
                     "result": "PASS" if ok else "FAIL", "detail": why, "note": c.get("note", "")})

    with open("computation_test_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open("computation_test_transcripts.json", "w", encoding="utf-8") as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{passed_n}/{len(cases)} passed. Wrote computation_test_results.csv and computation_test_transcripts.json")
    sys.exit(0 if passed_n == len(cases) else 1)


if __name__ == "__main__":
    main()
