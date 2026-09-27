"""
The T-TED dataset (Taglish-Tax Evaluation Dataset) — loaded from
ted_dataset.json, extracted from the user's thesisFinal_3.docx and already
validated once (100 rows, no duplicate qids, no empty fields, every
A{i}-EN/A{i}-TL pair shares a source). This module just gives the rest of
rag_eval/ a typed, validated way to read it.

Layout, per the SOP:
    Subset A  "A: Matched pairs (EN)" / "A: Matched pairs (TL)"   20+20 rows
              RQ1: does Sagot AI perform the same on an English question
              as on its Taglish translation? Matched by pair index
              (A1-EN <-> A1-TL, ... A20-EN <-> A20-TL).
              Also RQ2's ground truth: -EN rows are known-English,
              -TL rows are known-non-English (see LANGUAGE NOTE below).
    Subset B  "B: Taglish queries"                                60 rows
              RQ3: Sagot AI vs. REVIE, matched by qid (B1..B60), each row
              carrying REVIE's already-collected `revie_response`.

LANGUAGE NOTE (found during validation, worth keeping visible in code, not
just in a chat message that scrolls away): running the production
`app.language.detect_language()` over every -TL/B query shows 77 of the 80
non-English queries land in the "filipino" bucket, not "taglish" — the
dataset's own "Taglish" label is being used the way the SOP and defense
conversation use it (the localized / code-switched-with-English-tax-jargon
question set, as opposed to the pure-English Subset A/-EN rows), not the
classifier's narrower three-way distinction (english / filipino / taglish)
that decides which fixed-message voice a reply uses. RQ2's ground truth
here is therefore built as BINARY — 0 = the classifier said "english",
1 = anything else — matching stats.confusion_metrics()'s own 1=Taglish/
0=Non-Taglish convention and the "TN = English correctly left alone"
framing in its docstring. If the thesis write-up wants the finer three-way
split called out explicitly, `language_ground_truth()` below is the
one place to change.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

DATASET_PATH = Path(__file__).parent / "ted_dataset.json"

SUBSET_A_EN = "A: Matched pairs (EN)"
SUBSET_A_TL = "A: Matched pairs (TL)"
SUBSET_B = "B: Taglish queries"


@dataclass(frozen=True)
class TedRow:
    qid: str
    subset: str
    query: str
    expected_answer: str
    revie_response: str
    source: str


def load(path: Optional[Path] = None) -> List[TedRow]:
    """Reads `path`, or the module-level DATASET_PATH if omitted — resolved
    at CALL time (not baked in as a default), so a test can point
    dataset.DATASET_PATH at a fixture file and this picks it up without
    needing to pass path explicitly everywhere."""
    with open(path or DATASET_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    rows = [TedRow(**r) for r in raw]
    _validate(rows)
    return rows


def _validate(rows: List[TedRow]) -> None:
    qids = [r.qid for r in rows]
    if len(qids) != len(set(qids)):
        dupes = {q for q in qids if qids.count(q) > 1}
        raise ValueError(f"duplicate qid(s) in {DATASET_PATH.name}: {sorted(dupes)}")
    for r in rows:
        if not r.query.strip():
            raise ValueError(f"{r.qid}: empty query")


def subset_a_pairs(rows: List[TedRow]) -> List[Tuple[TedRow, TedRow]]:
    """20 (EN, TL) row pairs, matched by pair index (A1-EN with A1-TL, ...)."""
    by_index = {}
    for r in rows:
        if r.subset not in (SUBSET_A_EN, SUBSET_A_TL):
            continue
        idx = r.qid.split("-")[0]  # "A1" from "A1-EN" / "A1-TL"
        by_index.setdefault(idx, {})[r.subset] = r
    pairs = []
    for idx in sorted(by_index, key=lambda s: int(s[1:])):
        pair = by_index[idx]
        if SUBSET_A_EN in pair and SUBSET_A_TL in pair:
            pairs.append((pair[SUBSET_A_EN], pair[SUBSET_A_TL]))
    return pairs


def subset_b(rows: List[TedRow]) -> List[TedRow]:
    return [r for r in rows if r.subset == SUBSET_B]


def language_ground_truth(row: TedRow) -> int:
    """1 = expected non-English (Taglish, per this dataset's own labeling),
    0 = expected English. See the LANGUAGE NOTE above for why -TL/B rows
    are 1 even though most of them classify as "filipino" rather than
    "taglish" under the finer three-way production label."""
    if row.subset == SUBSET_A_EN:
        return 0
    return 1  # SUBSET_A_TL or SUBSET_B
