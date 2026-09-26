"""
Query routing and labeling — rule-based, no trained model.

Two separate jobs, deliberately kept apart:

1. ROUTING (before retrieval): `get_year_filter()` pulls an explicit source
   year out of the query text with a regex. This is the only thing that
   ever influences retrieval, via a hard Chroma `where` filter.

2. LABELING (after retrieval): `label_from_docs()` reports which corpus
   categories (year folders / faq) the retrieved evidence actually came
   from, read straight from each chunk's `year` metadata. This is exactly
   accurate by construction — it describes the sources, so it can't
   disagree with them.

The previous TF-IDF + Naive Bayes classifier was removed: it was trained on
73 short examples, nearly all of which were issuance citations the regex
already handles perfectly, and its prediction was never used for anything
except the label shown in the UI.
"""

import re
from typing import List, Optional

from langchain_core.documents import Document

# Years that actually exist as folders/metadata in this corpus (see
# app/database.py's ingestion log). An explicit citation for a year
# outside this set can't be satisfied by anything in the DB, so it must
# NOT be turned into a Chroma filter — that would just guarantee a
# no-answer instead of letting hybrid retrieval try the rest of the corpus
# (e.g. a typo'd year, or a real citation from a year not yet ingested).
KNOWN_YEARS = {"2001", "2002", "2003", "2022", "2023", "2024", "2025", "2026"}

# Shown when nothing was retrieved / nothing supported an answer.
NO_SOURCES_LABEL = "No matching sources"

# Matches the "<sequence number>-<year>" suffix every BIR issuance number
# in this corpus uses: "1-2001", "14-2023", "34-2024", "001-2023", etc.
# Capped at 3 digits for the sequence number (the corpus's highest is
# "105-2023") specifically so it does NOT match a 4-digit year range like
# "2024-2025" (school year, fiscal year, a date range) — "2024" would
# overflow the {1,3} bound.
_ISSUANCE_YEAR_RE = re.compile(r"\b\d{1,3}[A-Za-z]?-((?:20)\d{2})\b")
_BARE_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def extract_explicit_year(query: str) -> Optional[str]:
    """
    Deterministically pull a source year out of the query text — no
    guessing, no model. Two tiers, checked in order:

    1. A formal issuance citation, e.g. "RMC 34-2024", "RR 2-2003",
       "RULING NO. 1-2001", "RDAO No. 22-2025" — the "<number>-<year>"
       pattern every issuance number in this corpus uses. Unambiguous by
       construction: if "14-2023" is in the query, the year is 2023.

    2. A bare year mention with nothing else competing for it, e.g.
       "2001 tax deadline", "nung 2023" — exactly one KNOWN_YEARS year
       appears anywhere in the query and no citation matched. It backs off
       to None the moment a SECOND distinct year also appears (e.g. "yung
       2025 ruling, still valid ba ngayong 2026?", or a range like
       "2024-2025") rather than picking one arbitrarily — unfiltered search
       still has both years available to it.

    Returns None — not a guess — when neither tier fires, or when a year
    is found but isn't in KNOWN_YEARS: a filter that can only ever match
    zero documents is strictly worse than no filter.
    """
    for m in _ISSUANCE_YEAR_RE.finditer(query):
        year = m.group(1)
        if year in KNOWN_YEARS:
            return year

    bare_years = {y for y in _BARE_YEAR_RE.findall(query) if y in KNOWN_YEARS}
    if len(bare_years) == 1:
        return next(iter(bare_years))
    return None


def get_year_filter(query: str) -> Optional[str]:
    """
    Routing step, run BEFORE retrieval. Returns the year to hard-filter
    Chroma on, or None to search the whole corpus.

    Only an explicit in-corpus citation or unambiguous bare year is ever
    enforced. A wrong hard filter excludes the correct chunks entirely, so
    when the query gives no explicit signal the search stays unfiltered.
    """
    year = extract_explicit_year(query)
    print(f"\n🔍 [Router] Query: '{query}'")
    if year:
        print(f"📅 [Year Filter Applied]: {year}  (explicit year in query)")
    else:
        print("📅 [Year Filter]: none — searching the whole corpus")
    return year


def _describe(categories: List[str]) -> str:
    if categories == ["faq"]:
        return "FAQ"
    years = [c for c in categories if c != "faq"]
    parts = []
    if years:
        noun = "Year" if len(years) == 1 else "Years"
        parts.append(f"BIR Tax Query (Source {noun}: {', '.join(years)})")
    if "faq" in categories:
        parts.append("FAQ")
    return " + ".join(parts)


def label_from_docs(docs: List[Document], max_categories: int = 3) -> str:
    """
    Labeling step, run AFTER retrieval. Summarizes which corpus categories
    the retrieved chunks came from, ordered by total RRF score (so the
    category carrying the most evidence comes first), e.g.:

        "BIR Tax Query (Source Year: 2024)"
        "BIR Tax Query (Source Years: 2023, 2024)"
        "FAQ"

    Reads each chunk's `year` metadata (the folder category stamped at
    ingestion) and the `_rrf_score` stamped by reciprocal_rank_fusion().
    Chunks missing a score count equally. Returns NO_SOURCES_LABEL when
    there are no chunks.
    """
    if not docs:
        return NO_SOURCES_LABEL

    scores: dict = {}
    for doc in docs:
        category = str(doc.metadata.get("year", "unknown"))
        scores[category] = scores.get(category, 0.0) + float(
            doc.metadata.get("_rrf_score", 1.0)
        )

    ranked = sorted(scores, key=lambda c: scores[c], reverse=True)[:max_categories]
    return _describe(ranked)