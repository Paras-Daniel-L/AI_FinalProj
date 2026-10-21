"""
Deterministic grounding check and claim repair (system v1.6).

The LLM verifier (app/llm.py) judges whether a draft is supported by the
excerpts. This module runs FIRST, with plain rules and no model call, on the
claims that are easiest to get wrong and easiest to check mechanically:

    peso amounts      "₱3,000,000", "P500,000.00", "₱3.431 trillion"
    percentages       "12%", "20 percent", "1 porsyento"
    issuance numbers  "RMC No. 37-2024", "RR 8-2024" (app/textproc.parse_issuance_ids)
    Republic Acts     "RA 11976", "Republic Act No. 10963"
    BIR form numbers  "BIR Form No. 1701A", "Form 2316"
    other numbers     2+ digit figures: days, dates, years, section numbers
    e-mail / web      "tin.inquiry@bir.gov.ph", "www.bir.gov.ph"

Rules, per sentence of the draft:

  1. Every such token must appear in an excerpt the sentence cites. A token
     found only in a DIFFERENT excerpt is a wrong citation; a token found in
     no excerpt at all is unsupported (e.g. "RA 12023" that no excerpt names).
  2. A sentence's citations are its own [n], or those of a later sentence on
     the same line (the usual "X. Y [1]." style). An uncited fact that IS in
     some excerpt (a bullet under a cited lead-in) is left to the verifier;
     one that is in no excerpt is flagged.
  3. A citation number must exist ([4] when there are 3 excerpts is wrong).
  4. A figure the user's own question contains ("my gross pay is
     ₱360,000") may be repeated.

On the 390 distinct verified answers in logs/queries.jsonl (2026-10-02 to
10-08) this flags 14, all real: 9 name "RA No. 11590" and 1 names "RA No.
11976", neither of which any excerpt contains (the LLM verifier passed all
ten); 1 computes a figure itself (₱240,000 = 12 x ₱20,000); 3 cite the wrong
excerpt.

Numbers are compared by VALUE, so "₱3,000,000.00" matches "P3,000,000",
"3 million" matches "3,000,000", and "20%" matches "twenty percent (20%)".
Written-out English numbers in the excerpts ("thirty (30) days", "ten days")
count. Single-digit plain numbers ("2 copies", "Step 1") are not checked:
they are too often list numbering or ordinals.

Claim repair
------------
When the check (or the LLM verifier) flags specific sentences, deleting
those sentences often leaves a complete, supported answer. `repair()` does
that, but only when the flagged sentences are NOT the opening answer (the
first sentence, plus the second if the first is just "Yes."/"Oo.") and at
least one cited sentence remains. The repaired draft is then verified again
by the LLM verifier, so nothing unverified is ever shown. Otherwise the draft
is regenerated as before, with the issues as the audit notice.

Switches: GROUNDING_CHECK=0 and CLAIM_REPAIR=0 restore v1.5.2.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Sequence, Set, Tuple

from dotenv import load_dotenv

from .textproc import parse_issuance_ids

load_dotenv()

GROUNDING_CHECK: bool = os.environ.get("GROUNDING_CHECK", "1").strip() in ("1", "true", "True", "yes")
CLAIM_REPAIR: bool = os.environ.get("CLAIM_REPAIR", "1").strip() in ("1", "true", "True", "yes")

# ── Context parsing ──────────────────────────────────────────────────────

_BLOCK_SEP = "\n\n---\n\n"
_BLOCK_HEAD_RE = re.compile(r"^\[(\d+)\]")


# What a label adds that is not document content: "[1] ", ", p.3", " (2024)", " · OCR".
_LABEL_NOISE_RE = re.compile(r"^\[\d+\]\s*|,\s*p\.\d+|\s*\([^)]*\)\s*$|\s*·\s*OCR\s*$")


def parse_context(context_text: str) -> Dict[int, str]:
    """Excerpt number -> the issuance title + the excerpt text, from
    app/retrieval.format_context(). The label's excerpt number, page and
    folder year are dropped: they are not document content, and "[1]" would
    otherwise "support" any figure 1."""
    out: Dict[int, str] = {}
    for block in (context_text or "").split(_BLOCK_SEP):
        block = block.strip()
        m = _BLOCK_HEAD_RE.match(block)
        if m:
            label, _, text = block.partition("\n")
            title = label
            for _ in range(3):
                title = _LABEL_NOISE_RE.sub("", title).strip()
            out[int(m.group(1))] = f"{title}\n{text}"
    return out


# ── Numbers ──────────────────────────────────────────────────────────────

_MULT = {"k": 10**3, "thousand": 10**3, "libo": 10**3, "m": 10**6, "million": 10**6, "milyon": 10**6,
         "b": 10**9, "billion": 10**9, "bilyon": 10**9, "t": 10**12, "trillion": 10**12, "trilyon": 10**12}
_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_NUM_RE = re.compile(rf"(?<![\d.,/])({_NUM})(?:\s*((?i:thousand|million|billion|trillion|milyon|bilyon|trilyon|libo)|[kKmMbBtT])\b)?")

_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_WORD_NUM_RE = re.compile(
    r"\b(" + "|".join(_TENS) + r")(?:[-\s](" + "|".join(k for k in _UNITS if 0 < _UNITS[k] < 10) + r"))?\b|\b("
    + "|".join(_UNITS) + r")\b",
    re.IGNORECASE,
)


def _dec(raw: str) -> Optional[Decimal]:
    try:
        return Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return None


def _norm(d: Decimal) -> Decimal:
    """Decimal compares (and hashes) by value, so 1, 1.0 and 1.00 already
    match; only trailing zeros are dropped for a readable trace."""
    return d.normalize() if d == d.to_integral_value() else d


def numbers_in(text: str) -> Set[Decimal]:
    """Every number value in an excerpt: digits (with any million/billion
    multiplier, both with and without it) and written-out English numbers."""
    vals: Set[Decimal] = set()
    for m in _NUM_RE.finditer(text or ""):
        d = _dec(m.group(1))
        if d is None:
            continue
        vals.add(_norm(d))
        if m.group(2):
            vals.add(_norm(d * _MULT[m.group(2).lower()]))
    for m in _WORD_NUM_RE.finditer(text or ""):
        if m.group(1):
            n = _TENS[m.group(1).lower()] + (_UNITS[m.group(2).lower()] if m.group(2) else 0)
        else:
            n = _UNITS[m.group(3).lower()]
        vals.add(Decimal(n))
    return vals


# ── Fact tokens in the draft ─────────────────────────────────────────────

@dataclass
class Token:
    kind: str          # peso | percent | issuance | ra | form | number | contact
    raw: str
    value: str         # comparable value (a number string, an issuance id, digits, lowercase text)


_CITE_RE = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
_PESO_RE = re.compile(rf"(?:₱|\bPHP\s?|\bPhp\s?|\bP(?=\s?\d))\s?({_NUM})(?:\s*((?i:thousand|million|billion|trillion|milyon|bilyon|trilyon)|[kKmMbBtT])\b)?")
_PCT_RE = re.compile(rf"(?<![\d.,])({_NUM})\s?(?:%|percent\b|per\s?cent\b|porsyento\b|porsiyento\b|pursyento\b)", re.I)
_RA_RE = re.compile(r"\b(?:R\.?\s?A\.?|Republic\s+Act)\s*(?:No\.?\s*)?(\d{3,5})\b", re.I)
_FORM_RE = re.compile(r"\bForm\s*(?:No\.?\s*)?(\d{4}[A-Za-z]{0,3}(?:-[A-Za-z0-9]{1,3})?)\b", re.I)
_CONTACT_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|\b(?:https?://|www\.)[^\s)\]]+", re.I)
_ISSUANCE_SPAN_RE = re.compile(
    r"(?<![A-Za-z])(?:RDAO|RAMO|RMC|RMO|RMR|RAO|RR|Revenue\s+\w+(?:\s+\w+){0,2})\.?\s*(?:\((?:RDAO|RAMO|RMC|RMO|RMR|RAO|RR)\))?\s*"
    r"(?:No\.?|Nos\.?|Number)?\s*\d{1,4}[A-Za-z]?\s*-\s*(?:19|20)\d{2}(?!\d)",
    re.I,
)


def citations(sentence: str) -> List[int]:
    out: List[int] = []
    for m in _CITE_RE.finditer(sentence or ""):
        for part in re.split(r"\s*,\s*", m.group(1)):
            if re.search(r"[–-]", part):
                a, b = [int(x) for x in re.split(r"\s*[–-]\s*", part)[:2]]
                out += list(range(min(a, b), max(a, b) + 1)) if abs(b - a) <= 10 else [a, b]
            else:
                out.append(int(part))
    return out


def fact_tokens(sentence: str) -> List[Token]:
    """The checkable facts in one sentence of the draft (citations removed)."""
    text = _CITE_RE.sub(" ", re.sub(r"[*`]", "", sentence or ""))
    toks: List[Token] = []
    spans: List[Tuple[int, int]] = []

    def take(m, kind: str, value: str) -> None:
        toks.append(Token(kind, m.group(0).strip(), value))
        spans.append(m.span())

    for m in _CONTACT_RE.finditer(text):
        take(m, "contact", m.group(0).lower().rstrip(".,;"))
    for ident in parse_issuance_ids(text):
        toks.append(Token("issuance", ident, ident))
    for m in _ISSUANCE_SPAN_RE.finditer(text):
        spans.append(m.span())
    for m in _RA_RE.finditer(text):
        take(m, "ra", m.group(1))
    for m in _FORM_RE.finditer(text):
        take(m, "form", re.sub(r"\D", "", m.group(1)))
    for m in _PESO_RE.finditer(text):
        d = _dec(m.group(1))
        if d is not None:
            mult = _MULT[m.group(2).lower()] if m.group(2) else 1
            take(m, "peso", str(_norm(d * mult)))
    for m in _PCT_RE.finditer(text):
        if any(s <= m.start() < e for s, e in spans):
            continue
        d = _dec(m.group(1))
        if d is not None:
            take(m, "percent", str(_norm(d)))

    for m in _NUM_RE.finditer(text):
        if any(s <= m.start() < e for s, e in spans):
            continue
        raw = m.group(1)
        if len(re.sub(r"\D", "", raw)) < 2:
            continue  # single digits: list numbering, ordinals, "2 copies"
        d = _dec(raw)
        if d is None:
            continue
        mult = _MULT[m.group(2).lower()] if m.group(2) and m.group(2).lower() in _MULT else 1
        toks.append(Token("number", m.group(0).strip(), str(_norm(d * mult))))
    return toks


def _found(tok: Token, excerpt: str, nums: Set[Decimal]) -> bool:
    if tok.kind in ("peso", "percent", "number"):
        d = Decimal(tok.value)
        if d in nums:
            return True
        # "₱3,000,000" in the draft vs "₱3 million" in the excerpt, either way.
        return any(d == n * m for n in nums for m in (10**3, 10**6, 10**9, 10**12) if n < 1000)
    if tok.kind == "issuance":
        return tok.value in parse_issuance_ids(excerpt)
    if tok.kind in ("ra", "form"):
        return re.search(rf"(?<!\d){re.escape(tok.value)}(?!\d)", excerpt) is not None
    if tok.kind == "contact":
        return tok.value in excerpt.lower()
    return True


# ── Draft sentences ──────────────────────────────────────────────────────

_ABBREV_END_RE = re.compile(
    r"(?:\b(?:No|Nos|Sec|Secs|Art|Arts|par|Par|vs|Inc|Co|Corp|Rev|Reg|Regs|Mr|Ms|Dr|Atty|St|Jr|Sr|e\.g|i\.e|etc|cf|approx)|\b[A-Z])\.$"
)
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"“(\[*_]*[A-Z0-9₱(•\-–—])")


def _split_line(line: str) -> List[str]:
    parts: List[str] = []
    buf = ""
    pieces = _SPLIT_RE.split(line)
    for piece in pieces:
        buf = f"{buf} {piece}".strip() if buf else piece
        if not _ABBREV_END_RE.search(buf.rstrip("*_ ")):
            parts.append(buf)
            buf = ""
    if buf:
        parts.append(buf)
    return parts


def split_draft(draft: str) -> List[List[str]]:
    """The draft as lines, each a list of sentences. Lines are kept so a
    repaired draft keeps its bullets and paragraphs."""
    return [_split_line(line) if line.strip() else [] for line in (draft or "").split("\n")]


def join_draft(lines: List[List[str]]) -> str:
    out = [" ".join(s) for s in lines]
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ── The check ────────────────────────────────────────────────────────────

@dataclass
class SentenceIssue:
    index: int            # flat sentence index in the draft
    sentence: str
    problems: List[str] = field(default_factory=list)


@dataclass
class GroundingReport:
    passed: bool
    n_sentences: int
    n_tokens: int
    flagged: List[SentenceIssue] = field(default_factory=list)

    @property
    def issues(self) -> List[str]:
        return [p for f in self.flagged for p in f.problems]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["issues"] = self.issues
        return d


def check(draft: str, context_text: str, question: str = "") -> GroundingReport:
    """Run the rules in the module docstring. A token the user's own question
    contains ("my gross pay is ₱360,000") may be repeated without support."""
    excerpts = parse_context(context_text)
    if not excerpts:  # not numbered excerpts (format_context changed?): nothing to check against
        return GroundingReport(True, 0, 0)
    nums = {n: numbers_in(t) for n, t in excerpts.items()}
    asked = ("question", question or "", numbers_in(question or ""))
    lines = split_draft(draft)
    flat: List[Tuple[int, int, str]] = [(li, si, s) for li, sents in enumerate(lines) for si, s in enumerate(sents)]
    flagged: List[SentenceIssue] = []
    n_tokens = 0

    for idx, (li, si, sentence) in enumerate(flat):
        problems: List[str] = []
        own = citations(sentence)
        bad = sorted({c for c in own if c not in excerpts})
        if bad:
            problems.append(f"cites excerpt {', '.join(f'[{c}]' for c in bad)}, which does not exist")
        cited = [c for c in own if c in excerpts]
        if not own:
            # "... X. Y [1]." style: a later sentence on the same line cites for both.
            later = [c for (l2, s2, s) in flat[idx + 1:] if l2 == li for c in citations(s)]
            cited = [c for c in later if c in excerpts][:6]
        toks = [t for t in fact_tokens(sentence) if not _found(t, asked[1], asked[2])]
        n_tokens += len(toks)
        if toks and not cited and not bad:
            # An uncited fact that IS in some excerpt (a bullet under a cited
            # lead-in, say) is left to the verifier; one in no excerpt is not.
            for t in toks:
                if not any(_found(t, excerpts[n], nums[n]) for n in excerpts):
                    problems.append(f"\"{t.raw}\" is not in any excerpt")
        elif toks and cited:
            for t in toks:
                if any(_found(t, excerpts[n], nums[n]) for n in cited):
                    continue
                elsewhere = [n for n in excerpts if n not in cited and _found(t, excerpts[n], nums[n])]
                if elsewhere:
                    problems.append(f"\"{t.raw}\" is in excerpt [{elsewhere[0]}], not in the excerpt it cites")
                else:
                    problems.append(f"\"{t.raw}\" is not in any excerpt")
        if problems:
            flagged.append(SentenceIssue(idx, sentence, problems))
    return GroundingReport(not flagged, len(flat), n_tokens, flagged)


# ── Repair ───────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[A-Za-zÀ-ÿñÑ]{3,}|\d[\d,.]*")


def protected_indices(draft: str) -> Set[int]:
    """The opening answer: sentence 0, plus sentence 1 when sentence 0 is a
    bare "Yes."/"Oo."/"Hindi na." (3 words or fewer)."""
    flat = [s for sents in split_draft(draft) for s in sents]
    if not flat:
        return set()
    prot = {0}
    if len(re.findall(r"\w+", _CITE_RE.sub("", flat[0]))) <= 3 and len(flat) > 1:
        prot.add(1)
    return prot


def map_issues_to_sentences(draft: str, issues: Sequence[str]) -> Optional[Set[int]]:
    """Which sentences the verifier's issues refer to. Each issue is a brief
    quote of the claim; it maps to the sentence sharing most of its words (at
    least half) or containing its number. None when ANY issue can't be
    mapped confidently: then the draft is regenerated instead of repaired."""
    flat = [s for sents in split_draft(draft) for s in sents]
    if not flat or not issues:
        return None
    sent_words = [set(w.lower().rstrip(".,") for w in _WORD_RE.findall(_CITE_RE.sub("", s))) for s in flat]
    hit: Set[int] = set()
    for issue in issues:
        words = set(w.lower().rstrip(".,") for w in _WORD_RE.findall(issue or ""))
        if not words:
            return None
        nums = {w for w in words if w[0].isdigit()}
        best, best_score = None, 0.0
        for i, sw in enumerate(sent_words):
            score = len(words & sw) / len(words)
            if nums and nums <= sw:
                score = max(score, 0.75)
            if score > best_score:
                best, best_score = i, score
        if best is None or best_score < 0.5:
            return None
        hit.add(best)
    return hit


def repair(draft: str, drop: Set[int]) -> Optional[str]:
    """The draft without the sentences in `drop`, or None when repair is not
    allowed: an opening-answer sentence is flagged, nothing would be left, or
    no cited sentence would remain."""
    if not drop or drop & protected_indices(draft):
        return None
    lines = split_draft(draft)
    out: List[List[str]] = []
    idx = 0
    for sents in lines:
        keep = []
        for s in sents:
            if idx not in drop:
                keep.append(s)
            idx += 1
        if sents and not keep:
            continue  # the whole line (e.g. a bullet) was removed
        out.append(keep)
    text = join_draft(out)
    if not text or not citations(text):
        return None
    # A bullet list whose items were all removed leaves a dangling "...:" lead-in.
    if re.search(r":\s*$", text):
        return None
    return text


def settings() -> dict:
    return {"grounding_check": GROUNDING_CHECK, "claim_repair": CLAIM_REPAIR}
