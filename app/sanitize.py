"""
Input sanitization for /query: clean the text, then check the question is one
that the retrieval budget can actually answer.

Why the limits exist
--------------------
Every answer is built from at most 5 retrieved excerpts (RRF_FINAL_TOP_K),
each up to ~800 characters — roughly 4,000 characters of evidence in total.
A question can only be answered if that much evidence can cover it, so a
question must be short, about one thing, and name only as many documents as
five excerpts can reasonably cover. A long or multi-part question also hurts
retrieval directly: the whole text is embedded as ONE vector and scored by
BM25 as one bag of words, so unrelated parts dilute each other and none of
them retrieves well. A too-broad question is therefore rejected with a
message asking the user to narrow it, rather than answered badly or
truncated silently (the real question is often at the END of a long text).

  MAX_QUERY_CHARS      400   characters after cleaning (typical questions are
                             well under 150)
  MAX_QUERY_QUESTIONS  3     separate question marks
  MAX_QUERY_CITATIONS  3     DISTINCT issuance numbers ("24-2026", "9-2026" ...)

What cleaning does
------------------
Unicode is normalized (NFKC: full-width digits become normal digits, so the
year-filter regex and BM25 see "2026"); invisible and control characters
(zero-width spaces, bidi overrides, BOM, null bytes, ...) are removed; every
run of whitespace, INCLUDING newlines, becomes one space — so a question can't
fake a new prompt section such as "USER QUESTION:" on its own line.

Client-supplied conversation history goes through the same cleaning: only
"user" / "assistant" roles are kept (a fake "system" turn is dropped), each
message is capped, and only the most recent messages are kept.

What this is NOT
----------------
It is not a prompt-injection detector. Phrase-matching "ignore previous
instructions" is unreliable and easy to evade, and there is no privilege
boundary to protect here (the chat has one user, and the prompt contains no
secrets). The real defenses are structural: the model may only use the
retrieved excerpts, and a second, different model verifies every answer
against them. These limits mainly protect quality and cost.

All limits are overridable with env vars of the same names.
"""

import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dotenv import load_dotenv

from .schemas import ConvMessage

load_dotenv()

MAX_QUERY_CHARS: int = int(os.environ.get("MAX_QUERY_CHARS", "400"))
MAX_QUERY_QUESTIONS: int = int(os.environ.get("MAX_QUERY_QUESTIONS", "3"))
MAX_QUERY_CITATIONS: int = int(os.environ.get("MAX_QUERY_CITATIONS", "3"))
MAX_HISTORY_MESSAGES: int = int(os.environ.get("MAX_HISTORY_MESSAGES", "10"))
MAX_HISTORY_CHARS: int = int(os.environ.get("MAX_HISTORY_CHARS", "600"))

# "<number>-<year>" as in every BIR issuance number in the corpus ("24-2026",
# "105-2023", "1-2001"). Same shape as classifier._ISSUANCE_YEAR_RE. A year
# range like "2024-2025" does not match (the part before the hyphen may have
# at most 3 digits).
_CITATION_RE = re.compile(r"\b\d{1,3}[A-Za-z]?-20\d{2}\b")
_QUESTION_RE = re.compile(r"\?+")  # "???" counts as one question mark


def clean_text(text: str) -> str:
    """NFKC, drop control/invisible characters, collapse all whitespace to single spaces."""
    text = unicodedata.normalize("NFKC", text or "")
    out: List[str] = []
    for ch in text:
        category = unicodedata.category(ch)
        if ch in "\t\n\r\v\f" or category in ("Zs", "Zl", "Zp"):
            out.append(" ")
        elif category in ("Cc", "Cf", "Cs", "Co", "Cn"):
            continue  # control, format (zero-width, bidi), surrogate, private-use, unassigned
        else:
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


@dataclass
class SanitizedQuery:
    query: str                      # the cleaned text (use THIS downstream, not the raw input)
    reason: Optional[str] = None    # None = accepted; else "empty" | "too_long" | "too_many_questions" | "too_many_citations"
    info: Dict[str, int] = field(default_factory=dict)   # the numbers behind a rejection

    @property
    def ok(self) -> bool:
        return self.reason is None


def sanitize_query(raw: str) -> SanitizedQuery:
    """Clean `raw` and decide whether it is a question the 5-excerpt budget can answer."""
    text = clean_text(raw)

    if sum(ch.isalnum() for ch in text) < 2:
        return SanitizedQuery(text, "empty")

    if len(text) > MAX_QUERY_CHARS:
        return SanitizedQuery(text, "too_long", {"n": len(text), "max": MAX_QUERY_CHARS})

    questions = len(_QUESTION_RE.findall(text))
    if questions > MAX_QUERY_QUESTIONS:
        return SanitizedQuery(text, "too_many_questions", {"n": questions, "max": MAX_QUERY_QUESTIONS})

    citations = len({m.group(0).lower() for m in _CITATION_RE.finditer(text)})
    if citations > MAX_QUERY_CITATIONS:
        return SanitizedQuery(text, "too_many_citations", {"n": citations, "max": MAX_QUERY_CITATIONS})

    return SanitizedQuery(text)


def sanitize_history(history: Optional[List[ConvMessage]]) -> List[ConvMessage]:
    """
    Clean client-supplied history: keep only user/assistant turns, clean and cap
    each message, drop empty ones, keep the most recent MAX_HISTORY_MESSAGES.
    """
    cleaned: List[ConvMessage] = []
    for msg in history or []:
        role = (msg.role or "").strip().lower()
        if role not in ("user", "assistant"):
            continue
        content = clean_text(msg.content)
        if not content:
            continue
        if len(content) > MAX_HISTORY_CHARS:
            content = content[: MAX_HISTORY_CHARS - 1].rstrip() + "…"
        cleaned.append(ConvMessage(role=role, content=content))
    return cleaned[-MAX_HISTORY_MESSAGES:]


# ── Localized rejection messages (fixed text; no model call is made) ─────

_MESSAGES = {
    "empty": {
        "english": "Please type a question about BIR tax regulations.",
        "filipino": "Mag-type po ng tanong tungkol sa mga regulasyon ng BIR.",
        "taglish": "Please mag-type ng question tungkol sa BIR tax regulations.",
    },
    "too_long": {
        "english": (
            "Your question is too long ({n} characters; the limit is {max}). "
            "Please shorten it to one focused question so the answer can be "
            "found in the available documents."
        ),
        "filipino": (
            "Masyadong mahaba ang tanong mo ({n} na karakter; ang limitasyon ay {max}). "
            "Paikliin mo po ito sa isang malinaw na tanong para makahanap ng sagot "
            "sa mga available na dokumento."
        ),
        "taglish": (
            "Masyadong mahaba ang question mo ({n} characters; ang limit ay {max}). "
            "Paikliin mo lang into one focused question para makita ang sagot "
            "sa mga available na documents."
        ),
    },
    "too_many_questions": {
        "english": (
            "Please ask one question at a time (I found {n} question marks; the "
            "limit is {max}). I can only draw on a few document excerpts per answer."
        ),
        "filipino": (
            "Isang tanong lang po sa bawat pagkakataon (may {n} tandang pananong sa "
            "mensahe mo; ang limitasyon ay {max}). Kaunting bahagi lang ng mga dokumento "
            "ang magagamit ko sa bawat sagot."
        ),
        "taglish": (
            "Isang question lang per message, please (may {n} question marks; ang limit "
            "ay {max}). Kaunting excerpts lang ng documents ang magagamit ko sa bawat sagot."
        ),
    },
    "too_many_citations": {
        "english": (
            "Your question mentions {n} different issuances (the limit is {max}). "
            "I can only draw on about 5 document excerpts per answer, so please ask "
            "about {max} or fewer at a time."
        ),
        "filipino": (
            "May {n} magkakaibang issuance sa tanong mo (ang limitasyon ay {max}). "
            "Mga 5 bahagi lang ng dokumento ang magagamit ko sa bawat sagot, kaya "
            "magtanong po tungkol sa {max} o mas kaunti sa bawat pagkakataon."
        ),
        "taglish": (
            "May {n} different issuances sa question mo (ang limit ay {max}). "
            "Mga 5 document excerpts lang ang magagamit ko sa bawat sagot, kaya "
            "mag-ask about {max} or fewer at a time."
        ),
    },
}


def rejection_message(result: SanitizedQuery, language_label: str) -> str:
    """The fixed, localized reply for a rejected query."""
    by_language = _MESSAGES.get(result.reason or "empty", _MESSAGES["empty"])
    template = by_language.get(language_label, by_language["english"])
    return template.format(**result.info) if result.info else template
