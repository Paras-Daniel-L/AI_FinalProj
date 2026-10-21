"""
Answerability: two cheap, rule-based checks around generation (system v1.6).

1. Evidence gate (BEFORE generation)
   If the best rerank score of the retrieved excerpts is below
   EVIDENCE_GATE_MIN_SCORE, no excerpt is really about the question, so the
   generator is not called at all and the user gets guidance. The outcome is
   the existing "no_retrieval" (nothing usable was retrieved), so rag_eval
   still counts it as a refusal.

   Why: the reranker always keeps at least one chunk (RERANK_MIN_DOCS=1) and
   the dense floor adds more, so the generator was nearly always handed
   something and rarely said NO_ANSWER. In the 2026-10-07 live logs every
   question whose best rerank score was under ~0.14 was off-topic or not in
   the corpus (Netflix/Roblox tax, "malungkot ako", "sino ang pumatay kay
   Rizal", SSS registration), while answerable BIR questions scored 0.26 and
   up. The default sits in that gap; calibrate it on a DEV set (never on
   T-TED) with `python -m tools.dev_tune`. 0 turns the gate off.

2. Non-answer detector (AFTER generation, BEFORE verification)
   A draft whose opening sentence says the excerpts don't answer the question
   ("Hindi direktang sinasagot ng mga excerpt...", "The excerpts do not
   mention...") is a refusal written in the model's own words, usually
   followed by loosely related facts. Those were passing the verifier (the
   facts ARE in the excerpts) and being shown, cached and scored as answers.
   They are now treated exactly like the NO_ANSWER sentinel: outcome
   "generator_no_answer", guidance shown, nothing cached.

   Only the FIRST sentence is checked. A partial answer that first answers and
   then says which part the documents don't cover ("... [1]. Hindi sakop ng
   mga dokumento ang ...") is a real answer and is kept, as prompt v4 asks.

Both are deterministic, logged in the trace, and switchable for ablations:
EVIDENCE_GATE_MIN_SCORE=0 and NON_ANSWER_DETECTION=0 restore v1.5.2.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, Optional

from dotenv import load_dotenv

load_dotenv()

EVIDENCE_GATE_MIN_SCORE: float = float(os.environ.get("EVIDENCE_GATE_MIN_SCORE", "0.14"))
NON_ANSWER_DETECTION: bool = os.environ.get("NON_ANSWER_DETECTION", "1").strip() in ("1", "true", "True", "yes")


def top_rerank_score(docs: Iterable) -> Optional[float]:
    """Best `_rerank_score` among the retrieved excerpts, or None when the
    reranker did not run (off, or failed and fell back to RRF)."""
    scores = [d.metadata.get("_rerank_score") for d in docs]
    scores = [float(s) for s in scores if isinstance(s, (int, float))]
    return max(scores) if scores else None


def passes_gate(docs, threshold: Optional[float] = None) -> tuple:
    """(passed, top_score). Passes when the gate is off, when there are no
    rerank scores to judge by, or when the best score clears the threshold."""
    threshold = EVIDENCE_GATE_MIN_SCORE if threshold is None else threshold
    top = top_rerank_score(docs)
    if threshold <= 0 or top is None:
        return True, top
    return top >= threshold, top


# ── Non-answer detector ──────────────────────────────────────────────────
# A reference to the documents...
_DOCS = r"(?:excerpts?|documents?|dokumento|sources?|context|konteksto|provided\s+(?:text|information)|retrieved|mga\s+excerpt|ibinigay)"
# ...combined with a statement that they don't cover the question.
_NON_ANSWER_PATTERNS = [
    # Filipino / Taglish
    rf"\bhindi\s+sapat\s+ang\s+(?:mga\s+)?{_DOCS}",
    rf"\bhindi\s+(?:direktang\s+|tahasang\s+|malinaw(?:\s+na)?\s+)?(?:sinasagot|saklaw|sakop|tinutukoy|binabanggit|nababanggit|sinasabi|nakasaad|tinatalakay|kasama|masasagot|matutukoy|makikita)\b[^.]*\b{_DOCS}",
    rf"\b{_DOCS}[^.]*\bhindi\s+(?:direktang\s+)?(?:sinasagot|saklaw|sakop|tinutukoy|binabanggit|nababanggit|sinasabi|nakasaad|tinatalakay|kasama)\b",
    rf"\bwala(?:ng)?\s+(?:sa\s+(?:mga\s+)?{_DOCS}|{_DOCS}|impormasyon|binanggit|nakasaad|sinasabi)\b",
    rf"\bhindi\s+(?:ko\s+)?masasagot\b",
    rf"\bhindi\s+malinaw\s+sa\s+(?:mga\s+)?{_DOCS}",
    # English
    rf"\b{_DOCS}[^.]*\b(?:do(?:es)?\s+not|don'?t|doesn'?t|did\s+not|fail\s+to)\s+(?:directly\s+|explicitly\s+|specifically\s+)?"
    r"(?:answer|address|cover|mention|state|specify|say|discuss|include|contain|provide|indicate|name|refer)",
    rf"\b(?:no|none\s+of\s+the)\s+{_DOCS}\b[^.]*\b(?:answer|address|cover|mention|state|specify|say|discuss|contain|provide|refer)",
    rf"\b(?:not|isn'?t|is\s+not)\s+(?:covered|addressed|mentioned|stated|specified|found)\s+(?:in|by)\s+(?:the\s+)?{_DOCS}",
    r"\b(?:cannot|can'?t|unable\s+to)\s+(?:be\s+)?(?:answer(?:ed)?|determine[d]?|tell)\b",
    r"\bthere\s+is\s+no\s+(?:information|mention)\b",
    rf"\b(?:not\s+enough|insufficient)\s+(?:information|detail)\s+(?:in|from)\s+(?:the\s+)?{_DOCS}",
]
_NON_ANSWER_RE = re.compile("|".join(f"(?:{p})" for p in _NON_ANSWER_PATTERNS), re.IGNORECASE)
# First sentence = up to the first ., ! or ? followed by space/end, or a line break.
_FIRST_SENTENCE_RE = re.compile(r"^\s*(.+?(?:[.!?](?=\s|$)|\n|$))", re.DOTALL)


def first_sentence(text: str) -> str:
    m = _FIRST_SENTENCE_RE.match(text or "")
    return (m.group(1) if m else (text or "")).strip()


def is_non_answer(draft: str) -> bool:
    """True when the draft's opening sentence says the documents don't answer
    the question (see the module docstring)."""
    if not NON_ANSWER_DETECTION:
        return False
    head = first_sentence(re.sub(r"[*_`#>]", "", draft or ""))
    return bool(_NON_ANSWER_RE.search(head))


def settings() -> dict:
    return {"evidence_gate_min_score": EVIDENCE_GATE_MIN_SCORE, "non_answer_detection": NON_ANSWER_DETECTION}
