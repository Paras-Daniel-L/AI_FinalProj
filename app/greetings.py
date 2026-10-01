"""
Lightweight, rule-based greeting detector.

NOT a chit-chat classifier and NOT a substitute for one. Sagot AI has no
general-knowledge / open-domain conversation mode (removed — thesis scope
is strictly grounded BIR tax Q&A, see prompts.SYSTEM_PROMPT). This module
exists purely so a bare "hi" or "kumusta ka" gets a short, fixed greeting
back instead of the canned no-answer refusal — without ever routing
through an LLM call that could answer from outside knowledge.

Deliberately narrow, and deliberately a WHOLE-MESSAGE match (`^...$`):
"hi" matches; "hi, magkano ang VAT?" does not. Any real content attached
to a greeting must fall through to normal classification + RAG rather
than being short-circuited into a canned reply that ignores the actual
question.

This also means casual small talk beyond a bare greeting/closing —
"ano ang pangalan mo", "kumain ka na ba", "I wanna ask something" — is
intentionally NOT matched here. That's the point: it's general chat, not
a greeting, and general chat no longer has a home in this system: it
falls through to classification + RAG like any other input and gets the
same grounded-answer-or-refuse treatment.
"""

import re

_GREETING_RE = re.compile(
    r"""^\s*(
        hi+ | hello | hey |
        good\s+(morning|afternoon|evening|day) |
        (magandang\s+(araw|umaga|hapon|gabi))(\s+(sayo|po))? |
        (kumusta|kamusta)(\s+(ka|kayo|po))? |
        how\s+are\s+you |
        (salamat|thank\s+you|thanks)(\s+po)? |
        (bye|paalam|goodbye)
    )\s*[!.?]*\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def is_greeting(text: str) -> bool:
    """True only when the ENTIRE message is a bare greeting/closing —
    never when a real question is attached to one."""
    return bool(_GREETING_RE.match((text or "").strip()))
