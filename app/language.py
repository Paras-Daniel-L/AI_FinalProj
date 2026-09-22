"""
Lightweight user-language detection for prompt construction.

This is intentionally NOT the topic classifier in classifier.py — that
model was trained on ~35 short examples to route BIR-year / chit-chat /
board-game *topics*, and has no language signal whatsoever. Language
identification needs to be its own small, fast, dependency-light step
that runs once per query in query processing, before retrieval, and
travels alongside the untouched original query all the way to the
generation prompt (see app/llm.py).

Approach: a small curated list of high-frequency Filipino/Tagalog function
words (particles, question words, pronouns) counted against a similarly
short list of English function words. Function words are used deliberately
instead of content words — "requirements" and "enrollment" say nothing
about language (they show up in all-English AND all-Filipino sentences
about BIR enrollment), whereas "ang", "ba", "po", "kasi", "yung" are
near-unambiguous Filipino markers even in heavily code-switched speech.

No fastText/langdetect dependency: BIR queries here are short (often a
single clause), and general-purpose language-ID models tend to average
over the whole string, which misclassifies short Taglish queries as
English on the strength of one or two English content words. A binary
marker count is more robust for this specific input shape.
"""

import re
from dataclasses import dataclass

FILIPINO_MARKERS = {
    "ang", "ng", "mga", "sa", "na", "ay", "ba", "po", "opo", "kasi",
    "yung", "yun", "ito", "iyan", "iyon", "dito", "diyan", "doon",
    "ako", "ikaw", "ka", "siya", "kami", "tayo", "kayo", "sila",
    "ano", "sino", "saan", "kailan", "bakit", "paano", "magkano",
    "hindi", "oo", "wala", "meron", "mayroon", "may", "para",
    "kung", "kapag", "dapat", "gusto", "gagawin", "pwede", "puwede",
    "salamat", "maganda", "magandang", "kumusta", "kamusta",
    "nang", "mo", "ko", "niya", "namin", "natin", "ninyo", "nila",
    "din", "rin", "lang", "naman", "talaga", "siguro", "kanina",
}

ENGLISH_MARKERS = {
    "the", "is", "are", "was", "were", "what", "where", "when", "why",
    "how", "who", "which", "do", "does", "did", "can", "could", "would",
    "should", "will", "shall", "have", "has", "had", "this", "that",
    "these", "those", "and", "or", "but", "for", "with", "about",
    "please", "thanks", "thank", "you", "your", "my", "our",
}

_WORD_RE = re.compile(r"[A-Za-zÀ-ÿñÑ']+")


@dataclass
class LanguageResult:
    label: str          # "english" | "filipino" | "taglish"
    filipino_hits: int
    english_hits: int
    display_name: str   # human-readable phrase, injected directly into prompts


def detect_language(text: str) -> LanguageResult:
    """
    Classify a query as english / filipino / taglish using function-word
    marker counts.

    Falls back to "english" when no markers of either kind are found (a
    bare year, an acronym like "RDAO 1-2022", a single content word) —
    this is a deliberate default-safe choice matching SYSTEM_PROMPT's
    baseline voice, rather than guessing from too little signal.
    """
    tokens = [t.lower() for t in _WORD_RE.findall(text or "")]

    fil_hits = sum(1 for t in tokens if t in FILIPINO_MARKERS)
    eng_hits = sum(1 for t in tokens if t in ENGLISH_MARKERS)

    if fil_hits == 0 and eng_hits == 0:
        return LanguageResult("english", 0, 0, "English")

    if fil_hits > 0 and eng_hits > 0:
        dominant = "Filipino" if fil_hits >= eng_hits else "English"
        return LanguageResult(
            "taglish",
            fil_hits,
            eng_hits,
            f"Taglish (Filipino-English code-switched, leaning {dominant})",
        )

    if fil_hits > 0:
        return LanguageResult("filipino", fil_hits, eng_hits, "Filipino")

    return LanguageResult("english", fil_hits, eng_hits, "English")
