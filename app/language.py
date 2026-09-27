"""
Lightweight user-language detection for prompt construction.

Runs once per query (on the sanitized text, in api.py) before retrieval;
the result travels with the untouched question into the generation prompt
and picks the language of the fixed messages (refusals, greetings,
rejections). It has nothing to do with routing: the year filter lives in
classifier.py and is purely regex-based.

Approach: a small curated list of high-frequency Filipino/Tagalog function
words (particles, question words, pronouns) counted against a similarly
short list of English function words. Function words are used deliberately
instead of content words — "requirements" and "enrollment" say nothing
about language (they show up in all-English AND all-Filipino sentences
about BIR enrollment), whereas "ang", "ba", "po", "kasi", "yung" are
near-unambiguous Filipino markers even in heavily code-switched speech.

Ambiguous markers: "may" (Filipino "there is/has", English month and modal
verb) and "para" (Filipino "for", English "para 3" = paragraph) are real
English words too. On their own they said nothing, but they used to turn
"What is the deadline in May 2025?" into Taglish, so the answer came back
in Taglish. They now count only when the query ALSO has an unambiguous
Filipino marker ("May penalty ba?" is still Filipino via "ba").

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
    "hindi", "oo", "wala", "meron", "mayroon",
    "kung", "kapag", "dapat", "gusto", "gagawin", "pwede", "puwede",
    "salamat", "maganda", "magandang", "kumusta", "kamusta",
    "nang", "mo", "ko", "niya", "namin", "natin", "ninyo", "nila",
    "din", "rin", "lang", "naman", "talaga", "siguro", "kanina",
}

# Also ordinary English words ("in May 2025", "you may file", "para 3"):
# counted as Filipino only when an unambiguous marker above is also present.
AMBIGUOUS_FILIPINO_MARKERS = {"may", "para"}

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
    bare year, an acronym like "RDAO 1-2022", a single content word, or
    only ambiguous words like "may"/"para") —
    this is a deliberate default-safe choice matching SYSTEM_PROMPT's
    baseline voice, rather than guessing from too little signal.
    """
    tokens = [t.lower() for t in _WORD_RE.findall(text or "")]

    fil_hits = sum(1 for t in tokens if t in FILIPINO_MARKERS)
    if fil_hits:
        fil_hits += sum(1 for t in tokens if t in AMBIGUOUS_FILIPINO_MARKERS)
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
