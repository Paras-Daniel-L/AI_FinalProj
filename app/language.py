"""
Lightweight Taglish / Filipino-language detection.

Deliberately rule-based rather than an LLM call: it's free, instant, and -
most importantly for a thesis methodology section - it's a fixed, testable
function you can report precision/recall for against a labeled set. An
LLM "just noticing" the language while it answers isn't a component you
can evaluate or defend under questioning; this is.

The approach: a marker-word list of common Filipino/Tagalog function words
(pronouns, particles, question words, connectives) that essentially never
appear in English tax queries. If enough of them show up relative to the
query length, the query is code-mixed Taglish (or pure Filipino) rather
than English. This deliberately does NOT try to distinguish "Taglish" from
"pure Filipino" - for this pipeline's purposes (does the query need
rewriting into formal English before retrieval, and does the answer need
translating back?) that distinction doesn't matter; both need the same
handling.
"""
import re

# Common Filipino/Tagalog function words. These are chosen because they're
# near-exclusively Filipino (unlike content words that could be shared or
# borrowed) and appear across registers - casual chat and formal-ish
# Taglish tax questions alike.
FILIPINO_MARKERS = {
    "ang", "ng", "nang", "mga", "sa", "na", "pa", "ba", "po", "opo",
    "ko", "mo", "niya", "namin", "natin", "nila", "kanya", "akin", "iyo",
    "atin", "amin", "kanila",
    "ako", "ikaw", "siya", "kami", "tayo", "kayo", "sila",
    "ito", "iyan", "iyon", "yun", "yung", "nito", "niyan", "noon", "nung",
    "hindi", "oo", "wala", "meron", "mayroon", "walang",
    "paano", "saan", "kailan", "bakit", "sino", "alin", "magkano",
    "kasi", "dahil", "kung", "kapag", "para", "upang", "habang",
    "gusto", "ayaw", "pwede", "puwede", "dapat", "kailangan",
    "lang", "din", "rin", "naman", "talaga", "nga", "daw", "raw",
    "kumusta", "kamusta", "salamat", "magandang", "araw",
    "buwis", "magbayad", "bayad", "resibo",  # tax-domain Filipino terms
}

# Query needs at least this many marker hits (not ratio - short queries
# like "magkano ang buwis" would fail a high ratio threshold) to count as
# Taglish. One strong marker is often enough ("magkano", "paano", "ba").
MIN_MARKER_HITS = 1

_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def detect_taglish(text: str) -> bool:
    """
    Returns True if the text shows Filipino/Taglish code-mixing, False if
    it reads as English. Case-insensitive, punctuation-insensitive.
    """
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in FILIPINO_MARKERS)
    return hits >= MIN_MARKER_HITS
