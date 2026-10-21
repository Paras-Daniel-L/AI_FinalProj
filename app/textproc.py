"""
Text processing shared by ingestion, retrieval and evaluation.

Pure Python, no app imports, so every other module (database.py,
retrieval.py, rag_eval/judge.py) can use it without import cycles.

Why it exists (evidence from the T-TED evaluation run, 500 retrieved chunks):
  * 64% of chunks ended mid-sentence: the old RecursiveCharacterTextSplitter
    cut every 800 characters wherever that fell, so a chunk often carried the
    tail of one provision and the head of another — on-topic text mixed with
    off-topic text, which is exactly what Context Relevance penalizes.
  * ~290 broken-hyphen tokens ("Value -Added", "e -marketplace", "No. 8 -2026")
    and split words ("shal l", "creditab le") from PDF text extraction. BM25
    tokenizes "8 -2026" as "8" and "2026", so a query naming "RMC No. 8-2026"
    could not match the issuance's own text (T-TED B36 missed exactly this way).
  * The chunk text never contained the issuance name: page 3 of
    "RMC No. 19-2022" doesn't say "RMC No. 19-2022" anywhere, so neither BM25
    nor the embedding could connect a question naming that issuance to it.

What it provides:
  clean_pdf_text()      undo PDF extraction damage (line wraps, broken hyphens)
  build_vocab() / repair_split_words()
                        corpus-driven repair of words split by a stray space
  split_sentences()     one sentence splitter used EVERYWHERE (chunking,
                        evidence compression, and the evaluation judge), so
                        "a sentence" means the same thing in all three
  chunk_sentences()     pack whole sentences into chunks of <= max_chars
  parse_issuance_ids()  "RMC No. 8-2026", "Revenue Regulations No. 14-2022",
                        "RR 14-2022" -> canonical ids "RMC-8-2026", "RR-14-2022"
  issuance_title()      "data/2024/RMC No. 34-2024.pdf" -> "RMC No. 34-2024"
"""

import os
import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Optional

# Chunking settings (read by app/database.py; recorded in the index config by
# app/embeddings.py so changing them forces a rebuild instead of silently
# mixing two kinds of chunks in one index).
CHUNKING_MODE = os.environ.get("CHUNKING_MODE", "sentence").strip().lower()
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP_SENTENCES = int(os.environ.get("CHUNK_OVERLAP_SENTENCES", "1"))
# Bump when the cleaning/splitting code changes in a way that changes chunk text.
CHUNKER_VERSION = "2"


def chunking_config() -> Dict[str, str]:
    if CHUNKING_MODE == "legacy":
        return {"chunking": "legacy-recursive-800-80"}
    return {
        "chunking": f"sentence-v{CHUNKER_VERSION}",
        "chunk_size": str(CHUNK_SIZE),
        "chunk_overlap_sentences": str(CHUNK_OVERLAP_SENTENCES),
        "context_header": "embedded",
    }

# ── PDF text cleaning ─────────────────────────────────────────────────────

# A line that starts a list item: (a)  a.  a)  1.  1)  i.  iv)  •  -  –
_LIST_ITEM_RE = re.compile(r"^\s*(\(?[a-zA-Z0-9]{1,4}[.)]\s|\([a-zA-Z0-9]{1,4}\)|[•▪●◦\-–]\s)")
_SENTENCE_END_RE = re.compile(r"[.!?:;]\s*$")


def clean_pdf_text(text: str) -> str:
    """
    Undo the most common PyPDF / OCR extraction damage without changing
    meaning:

    * soft line wraps inside a paragraph are joined into one line, while
      paragraph breaks (blank lines), list items and lines ending in ":" stay
      on their own line;
    * a word hyphenated across a line break ("pass -\\nthrough") is re-joined
      with its hyphen ("pass-through");
    * a hyphen with a stray space before it ("Value -Added", "8 -2026",
      "e -marketplace") is closed up. A hyphen with spaces on BOTH sides
      (" - ", a dash) is left alone;
    * runs of spaces collapse to one.

    It never touches digits, currency signs, periods or percent signs.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("­", "")                      # soft hyphen
    text = re.sub(r"[ \t ]+", " ", text)
    # hyphenated line break: "pass -\n through" -> "pass-through"
    text = re.sub(r"(\w) ?-[ \t]*\n[ \t]*(\w)", r"\1-\2", text)
    # stray space before a hyphen: "Value -Added" -> "Value-Added", "8 -2026" -> "8-2026"
    text = re.sub(r"(\w) -(?=\w)", r"\1-", text)
    # stray space after a hyphen inside an issuance number: "8- 2026" -> "8-2026"
    text = re.sub(r"(\d)- (?=\d{4}\b)", r"\1-", text)

    lines = [ln.strip() for ln in text.split("\n")]
    out: List[str] = []
    for ln in lines:
        if not ln:
            if out and out[-1] != "":
                out.append("")                              # paragraph break
            continue
        if (
            out
            and out[-1] != ""
            and not _LIST_ITEM_RE.match(ln)
            and not out[-1].rstrip().endswith(":")
        ):
            out[-1] = f"{out[-1]} {ln}"                     # soft wrap -> join
        else:
            out.append(ln)
    joined = "\n".join(out)
    joined = re.sub(r"\n{2,}", "\n\n", joined)
    return re.sub(r" {2,}", " ", joined).strip()


# ── Corpus-driven split-word repair ("shal l" -> "shall") ─────────────────

_ALPHA_TOKEN_RE = re.compile(r"[A-Za-z]+")


def build_vocab(texts: Iterable[str]) -> Counter:
    """Lower-cased alphabetic token counts over the whole corpus."""
    vocab: Counter = Counter()
    for t in texts:
        vocab.update(w.lower() for w in _ALPHA_TOKEN_RE.findall(t or ""))
    return vocab


_WORD_OR_GAP_RE = re.compile(r"[A-Za-z]+|[^A-Za-z]+")


def repair_split_words(text: str, vocab: Counter, min_joined: int = 3, ratio: float = 5.0) -> str:
    """
    Join "shal l" -> "shall", "creditab le" -> "creditable", "t o" -> "to",
    using only evidence from the corpus itself (no dictionary):

    two alphabetic tokens a, b separated by exactly one space are joined when
      * one of them is short (<= 3 letters) — a stray space inside a word
        almost always strands a short fragment,
      * the joined word occurs in the corpus at least `min_joined` times, and
      * the joined word is at least `ratio` times more frequent than the rarer
        of the two pieces.
    The last rule is what keeps real phrases apart: "in to" stays "in to"
    because "in" and "to" are both far more common than "into"; "of the" never
    joins because "ofthe" doesn't occur.
    """
    if not vocab or not text:
        return text
    parts = _WORD_OR_GAP_RE.findall(text)
    out: List[str] = []
    i = 0
    while i < len(parts):
        a = parts[i]
        if (
            a.isalpha()
            and i + 2 < len(parts)
            and parts[i + 1] == " "
            and parts[i + 2].isalpha()
        ):
            b = parts[i + 2]
            if min(len(a), len(b)) <= 3:
                joined = (a + b).lower()
                n_joined = vocab.get(joined, 0)
                rarer = min(vocab.get(a.lower(), 0), vocab.get(b.lower(), 0))
                if n_joined >= min_joined and n_joined >= ratio * max(rarer, 1):
                    parts[i + 2] = a + b       # may join again with the next piece
                    i += 2
                    continue
        out.append(a)
        i += 1
    return "".join(out)


# ── Sentence splitting ────────────────────────────────────────────────────

# Abbreviations that end in a period but do not end a sentence in BIR text.
_ABBREVIATIONS = {
    "no", "nos", "sec", "secs", "art", "arts", "par", "pars", "para", "rev",
    "vs", "v", "e.g", "i.e", "etc", "inc", "co", "corp", "ltd", "phil",
    "p", "pp", "st", "jr", "sr", "mr", "mrs", "ms", "dr", "atty", "hon",
    "gen", "dept", "govt", "approx", "jan", "feb", "mar", "apr", "jun",
    "jul", "aug", "sept", "sep", "oct", "nov", "dec", "fig", "vol", "ch",
    "r.a", "p.d", "e.o", "b.p", "c.a", "no.s",
}

# Candidate boundary: sentence punctuation (or ";") then whitespace, then
# something that can start a sentence or a list item.
_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[\"“'(\[]?[A-Z0-9₱(•])|(?<=;)\s+")
MIN_SENTENCE_CHARS = 25


def _ends_with_abbreviation(fragment: str) -> bool:
    m = re.search(r"([A-Za-z][A-Za-z.]*)\.\s*$", fragment)
    if not m:
        return False
    word = m.group(1).lower()
    if word in _ABBREVIATIONS:
        return True
    return len(word) == 1  # an initial: "A." "J."


def split_sentences(text: str, min_chars: int = MIN_SENTENCE_CHARS) -> List[str]:
    """
    Split text into sentences: on paragraph breaks and list-item lines, then
    on ". ", "? ", "! " and "; " (list items in BIR issuances are usually
    separated by ";"), never after a known abbreviation ("No.", "Sec.",
    "R.A."). Fragments shorter than `min_chars` (a stray "a." or "SUBJECT :")
    are merged into the following sentence so they don't count as sentences
    of their own.
    """
    if not text or not text.strip():
        return []
    pieces: List[str] = []
    for block in re.split(r"\n+", text):
        block = re.sub(r"\s+", " ", block).strip()
        if not block:
            continue
        start = 0
        for m in _BOUNDARY_RE.finditer(block):
            candidate = block[start:m.start()]
            if _ends_with_abbreviation(candidate):
                continue
            pieces.append(candidate.strip())
            start = m.end()
        pieces.append(block[start:].strip())

    merged: List[str] = []
    carry = ""
    for p in pieces:
        if not p:
            continue
        p = f"{carry} {p}".strip() if carry else p
        carry = ""
        if len(p) < min_chars:
            carry = p
            continue
        merged.append(p)
    if carry:
        if merged:
            merged[-1] = f"{merged[-1]} {carry}"
        else:
            merged.append(carry)
    return merged


def _hard_split(sentence: str, max_chars: int) -> List[str]:
    """Split one over-long sentence at ", " / " " so no piece exceeds max_chars."""
    parts: List[str] = []
    rest = sentence
    while len(rest) > max_chars:
        cut = rest.rfind(", ", 0, max_chars)
        if cut < max_chars // 2:
            cut = rest.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        parts.append(rest[: cut + 1].strip())
        rest = rest[cut + 1:].strip()
    if rest:
        parts.append(rest)
    return parts


def chunk_sentences(sentences: List[str], max_chars: int = 800, overlap: int = 1) -> List[str]:
    """
    Pack whole sentences into chunks of at most `max_chars` characters. Each
    chunk after the first starts with the last `overlap` sentence(s) of the
    previous one, so a provision whose lead-in and list item straddle a
    boundary is still readable in one chunk. A single sentence longer than
    max_chars is split at commas/spaces as a last resort.
    """
    units: List[str] = []
    for s in sentences:
        units.extend(_hard_split(s, max_chars) if len(s) > max_chars else [s])

    chunks: List[str] = []
    current: List[str] = []
    size = 0
    for u in units:
        added = len(u) + (1 if current else 0)
        if current and size + added > max_chars:
            chunks.append(" ".join(current))
            tail = current[-overlap:] if overlap else []
            # Don't let the overlap alone fill the next chunk.
            while tail and sum(len(t) + 1 for t in tail) + len(u) > max_chars:
                tail = tail[1:]
            current, size = list(tail), sum(len(t) + 1 for t in tail)
            added = len(u) + (1 if current else 0)
        current.append(u)
        size += added
    if current:
        chunks.append(" ".join(current))
    return chunks


# ── Issuance identifiers ─────────────────────────────────────────────────

_LONG_FORMS = [
    (r"revenue\s+regulations?", "RR"),
    (r"revenue\s+memorandum\s+circulars?", "RMC"),
    (r"revenue\s+memorandum\s+orders?", "RMO"),
    (r"revenue\s+memorandum\s+rulings?", "RMR"),
    (r"revenue\s+administrative\s+orders?", "RAO"),
    (r"revenue\s+delegation\s+authority\s+orders?", "RDAO"),
    (r"revenue\s+audit\s+memorandum\s+orders?", "RAMO"),
]
_ABBREV = r"RDAO|RAMO|RMC|RMO|RMR|RAO|RR"
_NUMBER = r"(?:No\.?|Nos\.?|Number)?\s*(\d{1,4})[A-Za-z]?\s*-\s*((?:19|20)\d{2})(?!\d)"

_ABBREV_RE = re.compile(rf"(?<![A-Za-z])({_ABBREV})\.?\s*{_NUMBER}", re.IGNORECASE)
_LONG_RE = re.compile(
    "(" + "|".join(p for p, _ in _LONG_FORMS) + r")\s*(?:\((?:" + _ABBREV + r")\))?\s*" + _NUMBER,
    re.IGNORECASE,
)


def _long_to_abbrev(phrase: str) -> Optional[str]:
    for pattern, abbrev in _LONG_FORMS:
        if re.fullmatch(pattern, phrase.strip(), re.IGNORECASE):
            return abbrev
    return None


def parse_issuance_ids(text: str) -> List[str]:
    """
    Every BIR issuance number in `text`, canonicalized and de-duplicated in
    order of first appearance:

        "RMC No. 8-2026"                      -> "RMC-8-2026"
        "Revenue Regulations (RR) No. 7-2021" -> "RR-7-2021"
        "RR 14-2022", "RR No. 014 - 2022"     -> "RR-14-2022"
        "3089RMR 2-2002" (a filename)         -> "RMR-2-2002"

    Republic Acts and Executive Orders are deliberately not parsed: they are
    numbered without a year ("RA No. 11976"), so a bare number would collide
    with amounts and section numbers.
    """
    found: List[tuple] = []
    for m in _ABBREV_RE.finditer(text or ""):
        found.append((m.start(), f"{m.group(1).upper()}-{int(m.group(2))}-{m.group(3)}"))
    for m in _LONG_RE.finditer(text or ""):
        abbrev = _long_to_abbrev(m.group(1))
        if abbrev:
            found.append((m.start(), f"{abbrev}-{int(m.group(2))}-{m.group(3)}"))
    out: List[str] = []
    for _pos, ident in sorted(found):
        if ident not in out:
            out.append(ident)
    return out


def issuance_title(source: str) -> str:
    """
    Human-readable issuance name from a PDF path:
    "data\\2024\\RMC No. 34-2024.pdf" -> "RMC No. 34-2024" (either separator).
    """
    name = re.split(r"[\\/]", str(source or ""))[-1]
    name = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE).strip()
    return name or "Unknown source"


_SUBJECT_RE = re.compile(r"SUBJECT\s*:\s*(.+?)(?:\n\n|\bTO\s*:|$)", re.IGNORECASE | re.DOTALL)


def document_summary(first_page_text: str, max_chars: int = 220) -> str:
    """
    A one-line description of a whole issuance, taken from its first page:
    the SUBJECT line of a full issuance, or else the first sentence (a
    Digest's first sentence is "REVENUE MEMORANDUM CIRCULAR NO. 116-2024
    issued on ... clarifies ..."). Used only as context for embedding and
    BM25 — it is never shown to the generator as evidence.
    """
    text = first_page_text or ""
    m = _SUBJECT_RE.search(text)
    summary = m.group(1) if m else (split_sentences(text) or [""])[0]
    summary = re.sub(r"\s+", " ", summary).strip()
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0] + "…"
    return summary


def context_header(title: str, summary: str = "") -> str:
    """The text prepended to a chunk for embedding/BM25 only."""
    return f"{title} — {summary}" if summary else title


def jaccard(a: str, b: str) -> float:
    """Word-set overlap of two texts (used for near-duplicate removal)."""
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def overlap_coefficient(a: str, b: str) -> float:
    """|A∩B| / min(|A|,|B|) on word sets — 1.0 when one text contains the other."""
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def normalize_for_match(text: str) -> str:
    """Whitespace/case-folded text for substring checks."""
    return re.sub(r"\s+", " ", text or "").strip().lower()


def ids_in_metadata(meta: Dict) -> List[str]:
    """Issuance ids stored on a chunk (comma-joined string in Chroma metadata)."""
    raw = (meta or {}).get("issuance_ids") or ""
    return [x for x in str(raw).split(",") if x]
