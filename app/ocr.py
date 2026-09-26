"""
OCR fallback for scanned / image-only PDF pages.

PyPDF only reads a PDF's text layer, so a scanned page comes back empty and
contributes no chunks — the question "what does RMC 24-2026 say?" then can
never be answered. This module OCRs just those empty pages during ingestion
(see database.load_documents) so their text becomes searchable.

Engine: Tesseract (via pytesseract) with PyMuPDF rendering the page to an
image. Tesseract was chosen over pip-only engines after testing: RapidOCR
dropped the spaces between words on capitalized lines ("REPUBLICOFTHE...")
which breaks BM25 tokens and embeddings for exactly the title/heading lines
where issuance numbers live.

Setup (one time):
    pip install pymupdf pytesseract pillow
    Install the Tesseract program itself (Windows installer:
    https://github.com/UB-Mannheim/tesseract/wiki). If it isn't on PATH, set
    TESSERACT_CMD in .env, e.g.
    TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe

Settings (env vars, all optional):
    OCR_ENABLED=1        set 0 to skip OCR entirely
    OCR_LANG=eng         Tesseract language(s), e.g. eng+fil if installed
    OCR_DPI=300          render resolution
    OCR_MIN_CHARS=20     a page with fewer text characters than this is OCR'd
    OCR_CACHE_DIR=ocr_cache   results are cached by file hash so re-ingesting
                              (--reset) doesn't redo OCR
    OCR_IMAGE_COVERAGE=0.8    a page whose images cover >= this fraction of it
                              is treated as a scan
    OCR_REPLACE_BAD_LAYERS=1  for scanned pages that ALREADY carry a text layer
                              (a hidden OCR from another tool — often garbage,
                              e.g. "REPUBLI( OF THL PHILIPP:^.'ES"), OCR the
                              page and keep whichever text reads cleaner

OCR text can contain misread characters (digits especially). Pages OCR'd by
this module are marked metadata["ocr"] = True so the UI/source labels can say so.
"""

import hashlib
import json
import os
import re
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv()

OCR_ENABLED = os.environ.get("OCR_ENABLED", "1").strip() not in ("0", "false", "False", "")
OCR_LANG = os.environ.get("OCR_LANG", "eng")
OCR_DPI = int(os.environ.get("OCR_DPI", "300"))
OCR_MIN_CHARS = int(os.environ.get("OCR_MIN_CHARS", "20"))
OCR_CACHE_DIR = os.environ.get("OCR_CACHE_DIR", "ocr_cache")
OCR_IMAGE_COVERAGE = float(os.environ.get("OCR_IMAGE_COVERAGE", "0.8"))
OCR_REPLACE_BAD_LAYERS = os.environ.get("OCR_REPLACE_BAD_LAYERS", "1").strip() not in ("0", "false", "False", "")
# The OCR text must beat the existing layer's quality score by this much
# before it replaces it (avoids swapping one decent text for another).
OCR_QUALITY_MARGIN = 0.05

_WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

_unavailable_reason: Optional[str] = None  # set once if OCR can't run; then skipped quietly
_pytesseract = None
_fitz = None


def needs_ocr(page_text: str) -> bool:
    """True when a page's extracted text is too short to be a real text layer."""
    return len((page_text or "").strip()) < OCR_MIN_CHARS


_CLEAN_TOKEN_RE = re.compile(
    r"^[\(\[\"'“‘]*([A-Za-zÀ-ÿ]{2,}|\d[\d,.]*%?|[A-Za-z]|[IVXivx]+)[\)\]\"'”’.,;:?!]*$"
)


def text_quality(text: str) -> float:
    """
    Rough 0..1 score of how much of `text` looks like real words/numbers:
    the share of whitespace-separated tokens that are a plain word, a number
    or a bare letter (optionally wrapped in brackets/punctuation). Garbled
    OCR ("Otrc1n", "PHILIPP:^.'ES", "l4AR") scores low. It's only meaningful
    for COMPARING two readings of the same page, not as an absolute grade —
    table-heavy born-digital pages can score under 0.8 too.
    """
    tokens = (text or "").split()
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if _CLEAN_TOKEN_RE.match(t)) / len(tokens)


def image_coverage(pdf_path: str, page_index: int) -> float:
    """Fraction of the page's area covered by embedded images (~1.0 for a scan)."""
    if not _init():
        return 0.0
    try:
        with _fitz.open(pdf_path) as pdf:
            page = pdf[page_index]
            area = sum(
                (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1])
                for b in page.get_image_info()
            )
            return area / max(page.rect.width * page.rect.height, 1.0)
    except Exception:
        return 0.0


def title_from_path(pdf_path: str) -> str:
    """"data/2026/RMO No. 9-2026.pdf" -> "RMO No. 9-2026"."""
    name = re.split(r"[\\/]", pdf_path)[-1]
    return re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE).strip()


def choose_page_text(pdf_path: str, page_index: int, layer_text: str):
    """
    Decide the text to index for one page. Returns (text, how) where `how` is
    "layer" (unchanged), "ocr" (page had no text layer) or "ocr_replaced"
    (a poor hidden text layer on a scanned page was replaced).

    - No/very little text layer  -> OCR it.
    - Text layer present BUT the page is a full-page image (a scan carrying a
      hidden OCR layer) -> OCR it too, and use the OCR only if it reads
      clearly cleaner than the layer (text_quality margin). Born-digital pages
      have no full-page image, so they are never re-OCR'd (no cost).
    OCR'd text is prefixed with the document title from the filename: the
    issuance number in a scanned header is often misread ("NO.()_f) 9 - 20 26"),
    and this keeps the document findable by name.
    """
    if not OCR_ENABLED:
        return layer_text, "layer"

    if needs_ocr(layer_text):
        text = ocr_page(pdf_path, page_index)
        if text:
            return f"{title_from_path(pdf_path)}\n{text}", "ocr"
        return layer_text, "layer"

    if OCR_REPLACE_BAD_LAYERS and image_coverage(pdf_path, page_index) >= OCR_IMAGE_COVERAGE:
        text = ocr_page(pdf_path, page_index)
        if text and text_quality(text) > text_quality(layer_text) + OCR_QUALITY_MARGIN:
            return f"{title_from_path(pdf_path)}\n{text}", "ocr_replaced"
    return layer_text, "layer"


def _init() -> bool:
    """Import the OCR stack and locate Tesseract. Returns False (and remembers why) if unusable."""
    global _pytesseract, _fitz, _unavailable_reason
    if _unavailable_reason:
        return False
    if _pytesseract is not None:
        return True
    try:
        try:
            import pymupdf as fitz  # newer name
        except ImportError:
            import fitz  # older name
        import pytesseract
        from PIL import Image  # noqa: F401  (needed to hand the page image to pytesseract)
    except ImportError as e:
        _unavailable_reason = (
            f"missing Python package ({e.name}). Run: pip install pymupdf pytesseract pillow"
        )
        return False

    cmd = os.environ.get("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    elif os.name == "nt" and os.path.exists(_WINDOWS_TESSERACT):
        pytesseract.pytesseract.tesseract_cmd = _WINDOWS_TESSERACT
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        _unavailable_reason = (
            "the Tesseract program was not found. Install it "
            "(https://github.com/UB-Mannheim/tesseract/wiki) and either add it to PATH "
            "or set TESSERACT_CMD in .env"
        )
        return False

    _pytesseract, _fitz = pytesseract, fitz
    return True


def _file_key(pdf_path: str) -> str:
    h = hashlib.sha1()
    with open(pdf_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _cache_file(pdf_path: str) -> str:
    return os.path.join(OCR_CACHE_DIR, f"{_file_key(pdf_path)}.json")


def _load_cache(path: str) -> Dict[str, str]:
    """Cached page texts, only if they were made with the current OCR settings."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("lang") == OCR_LANG and data.get("dpi") == OCR_DPI:
            return data.get("pages", {})
    except (OSError, ValueError):
        pass
    return {}


def _save_cache(path: str, pages: Dict[str, str]) -> None:
    os.makedirs(OCR_CACHE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"lang": OCR_LANG, "dpi": OCR_DPI, "pages": pages}, f, ensure_ascii=False)


# Tesseract often reads the peso sign as "®" or "#" ("®3.580 Trillion",
# "#470.086 Billion"). Only convert when followed by a money-shaped number
# (thousands commas, a decimal, or a magnitude word) so real "#3"-style
# numbering is left alone.
_PESO_RE = re.compile(
    r"[®#]\s?(?=\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\d+\.\d+\b|\d+\s*(?:Trillion|Billion|Million|thousand)\b)"
)


def _clean(text: str) -> str:
    text = _PESO_RE.sub("₱", text)
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def ocr_page(pdf_path: str, page_index: int) -> Optional[str]:
    """
    OCR one page (0-indexed, matching PyPDF's `page` metadata) of `pdf_path`.

    Returns the text (possibly "" if the page is genuinely blank), or None if
    OCR is disabled/unavailable or this page failed — never raises, so a bad
    scan or a missing Tesseract can't abort ingestion of the rest of the
    corpus.
    """
    if not OCR_ENABLED or not _init():
        return None

    try:
        cache_path = _cache_file(pdf_path)
        pages = _load_cache(cache_path)
        key = str(page_index)
        if key in pages:
            return pages[key]

        from PIL import Image

        with _fitz.open(pdf_path) as pdf:
            pix = pdf[page_index].get_pixmap(dpi=OCR_DPI)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        text = _clean(_pytesseract.image_to_string(image, lang=OCR_LANG, config="--psm 3"))

        pages[key] = text
        _save_cache(cache_path, pages)
        return text
    except Exception as e:
        print(f"     ⚠️  OCR failed for {pdf_path} page {page_index + 1}: {e}")
        return None


def unavailable_reason() -> Optional[str]:
    """Why OCR couldn't run (None if it can, or hasn't been tried yet)."""
    return _unavailable_reason
