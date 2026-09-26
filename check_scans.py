"""
List PDFs under data/ that contain scanned pages, and say what ingestion will do.

Run from the project root (next to main.py):
    python check_scans.py

For every page that is (mostly) one big image it reports whether the page has
NO text layer (OCR will add it) or a hidden text layer of doubtful quality
(ingestion OCRs it and keeps whichever reads cleaner). This does not run OCR
itself, so it is fast. It only needs `pip install pymupdf`.

Why it matters: a scan with a garbage hidden text layer looks "indexed" in
check_ingestion_coverage.py (it has chunks), but those chunks are misspelled
gibberish. Files flagged "hidden text layer" only get fixed by a full
`python -m app.database --reset`, because their old chunk ids already exist.
"""

import os
import re
import sys

try:
    import pymupdf
except ImportError:
    sys.exit("Install PyMuPDF first: pip install pymupdf")

DATA_PATH = "data"
MIN_CHARS = int(os.environ.get("OCR_MIN_CHARS", "20"))
COVERAGE = float(os.environ.get("OCR_IMAGE_COVERAGE", "0.8"))

_TOKEN = re.compile(
    r"^[\(\[\"'“‘]*([A-Za-zÀ-ÿ]{2,}|\d[\d,.]*%?|[A-Za-z]|[IVXivx]+)[\)\]\"'”’.,;:?!]*$"
)


def quality(text: str) -> float:
    tokens = text.split()
    return sum(1 for t in tokens if _TOKEN.match(t)) / len(tokens) if tokens else 0.0


def coverage(page) -> float:
    area = sum(
        (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1])
        for b in page.get_image_info()
    )
    return area / max(page.rect.width * page.rect.height, 1.0)


def main() -> None:
    no_text, hidden_layer, scanned_files = [], [], 0
    total_files = 0
    for root, _dirs, files in os.walk(DATA_PATH):
        for name in sorted(files):
            if not name.lower().endswith(".pdf"):
                continue
            total_files += 1
            path = os.path.join(root, name)
            try:
                doc = pymupdf.open(path)
            except Exception as e:
                print(f"⚠️  cannot open {path}: {e}")
                continue
            empty_pages, layer_pages = [], []
            for i, page in enumerate(doc):
                text = page.get_text().strip()
                if len(text) < MIN_CHARS:
                    empty_pages.append(i + 1)
                elif coverage(page) >= COVERAGE:
                    layer_pages.append((i + 1, quality(text)))
            if empty_pages or layer_pages:
                scanned_files += 1
            if empty_pages:
                no_text.append((path, empty_pages, len(doc)))
            if layer_pages:
                hidden_layer.append((path, layer_pages, len(doc)))

    print(f"Scanned {total_files} PDF(s) under {DATA_PATH}/\n")

    print(f"🖼️  NO text layer — OCR will add these pages ({len(no_text)} file(s)):")
    for path, pages, n in no_text:
        print(f"   {path}: page(s) {pages} of {n}")

    print(f"\n🕵️  Scans WITH a hidden text layer ({len(hidden_layer)} file(s)) — quality is doubtful,")
    print("    ingestion OCRs these and keeps the cleaner text (low score = likely garbage):")
    for path, pages, n in hidden_layer:
        scores = ", ".join(f"p{p}={q:.2f}" for p, q in pages)
        print(f"   {path}: {scores}")

    if hidden_layer:
        print(
            "\n➡️  Files in the second group are already indexed with their old text. "
            "To replace it, install Tesseract and run:  python -m app.database --reset"
        )
    if not no_text and not hidden_layer:
        print("\n✅ No scanned pages found.")


if __name__ == "__main__":
    main()
