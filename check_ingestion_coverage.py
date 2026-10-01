"""
Ingestion coverage check.

Compares every PDF found anywhere under DATA_PATH against what's actually
indexed in ChromaDB (by distinct metadata["source"] value), and separately
flags PDFs that produce zero extractable text (near-certainly scanned/
image-only pages that OCR would be needed for).

Run from the project root (same place you run `python -m app.database`):

    python check_ingestion_coverage.py

Put this file directly in the project root, or in scripts/ and adjust the
two imports below to `from app.database import ...` / `from app.embeddings
import ...` accordingly — it already uses the package-qualified form,
assuming this file sits at the project root next to main.py.
"""

import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

from app.database import CHROMA_PATH, DATA_PATH
from app.embeddings import get_embedding_function


def main():
    # 1. Every PDF on disk, anywhere under DATA_PATH.
    disk_pdfs = []
    for root, _dirs, files in os.walk(DATA_PATH):
        for f in files:
            if f.lower().endswith(".pdf"):
                disk_pdfs.append(os.path.normpath(os.path.join(root, f)))
    disk_pdfs = sorted(disk_pdfs)
    print(f"📄 PDFs on disk under '{DATA_PATH}/': {len(disk_pdfs)}")

    # 2. Distinct source files actually indexed in Chroma.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    data = db.get(include=["metadatas"])
    indexed_sources = set()
    for meta in data["metadatas"]:
        src = meta.get("source")
        if src:
            indexed_sources.add(os.path.normpath(src))
    print(f"📦 Distinct source files indexed in Chroma: {len(indexed_sources)}")

    # 3. Diff: on disk but never made it into Chroma.
    missing = [p for p in disk_pdfs if p not in indexed_sources]
    print(f"\n🚫 On disk but NOT indexed ({len(missing)}):")
    for p in missing:
        print(f"   - {p}")
    if not missing:
        print("   (none — every PDF on disk has at least one chunk indexed)")

    # 4. Zero-extractable-text PDFs (scanned/image pages needing OCR).
    # This re-reads every PDF with PyPDFLoader, so it's slower than the
    # steps above — expect a few minutes for 150 files.
    print("\n🔍 Checking for scanned/zero-text PDFs (this re-reads every file)...")
    scanned = []
    for p in disk_pdfs:
        try:
            docs = PyPDFLoader(p).load()
            total_chars = sum(len(d.page_content.strip()) for d in docs)
            if total_chars == 0:
                scanned.append(p)
        except Exception as e:
            print(f"   ⚠️  couldn't read '{p}': {e}")
    print(f"\n🚫 Scanned / zero-extractable-text PDFs ({len(scanned)}):")
    for p in scanned:
        print(f"   - {p}")
    if not scanned:
        print("   (none — every PDF has at least some extractable text)")


if __name__ == "__main__":
    main()
