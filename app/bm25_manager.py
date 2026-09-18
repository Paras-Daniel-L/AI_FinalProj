import os
import pickle
from typing import Optional
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "chroma/bm25_index.pkl")


def build_and_save_bm25(db: Chroma, save_path: str = BM25_INDEX_PATH) -> Optional[BM25Retriever]:
    """
    Extracts all documents from ChromaDB, builds the BM25 index once,
    and pickles it to disk to prevent OOM memory exhaustion.
    """
    all_data = db.get(include=["documents", "metadatas"])
    docs = all_data.get("documents", [])
    metadatas = all_data.get("metadatas", [])

    if not docs:
        if os.path.exists(save_path):
            os.remove(save_path)
        return None

    all_docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(docs, metadatas)
    ]

    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = 5

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(bm25, f)

    return bm25


def load_bm25(save_path: str = BM25_INDEX_PATH) -> Optional[BM25Retriever]:
    """Loads the pre-indexed BM25 instance from disk into memory."""
    if not os.path.exists(save_path):
        return None
    try:
        with open(save_path, "rb") as f:
            bm25 = pickle.load(f)
            bm25.k = 5
            return bm25
    except Exception as e:
        print(f"⚠️ Failed to load BM25 index from {save_path}: {e}")
        return None