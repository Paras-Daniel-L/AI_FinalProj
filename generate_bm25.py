from langchain_chroma import Chroma
from app.embeddings import get_embedding_function
from app.bm25_manager import build_and_save_bm25

if __name__ == "__main__":
    print("Building BM25 index from ChromaDB...")
    db = Chroma(persist_directory="chroma", embedding_function=get_embedding_function())
    retriever = build_and_save_bm25(db)

    if retriever:
        print("✅ BM25 index pickled successfully at chroma/bm25_index.pkl!")
    else:
        print("⚠️ No documents found in ChromaDB to index.")