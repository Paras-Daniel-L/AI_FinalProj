import argparse
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv

from get_embedding_function import get_embedding_function
from classifier import build_classifier, classify_query  # 🆕

load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# 🆕 Train classifier once at startup (choose 'naive_bayes' or 'svm')
classifier_model = build_classifier(model_type="naive_bayes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query_text)


def query_rag(query_text: str):
    # ── Step 1: Classify the query ──────────────────────────────────────
    predicted_class, label, year_filter = classify_query(
        classifier_model, query_text
    )

    # ── Step 2: Route based on classification ───────────────────────────
    """if predicted_class == 0:
        print("💬 Chatbot: Musta! Ako ay isang BIR tax assistant. Magtanong ka ng tungkol sa buwis!")
        return ""

    if predicted_class == 4:
        print("🚫 Chatbot: This looks like a board game question. I only assist with real BIR tax laws!")
        return """""

    # ── Step 3: Tax query — proceed to RAG ──────────────────────────────
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    # ── Step 4: Apply year metadata filter if available ─────────────────
    if year_filter:
        print(f"🗂️  Filtering ChromaDB for year: {year_filter}")
        semantic_results = db.similarity_search(
            query_text,
            k=5,
            filter={"year": year_filter}   # metadata filter
        )
    else:
        semantic_results = db.similarity_search(query_text, k=5)

    # ── Step 5: BM25 keyword search ─────────────────────────────────────
    all_data = db.get(include=["documents", "metadatas"])
    all_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_data["documents"], all_data["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 5
    bm25_results = bm25_retriever.invoke(query_text)

    # ── Step 6: Merge & deduplicate results ─────────────────────────────
    seen = set()
    combined = []
    for doc in semantic_results + bm25_results:
        doc_id = doc.metadata.get("id")
        if doc_id not in seen:
            seen.add(doc_id)
            combined.append(doc)

    if not combined:
        print("⚠️ No relevant results found in the database.")
        return ""

    # ── Step 7: Generate answer via LLaMA ───────────────────────────────
    context_text = "\n\n---\n\n".join([doc.page_content for doc in combined])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text, question=query_text
    )

    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    response_text = model.invoke(prompt).content

    sources = [doc.metadata.get("id") for doc in combined]
    print(f"\n✅ Response: {response_text}")
    print(f"📚 Sources: {sources}")
    return response_text


if __name__ == "__main__":
    main()