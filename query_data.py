import argparse
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from get_embedding_function import get_embedding_function

load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query_text)


def query_rag(query_text: str):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    results = db.similarity_search_with_score(query_text, k=5)

    if not results:
        print("⚠️ No relevant results found in the database.")
        return ""

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text, question=query_text
    )

    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    response_text = model.invoke(prompt).content

    sources = [doc.metadata.get("id", None) for doc, _ in results]
    print(f"Response: {response_text}\nSources: {sources}")
    return response_text


if __name__ == "__main__":
    main()