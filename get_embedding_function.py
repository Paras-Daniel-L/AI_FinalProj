import os
from dotenv import load_dotenv
from langchain_community.embeddings import JinaEmbeddings

load_dotenv()

def get_embedding_function():
    embeddings = JinaEmbeddings(
        jina_api_key=os.environ.get("JINA_API_KEY"),
        model_name="jina-embeddings-v3",
    )
    return embeddings