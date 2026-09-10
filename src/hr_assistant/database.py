import chromadb
from chromadb.utils import embedding_functions

from .config import COLLECTION_NAME, EMBEDDING_MODEL, OPENAI_API_KEY
from .document_processor import load_documents


def create_collection():
    documents, metadatas, ids = load_documents()

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL
    )

    chroma_client = chromadb.Client()

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef
    )

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    return collection