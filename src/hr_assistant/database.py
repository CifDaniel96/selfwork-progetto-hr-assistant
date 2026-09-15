import chromadb
from chromadb.utils import embedding_functions

from .config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    PERSISTENT_DIR,
)


class Database:
    def __init__(self):
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=EMBEDDING_MODEL
        )

        self.client = chromadb.PersistentClient(
            path=PERSISTENT_DIR
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.openai_ef
        )

    def add_documents(self, documents, metadatas, ids):
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query_text, n_results=1):
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

    def get_tracked_files(self):
        result = self.collection.get()
        tracked_files = {}

        if result and result["metadatas"]:
            for metadata in result["metadatas"]:
                source = metadata["source"]

                if source not in tracked_files:
                    tracked_files[source] = {
                        "hash": metadata["hash"],
                        "last_modified": metadata["last_modified"],
                        "source": source
                    }

        return tracked_files

    def remove_document_by_source(self, source):
        result = self.collection.get(
            where={"source": source}
        )

        if result and result["ids"]:
            self.collection.delete(
                ids=result["ids"]
            )