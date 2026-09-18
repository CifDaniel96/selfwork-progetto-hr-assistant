import chromadb

from .config import COLLECTION_NAME, PERSISTENT_DIR
from .custom_embedding import CustomEmbeddingFunction

class Database:
    def __init__(self):
        self.embedding_function = CustomEmbeddingFunction()

        self.client = chromadb.PersistentClient(
            path=PERSISTENT_DIR
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function
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

    def get_stats(self):
        result = self.collection.get()

        distinct_sources = {
            metadata["source"]
            for metadata in result["metadatas"]
        }

        total_files = len(distinct_sources)

        return f"""
            Nome Collezione: {self.collection.name}
            Numero totale Frammenti: {self.collection.count()}
            Numero Files Elaborati: {total_files}
        """