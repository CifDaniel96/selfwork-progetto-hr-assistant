import os
import uuid

from .config import DOCUMENTS_DIR


def load_documents():
    documents = []
    metadatas = []
    ids = []

    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(DOCUMENTS_DIR, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                chunks = file.read().replace("\n", ".").split("### ")

                for chunk in chunks:
                    if chunk.strip():
                        documents.append(chunk.strip())
                        metadatas.append({"source": filename})
                        ids.append(str(uuid.uuid4()))

    return documents, metadatas, ids