import hashlib
import os
import uuid

from .config import DOCUMENTS_DIR


def calculate_file_hash(file_path):
    hasher = hashlib.md5()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def load_document_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        chunks = file.read().replace("\n", ".").split("### ")

    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]


def sync_documents(database):
    tracked_files = database.get_tracked_files()

    current_files = {
        filename
        for filename in os.listdir(DOCUMENTS_DIR)
        if filename.endswith(".txt")
    }

    tracked_filenames = set(tracked_files.keys())

    added = 0
    updated = 0
    removed = 0

    # Rimuove dal database i file che non esistono più
    removed_files = tracked_filenames - current_files

    for filename in removed_files:
        database.remove_document_by_source(filename)
        removed += 1

    # Controlla file nuovi o modificati
    for filename in current_files:
        file_path = os.path.join(DOCUMENTS_DIR, filename)

        file_hash = calculate_file_hash(file_path)
        last_modified = os.path.getmtime(file_path)

        tracked_file = tracked_files.get(filename)

        # File già presente e invariato
        if tracked_file and tracked_file["hash"] == file_hash:
            continue

        # File modificato
        if tracked_file:
            database.remove_document_by_source(filename)
            updated += 1
        else:
            # File nuovo
            added += 1

        chunks = load_document_chunks(file_path)

        documents = []
        metadatas = []
        ids = []

        for chunk in chunks:
            documents.append(chunk)

            metadatas.append(
                {
                    "source": filename,
                    "hash": file_hash,
                    "last_modified": last_modified,
                }
            )

            ids.append(str(uuid.uuid4()))

        if documents:
            database.add_documents(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

    return added, updated, removed