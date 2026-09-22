import hashlib
import mimetypes
import os
import shutil
import tempfile
import uuid
from zipfile import ZipFile

from markitdown import MarkItDown

from .config import DOCUMENTS_DIR
from .semantic_chunking import SemanticChunking


SUPPORTED_EXTENSIONS = {
    ".txt": "text",
    ".pdf": "document",
    ".doc": "document",
    ".docx": "document",
    ".ppt": "presentation",
    ".pptx": "presentation",
    ".xls": "spreadsheet",
    ".xlsx": "spreadsheet",
    ".html": "web",
    ".htm": "web",
    ".csv": "data",
    ".json": "data",
    ".xml": "data",
    ".zip": "archive",
}


markdown_converter = MarkItDown()


def calculate_file_hash(file_path):
    hasher = hashlib.md5()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def get_document_metadata(file_path):
    extension = os.path.splitext(file_path)[1].lower()

    return {
        "hash": calculate_file_hash(file_path),
        "last_modified": os.path.getmtime(file_path),
        "source": os.path.basename(file_path),
        "file_type": SUPPORTED_EXTENSIONS.get(extension, "unknown"),
        "mime_type": (
            mimetypes.guess_type(file_path)[0]
            or "application/octet-stream"
        ),
        "extension": extension,
    }


def convert_to_markdown(file_path):
    try:
        result = markdown_converter.convert(file_path)

        return result.text_content

    except Exception as error:
        print(
            f"Errore durante la conversione di "
            f"{file_path}: {str(error)}"
        )

        return ""


def process_zip_file(file_path):
    results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_absolute = os.path.abspath(temp_dir)

        with ZipFile(file_path, "r") as zip_file:
            for member in zip_file.infolist():
                if member.is_dir():
                    continue

                extracted_path = os.path.abspath(
                    os.path.join(
                        temp_dir,
                        member.filename
                    )
                )

                if not extracted_path.startswith(
                    temp_dir_absolute + os.sep
                ):
                    print(
                        f"File ZIP ignorato per percorso non sicuro: "
                        f"{member.filename}"
                    )
                    continue

                extension = os.path.splitext(
                    member.filename
                )[1].lower()

                if extension not in SUPPORTED_EXTENSIONS:
                    continue

                if extension == ".zip":
                    continue

                os.makedirs(
                    os.path.dirname(extracted_path),
                    exist_ok=True
                )

                with zip_file.open(member) as source:
                    with open(extracted_path, "wb") as destination:
                        shutil.copyfileobj(
                            source,
                            destination
                        )

                content = convert_to_markdown(
                    extracted_path
                )

                if content:
                    results.append(
                        (
                            os.path.basename(member.filename),
                            content
                        )
                    )

    return results


def load_document_chunks(file_path):
    extension = os.path.splitext(file_path)[1].lower()
    file_type = SUPPORTED_EXTENSIONS.get(extension)

    if not file_type:
        return []

    if file_type == "archive":
        zip_contents = process_zip_file(file_path)

        content = ""

        for filename, zip_content in zip_contents:
            content += (
                f"\n\nFile: {filename}\n"
                f"{zip_content}"
            )
    else:
        content = convert_to_markdown(file_path)

    if not content.strip():
        return []

    semantic_chunker = SemanticChunking()

    return semantic_chunker.chunk_text(content)


def sync_documents(database):
    tracked_files = database.get_tracked_files()

    current_files = {
        filename
        for filename in os.listdir(DOCUMENTS_DIR)
        if (
            os.path.isfile(
                os.path.join(
                    DOCUMENTS_DIR,
                    filename
                )
            )
            and os.path.splitext(filename)[1].lower()
            in SUPPORTED_EXTENSIONS
        )
    }

    tracked_filenames = set(tracked_files.keys())

    added = 0
    updated = 0
    removed = 0

    removed_files = tracked_filenames - current_files

    for filename in removed_files:
        database.remove_document_by_source(filename)
        removed += 1

    for filename in current_files:
        file_path = os.path.join(
            DOCUMENTS_DIR,
            filename
        )

        metadata = get_document_metadata(
            file_path
        )

        tracked_file = tracked_files.get(filename)

        if (
            tracked_file
            and tracked_file["hash"] == metadata["hash"]
        ):
            continue

        if tracked_file:
            database.remove_document_by_source(
                filename
            )
            updated += 1
        else:
            added += 1

        chunks = load_document_chunks(
            file_path
        )

        documents = []
        metadatas = []
        ids = []

        for chunk in chunks:
            if not chunk.strip():
                continue

            documents.append(chunk)
            metadatas.append(metadata)
            ids.append(str(uuid.uuid4()))

        if documents:
            database.add_documents(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

    return added, updated, removed