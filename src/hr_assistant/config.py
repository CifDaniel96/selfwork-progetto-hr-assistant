import os

from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("API key mancante. Controlla il file .env")

DOCUMENTS_DIR = "resumes"
PERSISTENT_DIR = "data/chromadb"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_PROVIDER = "local"

LOCAL_EMBEDDING_MODEL = "all-mpnet-base-v2"
LOCAL_MODEL_PATH = "modelli/mio_modello"

OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_MODEL = "llama3.2"
COLLECTION_NAME = "CVs"