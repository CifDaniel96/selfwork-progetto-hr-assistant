import os

from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("API key mancante. Controlla il file .env")

DOCUMENTS_DIR = "resumes"
EMBEDDING_MODEL = "text-embedding-3-small"
OLLAMA_MODEL = "llama3.2"
COLLECTION_NAME = "CVs"