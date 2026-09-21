import os

from dotenv import load_dotenv


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "API key mancante. Controlla il file .env"
    )


DOCUMENTS_DIR = "resumes"
PERSISTENT_DIR = "data/chromadb"

COLLECTION_NAME = "CVs"



EMBEDDING_PROVIDER = "openai"
EMBEDDING_MODEL = "text-embedding-3-small"



AI_API_URL = "https://api.openai.com/v1/"
LLM_MODEL = "gpt-4o"
LLM_MODEL_LOW = "gpt-4o-mini"


LOCAL_EMBEDDING_MODEL = "all-mpnet-base-v2"
LOCAL_MODEL_PATH = "modelli/mio_modello"

