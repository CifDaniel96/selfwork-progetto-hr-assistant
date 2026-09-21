import os

from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

from .config import (
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    LOCAL_EMBEDDING_MODEL,
    LOCAL_MODEL_PATH,
    OPENAI_API_KEY,
)


class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.provider = EMBEDDING_PROVIDER

        if self.provider == "openai":
            self._setup_openai()
        elif self.provider == "local":
            self._setup_local_model()
        else:
            raise ValueError(
                f"Embedding provider non supportato: {self.provider}"
            )

    def _setup_openai(self):
        self.embedding_function = (
            embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name=EMBEDDING_MODEL,
            )
        )

    def _setup_local_model(self):
        if os.path.exists(LOCAL_MODEL_PATH):
            self.embedding_function = SentenceTransformer(
                LOCAL_MODEL_PATH
            )
        else:
            self.embedding_function = SentenceTransformer(
                LOCAL_EMBEDDING_MODEL
            )

            self.embedding_function.save(
                LOCAL_MODEL_PATH
            )

    def __call__(self, input):
        if self.provider == "openai":
            return self.embedding_function(input)

        if self.provider == "local":
            return self.embedding_function.encode(input).tolist()

        raise ValueError(
            f"Embedding provider non supportato: {self.provider}"
        )