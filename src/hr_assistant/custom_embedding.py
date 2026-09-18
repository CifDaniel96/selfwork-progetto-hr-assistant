import os

import ollama
from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

from .config import (
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    LOCAL_EMBEDDING_MODEL,
    LOCAL_MODEL_PATH,
    OLLAMA_EMBEDDING_MODEL,
    OPENAI_API_KEY,
)


class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.provider = EMBEDDING_PROVIDER

        if self.provider == "openai":
            self._setup_openai()
        elif self.provider == "local":
            self._setup_local_model()
        elif self.provider == "ollama":
            self._setup_ollama()
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
            print(
                f"Modello locale trovato in '{LOCAL_MODEL_PATH}', "
                "caricamento in corso..."
            )

            self.embedding_function = SentenceTransformer(
                LOCAL_MODEL_PATH
            )
        else:
            print(
                f"Scaricamento di '{LOCAL_EMBEDDING_MODEL}' "
                "in corso..."
            )

            self.embedding_function = SentenceTransformer(
                LOCAL_EMBEDDING_MODEL
            )

            self.embedding_function.save(
                LOCAL_MODEL_PATH
            )

            print(
                f"Modello salvato in '{LOCAL_MODEL_PATH}'."
            )

    def _setup_ollama(self):
        print(
            f"Utilizzo embedding Ollama: "
            f"{OLLAMA_EMBEDDING_MODEL}"
        )

    def __call__(self, input):
        if self.provider == "openai":
            return self.embedding_function(input)

        if self.provider == "local":
            return self.embedding_function.encode(input).tolist()

        if self.provider == "ollama":
            return [
                ollama.embeddings(
                    model=OLLAMA_EMBEDDING_MODEL,
                    prompt=text,
                )["embedding"]
                for text in input
            ]