import os
import uuid

import chainlit as cl
import chromadb
import ollama
from chromadb.utils import embedding_functions
from dotenv import load_dotenv


load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    raise ValueError("API key mancante. Controlla il file .env")

documents_dir = "resumes"

documents = []
metadatas = []
ids = []

for filename in os.listdir(documents_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(documents_dir, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            chunks = file.read().replace("\n", ".").split("### ")

            for chunk in chunks:
                if chunk.strip():
                    documents.append(chunk)
                    metadatas.append({"source": filename})
                    ids.append(str(uuid.uuid4()))

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key,
    model_name="text-embedding-3-small"
)

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="CVs",
    embedding_function=openai_ef
)

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)


def leggi_prime_100_righe(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        righe = []

        for index, riga in enumerate(file):
            if index < 100:
                righe.append(riga.strip())
            else:
                break

    return righe


@cl.on_chat_start
def on_chat_start():
    cl.user_session.set(
        "messages",
        [
            {
                "role": "system",
                "content": (
                    "Sei un assistente specializzato nel mondo HR. "
                    "Rispondi in modo professionale, sintetico e pragmatico. "
                    "Il tuo ruolo è individuare il candidato ideale rispetto alle richieste dell'utente."
                ),
            }
        ],
    )


@cl.on_message
async def handle_message(message: cl.Message):
    user_question = message.content

    results = collection.query(
        query_texts=[user_question],
        n_results=1
    )

    filename = results["metadatas"][0][0]["source"]
    file_path = os.path.join(documents_dir, filename)

    context_nome_candidato = leggi_prime_100_righe(file_path)

    nome_response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": (
                    "Dato il seguente curriculum, individua il nome e cognome del candidato. "
                    "Rispondi solo con nome e cognome, senza spiegazioni.\n\n"
                    f"{context_nome_candidato}"
                ),
            }
        ],
    )

    nome = nome_response["message"]["content"]

    context = (
        f"Nome file: {filename}\n"
        f"Chunk più rilevante: {results['documents'][0][0]}"
    )

    prompt = (
        f"Domanda utente: {user_question}\n\n"
        f"Contesto recuperato dal RAG:\n{context}\n\n"
        f"Nome candidato individuato: {nome}\n\n"
        "Spiega che nel file individuato c'è il profilo più adatto. "
        "Indica il nome del file e il nome del candidato. "
        "Argomenta la scelta usando solo il contenuto del contesto. "
        "Se non trovi corrispondenza, non inventare informazioni."
    )

    messages = cl.user_session.get("messages", [])
    messages.append({"role": "user", "content": prompt})

    response_message = cl.Message(content="")
    await response_message.send()

    try:
        stream = ollama.chat(
            model="llama3.2",
            messages=messages,
            stream=True
        )

        for chunk in stream:
            await response_message.stream_token(chunk["message"]["content"])

        messages.append(
            {
                "role": "assistant",
                "content": response_message.content
            }
        )

        await response_message.update()

    except Exception as error:
        error_message = f"Errore durante la generazione della risposta: {str(error)}"
        await cl.Message(content=error_message).send()

    cl.user_session.set("messages", messages)