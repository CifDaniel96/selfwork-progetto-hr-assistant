import chainlit as cl
import ollama

from hr_assistant.config import OLLAMA_MODEL
from hr_assistant.database import create_collection
from hr_assistant.utils import build_context, build_prompt, leggi_prime_100_righe


collection = create_collection()


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
    retrieved_chunk = results["documents"][0][0]

    candidate_resume_rows = leggi_prime_100_righe(filename)

    candidate_name_response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "Dato il seguente curriculum, individua il nome e cognome del candidato. "
                    "Rispondi solo con nome e cognome, senza spiegazioni.\n\n"
                    f"{candidate_resume_rows}"
                ),
            }
        ],
    )

    candidate_name = candidate_name_response["message"]["content"]

    context = build_context(
        filename=filename,
        retrieved_chunk=retrieved_chunk,
        candidate_name=candidate_name
    )

    prompt = build_prompt(
        user_question=user_question,
        context=context
    )

    messages = cl.user_session.get("messages", [])
    messages.append({"role": "user", "content": prompt})

    response_message = cl.Message(content="")
    await response_message.send()

    try:
        stream = ollama.chat(
            model=OLLAMA_MODEL,
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