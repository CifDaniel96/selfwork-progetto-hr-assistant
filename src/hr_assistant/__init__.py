import chainlit as cl
import ollama

from hr_assistant.config import OLLAMA_MODEL
from hr_assistant.database import Database
from hr_assistant.document_processor import sync_documents
from hr_assistant.utils import build_prompt, leggi_prime_100_righe


database = Database()
sync_documents(database)


@cl.action_callback("db_stats")
async def show_db_stats(action: cl.Action):
    db_info = database.get_stats()

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "Descrivi in modo sintetico le statistiche del database "
                    "dei frammenti indicizzati dal sistema.\n\n"
                    f"{db_info}"
                ),
            }
        ],
    )

    await cl.Message(
        content=response["message"]["content"]
    ).send()


@cl.action_callback("db_reindex")
async def reindex_database(action: cl.Action):
    sync_documents(database)

    await cl.Message(
        content="Database reindicizzato con successo."
    ).send()


@cl.on_chat_start
async def on_chat_start():
    actions = [
        cl.Action(
            name="db_stats",
            icon="mouse-pointer-click",
            payload={"value": "db_stats"},
            label="Statistiche Database",
        ),
        cl.Action(
            name="db_reindex",
            icon="mouse-pointer-click",
            payload={"value": "db_reindex"},
            label="Reindex Database",
        ),
    ]

    await cl.Message(
        content="Informazioni del sistema:",
        actions=actions
    ).send()

    cl.user_session.set(
        "messages",
        [
            {
                "role": "system",
                "content": (
                    "Sei un assistente specializzato nel mondo HR. "
                    "Rispondi in modo professionale, sintetico e pragmatico. "
                    "Il tuo ruolo è individuare il candidato ideale "
                    "rispetto alle richieste dell'utente."
                ),
            }
        ],
    )


@cl.on_message
async def handle_message(message: cl.Message):
    user_question = message.content

    results = database.query(
        user_question,
        n_results=3
    )

    filename = results["metadatas"][0][0]["source"]
    retrieved_chunk = results["documents"][0][0]

    candidate_resume_rows = leggi_prime_100_righe(filename)[:10]
    candidate_info = "\n".join(candidate_resume_rows)

    context = (
        f"Nome file: {filename}\n\n"
        f"Paragrafo più significativo:\n{retrieved_chunk}\n\n"
        f"Informazioni del candidato:\n{candidate_info}"
    )

    prompt = build_prompt(
        user_question=user_question,
        context=context
    )

    messages = cl.user_session.get("messages", [])
    messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    response_message = cl.Message(content="")
    await response_message.send()

    try:
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            stream=True
        )

        for chunk in stream:
            await response_message.stream_token(
                chunk["message"]["content"]
            )

        messages.append(
            {
                "role": "assistant",
                "content": response_message.content
            }
        )

        await response_message.update()

    except Exception as error:
        error_message = (
            f"Errore durante la generazione della risposta: {str(error)}"
        )

        await cl.Message(
            content=error_message
        ).send()

    cl.user_session.set("messages", messages)