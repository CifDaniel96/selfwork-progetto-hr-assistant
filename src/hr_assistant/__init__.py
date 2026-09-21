import chainlit as cl

from hr_assistant.database import Database
from hr_assistant.document_processor import sync_documents
from hr_assistant.utils import (
    build_prompt,
    chat,
    classify_intent,
    get_db_stats_response,
    leggi_prime_100_righe,
)


database = Database()
sync_documents(database)


@cl.action_callback("db_stats")
async def show_db_stats(action: cl.Action):
    db_info = database.get_stats()

    response = get_db_stats_response(db_info)

    await cl.Message(
        content=response
    ).send()


@cl.action_callback("db_reindex")
async def reindex_database(action: cl.Action):
    added, updated, removed = sync_documents(database)

    await cl.Message(
        content=(
            "Database reindicizzato con successo.\n\n"
            f"Added: {added}\n"
            f"Updated: {updated}\n"
            f"Removed: {removed}"
        )
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

    cl.user_session.set("last_cv_context", "")
    cl.user_session.set("last_cv_header", "")


@cl.on_message
async def handle_message(message: cl.Message):
    user_question = message.content

    try:
        intent = classify_intent(user_question)
    except ValueError as error:
        await cl.Message(
            content=f"Non riesco a classificare la richiesta: {str(error)}"
        ).send()
        return

    messages = cl.user_session.get("messages", [])

    context = ""
    candidate_info = ""
    save_candidate_context = False

    if intent == "search_cv":
        results = database.query(
            user_question,
            n_results=3
        )

        if (
            not results
            or not results.get("documents")
            or not results["documents"][0]
        ):
            await cl.Message(
                content=(
                    "Nessun curriculum trovato per la tua richiesta. "
                    "Prova a specificare meglio competenze o esperienza."
                )
            ).send()
            return

        filename = results["metadatas"][0][0]["source"]
        retrieved_chunk = results["documents"][0][0]

        candidate_resume_rows = leggi_prime_100_righe(filename)[:20]
        candidate_info = "\n".join(candidate_resume_rows)

        context = (
            f"Nome file: {filename}\n\n"
            f"Paragrafo più significativo:\n"
            f"{retrieved_chunk}\n\n"
            f"Informazioni del candidato:\n"
            f"{candidate_info}"
        )

        prompt = build_prompt(
            user_question=user_question,
            context=context
        )

        save_candidate_context = True

    elif intent == "info_cv":
        context = cl.user_session.get(
            "last_cv_context",
            ""
        )

        candidate_info = cl.user_session.get(
            "last_cv_header",
            ""
        )

        if not context:
            await cl.Message(
                content=(
                    "Non c'è ancora un candidato selezionato. "
                    "Cerca prima un profilo e poi chiedimi informazioni "
                    "specifiche su quel candidato."
                )
            ).send()
            return

        prompt = (
            f"Domanda utente: {user_question}\n\n"
            f"Contesto del candidato già individuato:\n"
            f"{context}\n\n"
            f"Informazioni aggiuntive del CV:\n"
            f"{candidate_info}\n\n"
            "Rispondi esclusivamente usando le informazioni presenti "
            "nel contesto del candidato. "
            "Fornisci in modo diretto solo l'informazione richiesta. "
            "Non effettuare una nuova selezione del candidato e "
            "non inventare informazioni mancanti."
        )

    else:
        await cl.Message(
            content="Non ho capito la richiesta. Puoi riformularla?"
        ).send()
        return

    messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    response_message = cl.Message(content="")
    await response_message.send()

    try:
        stream = chat(
            messages=messages,
            stream=True
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content

            if token:
                await response_message.stream_token(token)

        messages.append(
            {
                "role": "assistant",
                "content": response_message.content
            }
        )

        await response_message.update()

        if save_candidate_context:
            cl.user_session.set(
                "last_cv_context",
                context
            )

            cl.user_session.set(
                "last_cv_header",
                candidate_info
            )

    except Exception as error:
        error_message = (
            f"Errore durante la generazione della risposta: {str(error)}"
        )

        await cl.Message(
            content=error_message
        ).send()

    cl.user_session.set(
        "messages",
        messages
    )