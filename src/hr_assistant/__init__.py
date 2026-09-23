import os
import shutil

import chainlit as cl

from hr_assistant.config import DOCUMENTS_DIR
from hr_assistant.database import Database
from hr_assistant.document_processor import (
    SUPPORTED_EXTENSIONS,
    process_single_document,
    sync_documents,
)
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
        author="system_assistant",
        content=response,
    ).send()


@cl.action_callback("db_reindex")
async def reindex_database(action: cl.Action):
    added, updated, removed = sync_documents(database)

    await cl.Message(
        author="system_assistant",
        content=(
            "Database reindicizzato con successo.\n\n"
            f"Added: {added}\n"
            f"Updated: {updated}\n"
            f"Removed: {removed}"
        ),
    ).send()


@cl.action_callback("db_clear")
async def clear_database(action: cl.Action):
    removed_fragments = database.clear_database()

    cl.user_session.set("last_cv_context", "")
    cl.user_session.set("last_cv_header", "")

    await cl.Message(
        author="system_assistant",
        content=(
            "Database azzerato con successo.\n\n"
            f"Frammenti rimossi: {removed_fragments}"
        ),
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
        cl.Action(
            name="db_clear",
            icon="mouse-pointer-click",
            payload={"value": "db_clear"},
            label="Azzera Database",
        ),
    ]

    await cl.Message(
        author="system_assistant",
        content="Informazioni del sistema:",
        actions=actions,
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


async def _process_and_index_file(file_path, file_name):
    documents, metadatas, ids = process_single_document(file_path)

    if not documents:
        return f"Errore nel processare il file '{file_name}'."

    database.remove_document_by_source(file_name)

    database.add_documents(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    return (
        f"File '{file_name}' caricato "
        "e indicizzato con successo."
    )


async def _file_upload(file):
    file_name = os.path.basename(file.name)

    extension = os.path.splitext(file_name)[1].lower()

    if extension not in SUPPORTED_EXTENSIONS:
        return (
            f"Formato non supportato per "
            f"'{file_name}'."
        )

    os.makedirs(
        DOCUMENTS_DIR,
        exist_ok=True,
    )

    destination = os.path.join(
        DOCUMENTS_DIR,
        file_name,
    )

    shutil.copy2(
        file.path,
        destination,
    )

    return await _process_and_index_file(
        destination,
        file_name,
    )


@cl.on_message
async def handle_message(message: cl.Message):
    if message.elements:
        upload_results = []

        for element in message.elements:
            if (
                not getattr(element, "name", None)
                or not getattr(element, "path", None)
            ):
                continue

            result = await _file_upload(element)
            upload_results.append(result)

        if upload_results:
            await cl.Message(
                author="system_assistant",
                content="\n".join(upload_results),
            ).send()

            return

    user_question = message.content

    try:
        intent = classify_intent(user_question)

    except ValueError as error:
        await cl.Message(
            author="system_assistant",
            content=(
                "Non riesco a classificare la richiesta: "
                f"{str(error)}"
            ),
        ).send()

        return

    messages = cl.user_session.get("messages", [])

    context = ""
    candidate_info = ""
    save_candidate_context = False

    if intent == "search_cv":
        results = database.query(
            user_question,
            n_results=3,
        )

        if (
            not results
            or not results.get("documents")
            or not results["documents"][0]
        ):
            await cl.Message(
                author="system_assistant",
                content=(
                    "Nessun curriculum trovato per la tua richiesta. "
                    "Prova a specificare meglio competenze o esperienza."
                ),
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
            context=context,
        )

        save_candidate_context = True

    elif intent == "info_cv":
        context = cl.user_session.get(
            "last_cv_context",
            "",
        )

        candidate_info = cl.user_session.get(
            "last_cv_header",
            "",
        )

        if not context:
            await cl.Message(
                author="system_assistant",
                content=(
                    "Non c'è ancora un candidato selezionato. "
                    "Cerca prima un profilo e poi chiedimi informazioni "
                    "specifiche su quel candidato."
                ),
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
            author="system_assistant",
            content="Non ho capito la richiesta. Puoi riformularla?",
        ).send()

        return

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    response_message = cl.Message(
        author="hr_assistant",
        content="",
    )

    await response_message.send()

    try:
        stream = chat(
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content

            if token:
                await response_message.stream_token(token)

        messages.append(
            {
                "role": "assistant",
                "content": response_message.content,
            }
        )

        await response_message.update()

        if save_candidate_context:
            cl.user_session.set(
                "last_cv_context",
                context,
            )

            cl.user_session.set(
                "last_cv_header",
                candidate_info,
            )

    except Exception as error:
        error_message = (
            "Errore durante la generazione della risposta: "
            f"{str(error)}"
        )

        await cl.Message(
            author="system_assistant",
            content=error_message,
        ).send()

    cl.user_session.set(
        "messages",
        messages,
    )