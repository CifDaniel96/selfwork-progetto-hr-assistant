import os
from .config import DOCUMENTS_DIR


def leggi_prime_100_righe(filename):
    file_path = os.path.join(DOCUMENTS_DIR, filename)

    with open(file_path, "r", encoding="utf-8") as file:
        righe = []

        for index, riga in enumerate(file):
            if index < 100:
                righe.append(riga.strip())
            else:
                break

    return righe


def build_context(filename, retrieved_chunk, candidate_name):
    return (
        f"Nome file: {filename}\n"
        f"Nome candidato individuato: {candidate_name}\n"
        f"Chunk più rilevante: {retrieved_chunk}"
    )


def build_prompt(user_question, context):
    return (
        f"Domanda utente: {user_question}\n\n"
        f"Contesto recuperato dal RAG:\n{context}\n\n"
        "Spiega che nel file individuato c'è il profilo più adatto. "
        "Indica il nome del file e il nome del candidato. "
        "Argomenta la scelta usando solo il contenuto del contesto. "
        "Se non trovi corrispondenza, non inventare informazioni."
    )