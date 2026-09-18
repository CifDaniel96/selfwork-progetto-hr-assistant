import os

import ollama

from .config import DOCUMENTS_DIR, OLLAMA_MODEL


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

def classify_intent(user_question):
    prompt = f"""
Sei un classificatore di intenti per un assistente HR.

Devi restituire ESCLUSIVAMENTE una delle seguenti etichette:

search_cv
info_cv

Regole:

- search_cv:
  l'utente sta cercando un nuovo candidato o un profilo con determinate
  competenze, esperienza o caratteristiche.

- info_cv:
  l'utente sta chiedendo informazioni su un candidato già trovato,
  ad esempio email, telefono, nome, esperienza, certificazioni,
  competenze o altri dettagli del suo CV.

Esempi:

"Mi serve un esperto di cybersecurity"
→ search_cv

"Cerco uno sviluppatore Laravel"
→ search_cv

"Chi ha esperienza con Python?"
→ search_cv

"Qual è la sua email?"
→ info_cv

"Che numero di telefono ha?"
→ info_cv

"Quali certificazioni possiede?"
→ info_cv

"Dimmi di più su questo candidato"
→ info_cv

Domanda da classificare:
"{user_question}"

Rispondi solamente con:
search_cv
oppure
info_cv
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    result = response["message"]["content"].strip().lower()

    if "search_cv" in result:
        return "search_cv"

    if "info_cv" in result:
        return "info_cv"

    raise ValueError(f"Intent non riconosciuto: {result}")

