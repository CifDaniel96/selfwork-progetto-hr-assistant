import re

import numpy as np
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from .config import EMBEDDING_MODEL, OPENAI_API_KEY


class SemanticChunking:

    @staticmethod
    def calculate_cosine_distances(sentences):
        distances = []

        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]["combined_sentence_embedding"]
            embedding_next = sentences[i + 1]["combined_sentence_embedding"]

            similarity = cosine_similarity(
                [embedding_current],
                [embedding_next]
            )[0][0]

            distance = 1 - similarity

            distances.append(distance)
            sentences[i]["distance_to_next"] = distance

        return distances, sentences

    @staticmethod
    def combine_sentences(sentences, buffer_size=1):
        for i in range(len(sentences)):
            combined_sentence = ""

            for j in range(i - buffer_size, i):
                if j >= 0:
                    combined_sentence += sentences[j]["sentence"] + " "

            combined_sentence += sentences[i]["sentence"]

            for j in range(i + 1, i + 1 + buffer_size):
                if j < len(sentences):
                    combined_sentence += " " + sentences[j]["sentence"]

            sentences[i]["combined_sentence"] = combined_sentence

        return sentences

    @staticmethod
    def chunk_it(txt):
        single_sentences_list = re.split(
            r"(?<=[.?!])\s+",
            txt
        )

        sentences = [
            {
                "sentence": sentence,
                "index": index
            }
            for index, sentence in enumerate(single_sentences_list)
        ]

        sentences = SemanticChunking.combine_sentences(sentences)

        embeddings_model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )

        embeddings = embeddings_model.embed_documents(
            [
                sentence["combined_sentence"]
                for sentence in sentences
            ]
        )

        for index, sentence in enumerate(sentences):
            sentence["combined_sentence_embedding"] = embeddings[index]

        distances, sentences = (
            SemanticChunking.calculate_cosine_distances(sentences)
        )

        breakpoint_percentile_threshold = 95

        breakpoint_distance_threshold = np.percentile(
            distances,
            breakpoint_percentile_threshold
        )

        indices_above_threshold = [
            index
            for index, distance in enumerate(distances)
            if distance > breakpoint_distance_threshold
        ]

        start_index = 0
        chunks = []

        for index in indices_above_threshold:
            end_index = index

            group = sentences[start_index:end_index + 1]

            combined_text = " ".join(
                sentence["sentence"]
                for sentence in group
            )

            chunks.append(combined_text)

            start_index = index + 1

        if start_index < len(sentences):
            combined_text = " ".join(
                sentence["sentence"]
                for sentence in sentences[start_index:]
            )

            chunks.append(combined_text)

        return chunks