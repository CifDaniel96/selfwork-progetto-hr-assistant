import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .custom_embedding import CustomEmbeddingFunction

class SemanticChunking:
    def __init__(self, breakpoint_percentile=95, buffer_size=1):
        self.embeddings = CustomEmbeddingFunction()
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_size = buffer_size

    def _process_sentences(self, text):
        sentences = [
            {
                "sentence": sentence,
                "index": index
            }
            for index, sentence in enumerate(
                re.split(r"(?<=[.?!])\s+", text)
            )
        ]

        for index, current in enumerate(sentences):
            context_range = range(
                max(0, index - self.buffer_size),
                min(
                    len(sentences),
                    index + self.buffer_size + 1
                )
            )

            current["combined_sentence"] = " ".join(
                sentences[position]["sentence"]
                for position in context_range
            )

        return sentences

    def _calculate_distances(self, sentences):
        embeddings = self.embeddings(
            [
                sentence["combined_sentence"]
                for sentence in sentences
            ]
        )

        distances = []

        for index in range(len(sentences) - 1):
            distance = 1 - cosine_similarity(
                [embeddings[index]],
                [embeddings[index + 1]]
            )[0][0]

            distances.append(distance)

        return distances

    def chunk_text(self, text):
        sentences = self._process_sentences(text)

        print("SENTENCES:", sentences[:2])

        distances = self._calculate_distances(sentences)

        print("DISTANCES:", distances[:2])

        threshold = np.percentile(
            distances,
            self.breakpoint_percentile
        )

        split_points = [
            index
            for index, distance in enumerate(distances)
            if distance > threshold
        ]

        print("SPLIT POINTS:", split_points)

        chunks = []
        start = 0

        for point in split_points + [len(sentences) - 1]:
            chunk = " ".join(
                sentence["sentence"]
                for sentence in sentences[start:point + 1]
            )

            print("CHUNK:", chunk)

            chunks.append(chunk)
            start = point + 1

        return chunks