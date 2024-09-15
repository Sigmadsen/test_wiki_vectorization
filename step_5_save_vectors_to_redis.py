import redis
import numpy as np

from const import ARTICLES_TXT
from step_2_split_text_to_chunks import chunk_text
from step_3_article_vectorization import vectorize_chunks

r = redis.Redis(host="localhost", port=6379)


def save_embedding(doc_id, text, embedding):
    r.hset(
        doc_id,
        mapping={
            "text": text,
            "embedding": np.array(
                embedding, dtype=np.float32
            ).tobytes(),  # Save vector as bytes
        },
    )


if __name__ == "__main__":
    i = 0
    for file in ARTICLES_TXT:
        with open(file, "r", encoding="utf-8") as file:
            text = file.read()

        chunks = chunk_text(text)
        vectors = vectorize_chunks(chunks)

        # Save the data
        for chunk, vector in zip(chunks, vectors):
            save_embedding(f"doc:{i}", chunk, vector)
            i += 1

    r.connection_pool.disconnect()
