import base64

import torch
import redis
import numpy as np
from transformers import BertTokenizer, BertModel

r = redis.Redis(host="localhost", port=6379)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


# Function for vectorizing single text query
def vectorize_text(query, expected_dim=768):
    # Text tokenization
    inputs = tokenizer(
        query, return_tensors="pt", truncation=True, padding=True, max_length=512
    )

    # Get input data model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get embeddings (using hidden states of the last layer)
    query_vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    if query_vector.shape[0] != expected_dim:
        raise ValueError(
            f"Vector dimension mismatch: expected {expected_dim}, got {query_vector.shape[0]}"
        )

    return query_vector


def search(query_vector, top_k=100):
    query_vector = np.array(query_vector, dtype=np.float32).tobytes()

    # For Euclidean distance, Inner product of two vectors, and Cosine distance where the smaller the value is,
    # the closer the two vectors are in the vector space.
    # I use Cosine Similarity
    results = r.execute_command(
        "FT.SEARCH",
        "idx:embeddings",
        "(*)=>[KNN {} @embedding $query_vec]".format(top_k),
        "LIMIT",
        "0",
        str(top_k),
        "SORTBY",
        "__embedding_score",
        "PARAMS",
        "2",
        "query_vec",
        query_vector,
        "DIALECT",
        "2",
    )

    # print("Search results:", results)
    return results


def get_document(doc_id):
    document = r.hgetall(doc_id)

    decoded_document = {
        key.decode("utf-8"): value.decode("utf-8") if key != b"embedding" else value
        for key, value in document.items()
    }

    return decoded_document


if __name__ == "__main__":
    # As more articles I use, as more irrelevant search it makes

    # if we use "machine learning field of study" string response is relevant
    # but if we use "What is Machine learning?" it returns not the definition
    # if I use "neural network field of study" I anyway finds a part of Machine_learning.txt

    # in examples below if the question starts with "What is ..bla.bla.bla" it always returns
    # "Dick considers the idea that our understanding of human subjectivity is altered by technology created with artificial intelligence."

    # query = "What is neural network?"
    # query = "What is data mining?"
    # query = "What is supervised learning?"
    query = "What is machine learning?"
    # query = "When the term machine learning was coined?"
    # query = "Machine learning definition?"
    query_vector = vectorize_text(query)
    results = search(query_vector)
    place = 1
    for res in results:
        if isinstance(res, bytes):
            info_place_and_id = f"{place}, {res}"
            place += 1
        if isinstance(res, list):
            embedding_score = f"embedding_score - {res[1]}"
            print(info_place_and_id, embedding_score)

    document_data = get_document(results[1])
    print(document_data["text"])

    r.connection_pool.disconnect()
