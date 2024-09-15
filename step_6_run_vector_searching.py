import torch
import redis
import numpy as np
from transformers import BertTokenizer, BertModel

r = redis.Redis(host="localhost", port=6379)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


# Function for vectorizing single text query
def vectorize_text(query):
    # Text tokenization
    inputs = tokenizer(
        query, return_tensors="pt", truncation=True, padding=True, max_length=512
    )

    # Get input data model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get embeddings
    query_vector = outputs.last_hidden_state.mean(dim=1)

    return query_vector.squeeze().numpy()


def search(query_vector, top_k=5):
    query_vector = np.array(query_vector, dtype=np.float32).tobytes()

    # Выполнение поиска
    results = r.execute_command(
        "FT.SEARCH",
        "idx:embeddings",
        "*",
        "NOCONTENT",
        "SORTBY",
        "embedding",
        "LIMIT",
        "0",
        str(top_k),
        "PARAMS",
        "2",
        "query_vec",
        query_vector,
    )

    print("Search results:", results)
    return results


def get_document(doc_id):
    document = r.hgetall(doc_id)

    decoded_document = {
        key.decode("utf-8"): value.decode("utf-8") if key != b"embedding" else value
        for key, value in document.items()
    }

    return decoded_document


if __name__ == "__main__":
    # if we use "machine learning field of study" string response is relevant
    # but if we use "What is Machine learning?" it returns not the definition
    # if i use "neural network field of study" I anyway finds a part of Machine_learning.txt
    query = "What is Machine learning?"
    query_vector = vectorize_text(query)
    results = search(query_vector)
    print(results)

    document_data = get_document(results[1])
    print(document_data["text"])

    r.connection_pool.disconnect()
