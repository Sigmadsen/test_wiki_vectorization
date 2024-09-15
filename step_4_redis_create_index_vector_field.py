import redis

# Connect to Redis
r = redis.Redis(host="localhost", port=6379)


def create_index():
    # Create index with vector field
    r.execute_command(
        "FT.CREATE",
        "idx:embeddings",
        "SCHEMA",
        "text",
        "TEXT",
        "embedding",
        "VECTOR",
        "FLAT",
        "6",
        "TYPE",
        "FLOAT32",
        "DIM",
        "768",
        "DISTANCE_METRIC",
        "COSINE",
    )


if __name__ == "__main__":
    create_index()

    info = r.execute_command("FT.INFO", "idx:embeddings")
    print(info)

    r.connection_pool.disconnect()
