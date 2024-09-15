import wikipedia

if __name__ == "__main__":
    articles = [
        "Natural_language_processing",
        "Machine_learning",
        "Artificial_intelligence",
        "Deep_learning",
        "Supervised_learning",
        "Unsupervised_learning",
        "Reinforcement_learning",
        "Neural_networks",
        "Feature_engineering",
        "Data_mining",
    ]

    for article in articles:
        content = wikipedia.page(article, auto_suggest=False).content
        with open(f"data/{article}.txt", "w", encoding="utf-8") as file:
            file.write(content)
