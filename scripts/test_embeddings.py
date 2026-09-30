import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.embeddings import EmbeddingModel


def main():

    model = EmbeddingModel()

    sentences = [
        "Rice blast is a fungal disease.",
        "Rice blast causes lesions on leaves.",
        "Tomatoes require regular irrigation."
    ]

    embeddings = model.encode(sentences)

    print("\nEmbedding shape:")
    print(embeddings.shape)

    print("\nFirst embedding:")
    print(embeddings[0])

    print("\nVector norm:")
    print((embeddings[0] ** 2).sum() ** 0.5)


if __name__ == "__main__":
    main()