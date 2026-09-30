import sys
import json
import pickle
from pathlib import Path

import faiss


sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.embeddings import EmbeddingModel


PROCESSED_DIR = Path("data/processed")
VECTORSTORE_DIR = Path("vectorstore")


def main():

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    chunks_path = (
        PROCESSED_DIR / "chunks.json"
    )

    if not chunks_path.exists():

        print(
            "chunks.json not found."
        )

        print(
            "Run: python scripts/ingest.py"
        )

        return

    # --------------------------------
    # Load chunks
    # --------------------------------

    with open(
        chunks_path,
        "r",
        encoding="utf-8"
    ) as f:

        documents = json.load(f)

    print(
        f"Loaded {len(documents)} chunks."
    )

    # --------------------------------
    # Extract text
    # --------------------------------

    texts = [
        document["text"]
        for document in documents
    ]

    # --------------------------------
    # Create embeddings
    # --------------------------------

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.encode(
        texts
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # --------------------------------
    # Create FAISS index
    # --------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings.astype("float32")
    )

    print(
        f"FAISS index contains "
        f"{index.ntotal} vectors."
    )

    # --------------------------------
    # Save index
    # --------------------------------

    index_path = (
        VECTORSTORE_DIR / "index.faiss"
    )

    faiss.write_index(
        index,
        str(index_path)
    )

    # --------------------------------
    # Save metadata
    # --------------------------------

    metadata_path = (
        VECTORSTORE_DIR /
        "metadata.pkl"
    )

    with open(
        metadata_path,
        "wb"
    ) as f:

        pickle.dump(
            documents,
            f
        )

    print("\nIndex successfully built.")

    print(
        f"FAISS index: {index_path}"
    )

    print(
        f"Metadata: {metadata_path}"
    )


if __name__ == "__main__":
    main()