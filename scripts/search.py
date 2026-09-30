import sys
import pickle
from pathlib import Path

import faiss


sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.embeddings import EmbeddingModel


INDEX_PATH = Path(
    "vectorstore/index.faiss"
)

METADATA_PATH = Path(
    "vectorstore/metadata.pkl"
)


def main():

    # -------------------------------
    # Load FAISS index
    # -------------------------------

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    # -------------------------------
    # Load metadata
    # -------------------------------

    with open(
        METADATA_PATH,
        "rb"
    ) as f:

        documents = pickle.load(f)

    # -------------------------------
    # Load embedding model
    # -------------------------------

    model = EmbeddingModel()

    print("\nSemantic Search")
    print("Type 'exit' to quit.\n")

    while True:

        query = input(
            "Ask a question: "
        ).strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        # ---------------------------
        # Embed query
        # ---------------------------

        query_embedding = model.encode(
            [query]
        )

        # ---------------------------
        # Search FAISS
        # ---------------------------

        scores, indices = index.search(
            query_embedding.astype(
                "float32"
            ),
            5
        )

        print("\nTop results:\n")

        for rank, (
            score,
            index_id
        ) in enumerate(
            zip(
                scores[0],
                indices[0]
            ),
            start=1
        ):

            document = documents[index_id]

            print(
                f"#{rank} "
                f"(score={score:.4f})"
            )

            print(
                f"Source: "
                f"{document['source']}"
            )

            print(
                f"Page: "
                f"{document['page']}"
            )

            print(
                f"\n{document['text']}"
            )

            print(
                "\n" + "-" * 70
            )


if __name__ == "__main__":
    main()