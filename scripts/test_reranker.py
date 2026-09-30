import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.rag_pipeline import RAGPipeline
from src.reranker import Reranker


def main():

    question = input(
        "Enter question: "
    )

    # --------------------------------
    # Load RAG pipeline
    # --------------------------------

    rag = RAGPipeline(
        model_name="qwen3:1.7b"
    )

    # --------------------------------
    # Retrieve candidates
    # --------------------------------

    documents = rag.retrieve(
        question=question,
        top_k=10,
        similarity_threshold=0.0
    )

    print("\n")
    print("=" * 80)
    print("FAISS RESULTS")
    print("=" * 80)

    for rank, document in enumerate(
        documents,
        start=1
    ):

        print(
            f"\nRank {rank}"
        )

        print(
            f"FAISS score: "
            f"{document['score']:.4f}"
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

    # --------------------------------
    # Reranking
    # --------------------------------

    reranker = Reranker()

    reranked = reranker.rerank(
        query=question,
        documents=documents,
        top_k=5
    )

    print("\n")
    print("=" * 80)
    print("RERANKED RESULTS")
    print("=" * 80)

    for rank, document in enumerate(
        reranked,
        start=1
    ):

        print(
            f"\nRank {rank}"
        )

        print(
            f"Rerank score: "
            f"{document['rerank_score']:.4f}"
        )

        print(
            f"Original FAISS score: "
            f"{document['score']:.4f}"
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
            "\n" + "-" * 80
        )


if __name__ == "__main__":
    main()