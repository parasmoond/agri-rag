import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.rag_pipeline import RAGPipeline


def main():

    rag = RAGPipeline(
        model_name="qwen3:1.7b"
    )

    question = input(
        "Enter question: "
    )

    results = rag.retrieve(
        question,
        top_k=10,
        similarity_threshold=0.0
    )

    print("\n")
    print("=" * 80)
    print("RETRIEVAL RESULTS")
    print("=" * 80)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRANK {rank}"
        )

        print(
            f"Similarity: "
            f"{result['score']:.4f}"
        )

        print(
            f"Source: "
            f"{result['source']}"
        )

        print(
            f"Page: "
            f"{result['page']}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print("\nText:")

        print(
            result["text"]
        )

        print(
            "\n" + "-" * 80
        )


if __name__ == "__main__":
    main()