import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.rag_pipeline import RAGPipeline


def main():

    print("=" * 60)
    print("       AGRICROP RAG ASSISTANT")
    print("=" * 60)

    print("\nLoading RAG system...\n")

    rag = RAGPipeline(
        model_name="qwen3:1.7b"
    )

    print("RAG system ready.")
    print("Type 'exit' to quit.\n")

    while True:

        question = input(
            "Ask an agricultural question: "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        print("\nSearching agricultural knowledge...\n")

        result = rag.answer(
            question,
            retrieval_k=10,
            final_k=3
        )

        print("=" * 60)
        print("ANSWER")
        print("=" * 60)

        print(
            result["answer"]
        )

        print("\n")
        print("=" * 60)
        print("SOURCES")
        print("=" * 60)

        for i, source in enumerate(
            result["sources"],
            start=1
        ):

            print(
                f"\n[{i}] "
                f"{source['source']} "
                f"(Page {source['page']})"
            )

            print(
                f"Similarity: "
                f"{source['score']:.4f}"
            )

        print("\n")


if __name__ == "__main__":
    main()