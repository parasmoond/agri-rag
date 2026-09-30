import sys
import json
import pickle
from pathlib import Path

import faiss

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.embeddings import EmbeddingModel


INDEX_PATH = Path("vectorstore/index.faiss")
METADATA_PATH = Path("vectorstore/metadata.pkl")
QUESTIONS_PATH = Path("evaluation/questions.json")


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vector_store():
    index = faiss.read_index(str(INDEX_PATH))

    with open(METADATA_PATH, "rb") as f:
        documents = pickle.load(f)

    return index, documents


def evaluate():
    questions = load_questions()
    index, documents = load_vector_store()

    embedding_model = EmbeddingModel()

    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0
    recall_at_10 = 0

    reciprocal_ranks = []

    print("=" * 80)
    print("RETRIEVAL EVALUATION")
    print("=" * 80)

    for item in questions:

        question = item["question"]
        expected_source = item["expected_source"]

        query_embedding = embedding_model.encode(
            [question]
        )

        scores, indices = index.search(
            query_embedding.astype("float32"),
            10
        )

        retrieved_sources = []

        for index_id in indices[0]:

            if index_id == -1:
                continue

            document = documents[index_id]

            retrieved_sources.append(
                document["source"]
            )

        # Find rank of expected source
        rank = None

        for position, source in enumerate(
            retrieved_sources,
            start=1
        ):
            if source == expected_source:
                rank = position
                break

        print("\nQuestion:")
        print(question)

        print(f"Expected source: {expected_source}")

        if rank is None:

            print("Expected source not found in top 10")

            reciprocal_ranks.append(0)

        else:

            print(f"Found at rank: {rank}")

            reciprocal_ranks.append(
                1 / rank
            )

            if rank <= 1:
                recall_at_1 += 1

            if rank <= 3:
                recall_at_3 += 1

            if rank <= 5:
                recall_at_5 += 1

            if rank <= 10:
                recall_at_10 += 1

    total_questions = len(questions)

    recall_at_1 /= total_questions
    recall_at_3 /= total_questions
    recall_at_5 /= total_questions
    recall_at_10 /= total_questions

    mrr = sum(reciprocal_ranks) / total_questions

    print("\n")
    print("=" * 80)
    print("FINAL RETRIEVAL METRICS")
    print("=" * 80)

    print(f"Questions evaluated: {total_questions}")

    print(f"Recall@1:  {recall_at_1:.4f}")
    print(f"Recall@3:  {recall_at_3:.4f}")
    print(f"Recall@5:  {recall_at_5:.4f}")
    print(f"Recall@10: {recall_at_10:.4f}")
    print(f"MRR:       {mrr:.4f}")


if __name__ == "__main__":
    evaluate()