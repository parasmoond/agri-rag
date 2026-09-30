import sys
import json
import pickle
from pathlib import Path

import faiss

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.embeddings import EmbeddingModel
from src.reranker import Reranker


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


def find_rank(documents, expected_source):
    for rank, document in enumerate(
        documents,
        start=1
    ):
        if document["source"] == expected_source:
            return rank

    return None


def main():

    questions = load_questions()

    index, documents = load_vector_store()

    embedding_model = EmbeddingModel()

    reranker = Reranker()

    faiss_reciprocal_ranks = []
    reranked_reciprocal_ranks = []

    faiss_recall_at_1 = 0
    reranked_recall_at_1 = 0

    print("=" * 80)
    print("RERANKING EVALUATION")
    print("=" * 80)

    for item in questions:

        question = item["question"]
        expected_source = item["expected_source"]

        # ------------------------------------------------
        # Stage 1: FAISS retrieval
        # ------------------------------------------------

        query_embedding = embedding_model.encode(
            [question]
        )

        scores, indices = index.search(
            query_embedding.astype("float32"),
            10
        )

        faiss_documents = []

        for score, index_id in zip(
            scores[0],
            indices[0]
        ):

            if index_id == -1:
                continue

            document = documents[index_id]

            faiss_documents.append({
                "score": float(score),
                "source": document["source"],
                "page": document["page"],
                "chunk_id": document["chunk_id"],
                "text": document["text"]
            })

        # ------------------------------------------------
        # Find FAISS rank
        # ------------------------------------------------

        faiss_rank = find_rank(
            faiss_documents,
            expected_source
        )

        # ------------------------------------------------
        # Stage 2: Cross-Encoder reranking
        # ------------------------------------------------

        reranked_documents = reranker.rerank(
            query=question,
            documents=faiss_documents,
            top_k=10
        )

        # ------------------------------------------------
        # Find reranked rank
        # ------------------------------------------------

        reranked_rank = find_rank(
            reranked_documents,
            expected_source
        )

        # ------------------------------------------------
        # Metrics
        # ------------------------------------------------

        if faiss_rank is not None:

            faiss_reciprocal_ranks.append(
                1 / faiss_rank
            )

            if faiss_rank == 1:
                faiss_recall_at_1 += 1

        else:

            faiss_reciprocal_ranks.append(0)

        if reranked_rank is not None:

            reranked_reciprocal_ranks.append(
                1 / reranked_rank
            )

            if reranked_rank == 1:
                reranked_recall_at_1 += 1

        else:

            reranked_reciprocal_ranks.append(0)

        # ------------------------------------------------
        # Print result
        # ------------------------------------------------

        print("\nQuestion:")
        print(question)

        print(
            f"FAISS rank:     {faiss_rank}"
        )

        print(
            f"Reranked rank:  {reranked_rank}"
        )

    # ----------------------------------------------------
    # Final metrics
    # ----------------------------------------------------

    total = len(questions)

    faiss_mrr = (
        sum(faiss_reciprocal_ranks)
        / total
    )

    reranked_mrr = (
        sum(reranked_reciprocal_ranks)
        / total
    )

    faiss_recall_at_1 /= total

    reranked_recall_at_1 /= total

    print("\n")
    print("=" * 80)
    print("FINAL RERANKING METRICS")
    print("=" * 80)

    print(
        f"Questions evaluated: {total}"
    )

    print(
        f"\nFAISS Recall@1:    "
        f"{faiss_recall_at_1:.4f}"
    )

    print(
        f"Reranked Recall@1: "
        f"{reranked_recall_at_1:.4f}"
    )

    print(
        f"\nFAISS MRR:         "
        f"{faiss_mrr:.4f}"
    )

    print(
        f"Reranked MRR:      "
        f"{reranked_mrr:.4f}"
    )


if __name__ == "__main__":
    main()