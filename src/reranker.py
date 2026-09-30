from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):

        print(
            f"Loading reranker: {model_name}"
        )

        self.model = CrossEncoder(
            model_name
        )

    def rerank(
        self,
        query: str,
        documents: list,
        top_k: int = 3
    ):

        if not documents:
            return []

        # Create query-document pairs
        pairs = [
            [query, document["text"]]
            for document in documents
        ]

        # Get relevance scores
        scores = self.model.predict(
            pairs
        )

        # Attach scores
        reranked = []

        for document, score in zip(
            documents,
            scores
        ):

            result = document.copy()

            result["rerank_score"] = float(
                score
            )

            reranked.append(result)

        # Sort by reranker score
        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        # Return best results
        return reranked[:top_k]