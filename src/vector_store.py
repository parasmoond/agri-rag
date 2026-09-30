import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension: int):

        self.dimension = dimension

        # Inner Product index
        self.index = faiss.IndexFlatIP(dimension)

        # Store metadata/chunks separately
        self.documents = []

    def add(
        self,
        embeddings: np.ndarray,
        documents: list
    ):

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        self.index.add(embeddings.astype("float32"))

        self.documents.extend(documents)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ):

        query_embedding = query_embedding.astype("float32")

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            results.append({
                "score": float(score),
                "document": self.documents[index]
            })

        return results