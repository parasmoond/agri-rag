import pickle
from pathlib import Path

import faiss

from src.embeddings import EmbeddingModel
from src.llm import GeminiLLM
from src.prompt import build_rag_prompt
from src.reranker import Reranker


INDEX_PATH = Path(
    "vectorstore/index.faiss"
)

METADATA_PATH = Path(
    "vectorstore/metadata.pkl"
)


class RAGPipeline:

    def __init__(
        self,
        model_name="qwen3:1.7b"
    ):

        # --------------------------------
        # Load FAISS index
        # --------------------------------

        self.index = faiss.read_index(
            str(INDEX_PATH)
        )

        # --------------------------------
        # Load metadata
        # --------------------------------

        with open(
            METADATA_PATH,
            "rb"
        ) as f:

            self.documents = pickle.load(f)

        # --------------------------------
        # Load embedding model
        # --------------------------------

        self.embedding_model = (
            EmbeddingModel()
        )

        # --------------------------------
        # Load LLM
        # --------------------------------

        self.llm = GeminiLLM()

        # --------------------------------
        # Load reranker
        # --------------------------------

        self.reranker = Reranker()

    # ====================================
    # RETRIEVAL
    # ====================================

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        similarity_threshold: float = 0.35
    ):

        # Convert question into embedding
        query_embedding = (
            self.embedding_model.encode(
                [question]
            )
        )

        # Search FAISS
        scores, indices = self.index.search(
            query_embedding.astype(
                "float32"
            ),
            top_k
        )

        results = []

        # Process results
        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            # Ignore weak matches
            if score < similarity_threshold:
                continue

            document = self.documents[index]

            results.append({
                "score": float(score),
                "source": document["source"],
                "page": document["page"],
                "chunk_id": document["chunk_id"],
                "text": document["text"]
            })

        return results

    # ====================================
    # ANSWER GENERATION
    # ====================================

    def answer(
        self,
        question: str,
        retrieval_k: int = 10,
        final_k: int = 3,
        similarity_threshold: float = 0.35
    ):

        # --------------------------------
        # Stage 1:
        # FAISS candidate retrieval
        # --------------------------------

        retrieved_documents = self.retrieve(
            question=question,
            top_k=retrieval_k,
            similarity_threshold=similarity_threshold
        )

        # --------------------------------
        # Stage 2:
        # Cross-encoder reranking
        # --------------------------------

        reranked_documents = (
            self.reranker.rerank(
                query=question,
                documents=retrieved_documents,
                top_k=final_k
            )
        )

        # --------------------------------
        # Stage 3:
        # Hard abstention guard
        # --------------------------------

        if not reranked_documents:

            answer = (
                "I don't have enough information "
                "in the available agricultural "
                "documents to answer this question."
            )

            return {
                "question": question,
                "answer": answer,
                "sources": []
            }

        # --------------------------------
        # Stage 4:
        # Build RAG prompt
        # --------------------------------

        prompt = build_rag_prompt(
            question,
            reranked_documents
        )

        # --------------------------------
        # Stage 5:
        # Generate answer using LLM
        # --------------------------------

        answer = self.llm.generate(
            prompt
        )

        # --------------------------------
        # Return complete result
        # --------------------------------

        return {
            "question": question,
            "answer": answer,
            "sources": reranked_documents
        }