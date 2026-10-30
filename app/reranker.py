from sentence_transformers import CrossEncoder
from app.logging_config import get_logger

logger = get_logger(__name__)


class Reranker:

    def __init__(self):
        logger.info("Loading reranker model")
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(
        self,
        question: str,
        chunks: list,
        top_k: int = 3
    ):
        logger.info("Reranking %d chunks; selecting top %d", len(chunks), top_k)
        pairs = []

        for chunk in chunks:

            pairs.append(
                (
                    question,
                    chunk["text"]
                )
            )

        scores = self.model.predict(
            pairs
        )

        scored_chunks = []

        for chunk, score in zip(
            chunks,
            scores
        ):

            chunk["rerank_score"] = float(
                score
            )

            scored_chunks.append(chunk)

        scored_chunks.sort(
            key=lambda chunk: chunk["rerank_score"],
            reverse=True
        )

        results = scored_chunks[:top_k]
        logger.info("Reranker returned %d chunks", len(results))
        return results
