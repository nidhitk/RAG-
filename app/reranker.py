from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(
        self,
        question: str,
        chunks: list,
        top_k: int = 3
    ):

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

        return scored_chunks[:top_k]