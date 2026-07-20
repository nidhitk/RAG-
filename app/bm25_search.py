from rank_bm25 import BM25Okapi


class BM25Search:

    def __init__(self, chunks: list):

        self.chunks = chunks

        tokenized_chunks = [
            chunk["text"].lower().split()
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            tokenized_chunks
        )

    def search(
        self,
        query: str,
        top_k: int = 20
    ):

        tokenized_query = (
            query.lower().split()
        )

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        results = []

        for index in ranked_indexes[:top_k]:

            results.append({
                "text": self.chunks[index]["text"],
                "source": self.chunks[index].get("source"),
                "page": self.chunks[index].get("page"),
                "chunk_index": self.chunks[index].get(
                    "chunk_index"
                ),
                "keyword_score": float(
                    scores[index]
                )
            })

        return results