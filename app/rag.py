from app.embeddings import EmbeddingService
from app.qdrant_store import QdrantStore
from app.prompt import build_prompt
from app.llm import LLMService
from app.reranker import Reranker
from app.query_rewriter import QueryRewriter
from app.bm25_search import BM25Search
from app.logging_config import get_logger

logger = get_logger(__name__)

class RAGPipeline:

    def __init__(self, api_key):
        logger.info("Initializing RAG pipeline")
        self.query_rewriter = QueryRewriter(api_key=api_key)


        self.embedding_service = (
            EmbeddingService()
        )
        self.reranker = Reranker()

        self.qdrant_store = QdrantStore()

        self.llm = LLMService(api_key=api_key)
        chunks = self.qdrant_store.get_all_chunks()
        self.bm25 = BM25Search(chunks)
        logger.info("RAG pipeline initialized with %d indexed chunks", len(chunks))

    def refresh_bm25(self):
        """Refresh keyword retrieval after the Qdrant collection changes."""
        chunks = self.qdrant_store.get_all_chunks()
        self.bm25 = BM25Search(chunks)
        logger.info("BM25 index refreshed with %d chunks", len(chunks))




    def ask(
        self,
        question: str,
        history: list = [],
        temperature: float = 0.0,
        rerank_top_k: int = 3,
        hybrid_retrieval: bool = True,
    ):
        logger.info("Processing question with %d history messages", len(history))
        search_query = self.query_rewriter.rewrite(
             question=question,
             history=history,
             temperature=temperature,
        )
        logger.debug("Query rewrite completed")

        # 1. Convert question to embedding
        query_embedding = (
            self.embedding_service.embed(
                search_query
            )
        )

        # 2. Search Qdrant
        results = self.qdrant_store.search(
            query_embedding=query_embedding,
            limit=20
        )

        keyword_results = []
        if hybrid_retrieval:
            keyword_results = self.bm25.search(
                query=search_query,
                top_k=20,
            )

        # 3. Filter low-relevance results
        # relevant_chunks = []
        candidate_chunks = []

        for result in results:
            
            candidate_chunks.append({
                    # result.payload["text"]
                    "text":result.payload["text"],
                    "source":result.payload.get("source"),
                    "page":result.payload.get("page"),
                    "chunk_index":result.payload.get("chunk_index"),
                    "vector_score":result.score


                })
            

        # Merge both retrieval paths by chunk identity when hybrid retrieval
        # is enabled. In vector-only mode, preserve Qdrant's ranking directly
        # so the frontend can compare it with the reranked hybrid result.
        unique_results = {}
        for chunk in candidate_chunks + keyword_results:
            key = (
                chunk["source"],
                chunk["page"],
                chunk["chunk_index"]
            )

            if key not in unique_results:
                unique_results[key] = dict(chunk)
            else:
                unique_results[key].update({
                    name: value
                    for name, value in chunk.items()
                    if name.endswith("_score") and value is not None
                })

        combined_results = list(unique_results.values())
        logger.info("Combined %d unique candidate chunks", len(combined_results))

        if hybrid_retrieval:
            relevant_chunks = self.reranker.rerank(
                question=question,
                chunks=combined_results,
                top_k=rerank_top_k,
            )
            retrieval_mode = "hybrid_reranked"
        else:
            relevant_chunks = candidate_chunks[:rerank_top_k]
            retrieval_mode = "qdrant_only"
        logger.info("Selected %d chunks using %s retrieval", len(relevant_chunks), retrieval_mode)
        # print("vectors :",candidate_chunks)
        # print("keyword search :",keyword_results)


        # 4. Build context
        context = "\n\n".join(
            chunk["text"]
            for chunk in relevant_chunks
        )

        # 5. Build prompt
        prompt = build_prompt(
            question=question,
            context=context
        )

        # 6. Call LLM
        answer = self.llm.generate(
            prompt,
            temperature=temperature,
        )

        response = {
            "answer":answer,
            "question asked":question,
            "rewritten_query":search_query,
            "retrieval_mode": retrieval_mode,
            "context":[chunk["text"] for chunk in relevant_chunks],
            "sources":[
                {
                    "source":chunk["source"],
                    "page":chunk["page"],
                    "vector_score": chunk.get("vector_score"),
                    "keyword_score": chunk.get("keyword_score"),
                    "rerank_score": chunk.get("rerank_score"),
                    "data":chunk["text"]
                }
                for chunk in relevant_chunks
            ]
        }
        logger.info("RAG response generated with %d source chunks", len(relevant_chunks))
        return response 
