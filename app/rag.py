from app.embeddings import EmbeddingService
from app.qdrant_store import QdrantStore
from app.prompt import build_prompt
from app.llm import LLMService
from app.reranker import Reranker
from app.query_rewriter import QueryRewriter
from app.bm25_search import BM25Search
import json

class RAGPipeline:

    def __init__(self, api_key):
        self.query_rewriter = QueryRewriter(
            api_key=api_key
        )

        self.embedding_service = (
            EmbeddingService()
        )
        self.reranker = Reranker()

        self.qdrant_store = QdrantStore()

        self.llm = LLMService(
            api_key=api_key
        )
        with open(
            "data/chunks.json",
            "r",
            encoding="utf-8"
        ) as file:

            chunks = json.load(file)

        self.bm25 = BM25Search(chunks)




    def ask(self, question: str,history: list = []):

        search_query = self.query_rewriter.rewrite(
             question=question,
             history=history
        )
        print("query",search_query)

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

        keyword_results = self.bm25.search(
        query=search_query,
        top_k=20
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
            

        combined_results = (
        candidate_chunks
        + keyword_results
        )    


        unique_results = {}
        for chunk in combined_results:

            key = (
                chunk["source"],
                chunk["page"],
                chunk["chunk_index"]
            )

            unique_results[key] = chunk

        combined_results = list(
            unique_results.values()
        )

            
        relevant_chunks = self.reranker.rerank(
        question=question,
        chunks=combined_results,
        top_k=3
        )

        print("vectors :",candidate_chunks)
        print("keyword search :",keyword_results)


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
            prompt
        )

        return {
            "answer":answer,
            "question asked":question,
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