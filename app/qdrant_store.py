from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)
from app.logging_config import get_logger

logger = get_logger(__name__)


class QdrantStore:

    def __init__(self):
        logger.info("Connecting to Qdrant at localhost:6333")
        self.client = QdrantClient(
            host="localhost",
            port=6333
        )

        self.collection_name = "documents"

    def create_collection(self):
        logger.info("Recreating Qdrant collection '%s'", self.collection_name)
        self.client.recreate_collection(
            collection_name=self.collection_name,

            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )

    def insert_chunks(self, chunks, embeddings,metadata):
        logger.info("Preparing %d chunks for Qdrant insertion", len(chunks))
        points = []

        for index, (chunk, embedding,chunk_metadata) in enumerate(
            zip(chunks, embeddings,metadata)
        ):

            point = PointStruct(
                id=index,

                vector=embedding,

                payload={
                    "text": chunk,
                    **chunk_metadata
                }
            )

            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,

            points=points
        )
        logger.info("Inserted %d chunks into Qdrant", len(points))

    def search(self, query_embedding, limit=3):
        logger.debug("Searching Qdrant collection '%s' with limit=%d", self.collection_name, limit)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            with_payload=True
        )

        logger.info("Qdrant search returned %d results", len(results.points))
        return results.points

    def get_all_chunks(self, batch_size=256):
        """Read the complete BM25 corpus from Qdrant payloads.

        Qdrant is the source of truth for indexed chunks. This keeps keyword
        retrieval in sync with the vector index and avoids a second local
        chunk store.
        """
        chunks = []
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}
                text = payload.get("text")
                if not text:
                    continue

                chunks.append({
                    "text": text,
                    "source": payload.get("source"),
                    "page": payload.get("page"),
                    "chunk_index": payload.get("chunk_index"),
                })

            if offset is None:
                break

        logger.info("Loaded %d chunks from Qdrant for BM25", len(chunks))
        return chunks
