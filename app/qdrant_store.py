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
