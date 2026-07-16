from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)


class QdrantStore:

    def __init__(self):

        self.client = QdrantClient(
            host="localhost",
            port=6333
        )

        self.collection_name = "documents"

    def create_collection(self):

        self.client.recreate_collection(
            collection_name=self.collection_name,

            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )

    def insert_chunks(self, chunks, embeddings):

        points = []

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):

            point = PointStruct(
                id=index,

                vector=embedding,

                payload={
                    "text": chunk
                }
            )

            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,

            points=points
        )    

    def search(self, query_embedding, limit=3):

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            with_payload=True
        )

        return results.points
    
    