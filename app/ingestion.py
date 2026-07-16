from pathlib import Path

from app.chunking import chunk_text
from app.embeddings import EmbeddingService
from app.qdrant_store import QdrantStore


def ingest_document(file_path: str):

    text = Path(file_path).read_text(
        encoding="utf-8"
    )

    chunks = chunk_text(
        text=text,
        chunk_size=50,
        overlap=10
    )

    embedding_service = EmbeddingService()

    embeddings = []

    for chunk in chunks:
        embedding = embedding_service.embed(chunk)
        embeddings.append(embedding)

    qdrant_store = QdrantStore()

    qdrant_store.create_collection()

    qdrant_store.insert_chunks(
        chunks=chunks,
        embeddings=embeddings
    )

    print(f"Ingested {len(chunks)} chunks")


def main():

    ingest_document(
        "documents/docker.txt"
    )


if __name__ == "__main__":
    main()