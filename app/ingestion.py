from pathlib import Path

from app.chunking import chunk_text
from app.embeddings import EmbeddingService
from app.qdrant_store import QdrantStore
from app.pdf_loader import load_pdf


def ingest_document(file_path: str):


    pages = load_pdf(file_path)

    all_chunks=[]
    all_metadata=[]

    for page in pages:

        chunks = chunk_text(
            text=page["text"],
            chunk_size=500,
            overlap=50
        )
        for chunk_index, chunk in enumerate(chunks):

            all_chunks.append(chunk)
            all_metadata.append(
                {
                    "source":Path(file_path).name,
                    "page":page["page"],
                    "chunk_index":chunk_index
                }
            )

    embedding_service = EmbeddingService()

    embeddings = []

    for chunk in all_chunks:
        embedding = embedding_service.embed(chunk)
        embeddings.append(embedding)

    qdrant_store = QdrantStore()

    qdrant_store.create_collection()

    qdrant_store.insert_chunks(
        chunks=all_chunks,
        embeddings=embeddings,
        metadata=all_metadata
    )

    print(f"Ingested {len(all_chunks)} chunks")


def main():

    ingest_document(
        "documents/sample.pdf"
    )


if __name__ == "__main__":
    main()