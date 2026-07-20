# RAG Documentation Assistant

A FastAPI-based Retrieval-Augmented Generation (RAG) service for answering questions from local documents.

The project ingests a PDF from the `documents` directory, splits it into page-aware chunks, stores embeddings in Qdrant, writes chunk metadata to `data/chunks.json`, and answers questions with Groq using retrieved context. Retrieval combines vector search, BM25 keyword search, and cross-encoder reranking.

## Features

- FastAPI API with a `/ask` endpoint
- PDF document ingestion with page metadata
- Sentence Transformer embeddings using `all-MiniLM-L6-v2`
- Qdrant vector storage through Docker Compose
- Hybrid retrieval with Qdrant vector search and BM25 keyword search
- Query rewriting for follow-up questions with conversation history
- Cross-encoder reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Groq chat completion using `llama-3.1-8b-instant`
- Retrieval evaluation script with hit-rate reporting

## Project Structure

```text
.
+-- app/
|   +-- bm25_search.py     # BM25 keyword retrieval over saved chunks
|   +-- chunking.py        # Splits document text into overlapping chunks
|   +-- config.py          # Reserved for configuration
|   +-- embeddings.py      # Generates sentence-transformer embeddings
|   +-- evaluate.py        # Evaluates retrieval against test cases
|   +-- ingestion.py       # Loads a PDF into Qdrant and data/chunks.json
|   +-- llm.py             # Calls the Groq API
|   +-- main.py            # FastAPI application
|   +-- pdf_loader.py      # Extracts text from PDF pages
|   +-- prompt.py          # Prompt template
|   +-- qdrant_store.py    # Qdrant collection, insert, and search logic
|   +-- query_rewriter.py  # Rewrites follow-up questions for retrieval
|   +-- rag.py             # End-to-end RAG pipeline
|   +-- reranker.py        # Cross-encoder reranking
+-- data/
|   +-- chunks.json        # Generated chunk text and metadata
|   +-- evaluation.json    # Retrieval evaluation test cases
+-- documents/
|   +-- docker.txt         # Example text document
|   +-- sample.pdf         # Default PDF ingested by the pipeline
+-- docker-compose.yml     # Qdrant service
+-- requirements.txt       # Python dependencies
+-- .env                   # Environment variables
```

## Requirements

- Python 3.10+
- Docker Desktop
- A Groq API key

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Start Qdrant:

```powershell
docker compose up -d
```

## Ingest Documents

Load the default PDF into Qdrant:

```powershell
python -m app.ingestion
```

By default, this ingests:

```text
documents/sample.pdf
```

Ingestion creates overlapping chunks, embeds them, recreates the Qdrant `documents` collection, and writes the same chunk records to:

```text
data/chunks.json
```

To ingest a different PDF, update the path passed to `ingest_document()` in `app/ingestion.py`.

## Run the API

Start the FastAPI server:

```powershell
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API docs:

```text
http://127.0.0.1:8000/docs
```

## Ask a Question

Send a POST request to `/ask`:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/ask" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"What are Docker volumes used for?"}'
```

The endpoint also accepts optional conversation history:

```json
{
  "question": "How does it help with persistence?",
  "history": [
    {
      "role": "user",
      "content": "What are Docker volumes?"
    },
    {
      "role": "assistant",
      "content": "Docker volumes store data outside a container lifecycle."
    }
  ]
}
```

Example response shape:

```json
{
  "answer": "Docker volumes provide persistent storage for containers...",
  "question asked": "What are Docker volumes used for?",
  "sources": [
    {
      "source": "sample.pdf",
      "page": 1,
      "vector_score": 0.82,
      "keyword_score": null,
      "rerank_score": 4.31,
      "data": "Retrieved chunk text..."
    }
  ]
}
```

## Evaluate Retrieval

Run the retrieval evaluation script:

```powershell
python -m app.evaluate
```

The script reads `data/evaluation.json`, asks each test question, checks whether the expected source and page appear in the returned sources, and prints the final hit rate.

## Notes

- Qdrant must be running before ingestion, querying, or evaluation.
- Run ingestion after changing the source PDF.
- The Qdrant `documents` collection is recreated during ingestion, so previous indexed chunks are replaced.
- `data/chunks.json` must exist before starting the API because BM25 search loads it at startup.
- The first run may download Sentence Transformer and cross-encoder model files.
- Answers are generated from retrieved document context. If no relevant context is found, the prompt instructs the model to say it does not know based on the provided documents.
