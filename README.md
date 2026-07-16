# RAG Documentation Assistant

A small FastAPI-based Retrieval-Augmented Generation (RAG) service that answers questions from local documents.

The project reads text files from the `documents` directory, chunks and embeds the text with Sentence Transformers, stores vectors in Qdrant, and uses Groq to generate answers from retrieved context.

## Features

- FastAPI API with a `/ask` endpoint
- Local document ingestion pipeline
- Sentence Transformer embeddings using `all-MiniLM-L6-v2`
- Qdrant vector storage through Docker Compose
- Groq chat completion for final answer generation

## Project Structure

```text
.
+-- app/
|   +-- chunking.py        # Splits document text into overlapping chunks
|   +-- embeddings.py      # Generates sentence-transformer embeddings
|   +-- ingestion.py       # Loads documents into Qdrant
|   +-- llm.py             # Calls the Groq API
|   +-- main.py            # FastAPI application
|   +-- prompt.py          # Prompt template
|   +-- qdrant_store.py    # Qdrant collection, insert, and search logic
|   +-- rag.py             # End-to-end RAG pipeline
+-- documents/
|   +-- docker.txt         # Example source document
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

Load the example document into Qdrant:

```powershell
python -m app.ingestion
```

By default, this ingests:

```text
documents/docker.txt
```

To use a different document, update the path in `app/ingestion.py`.

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

Example response:

```json
{
  "answer": "Docker volumes provide persistent storage for containers..."
}
```

## Notes

- Qdrant must be running before ingestion or querying.
- Run ingestion after changing files in the `documents` directory.
- The Qdrant collection is recreated during ingestion, so previous indexed chunks are replaced.
- Answers are generated only from retrieved document context. If no relevant context is found, the prompt instructs the model to say it does not know based on the provided documents.
