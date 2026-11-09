# RAG Documentation Assistant

A FastAPI Retrieval-Augmented Generation application for asking questions about indexed PDF documents. Groq handles query rewriting and answer generation; Qdrant handles vector search; BM25 provides keyword search; and a cross-encoder reranks hybrid candidates.

## Features

- PDF upload and ingestion with page-aware metadata.
- Overlapping text chunks and Sentence Transformer embeddings.
- Qdrant vector search in the documents collection.
- Optional hybrid retrieval using Qdrant plus BM25.
- Cross-encoder reranking for hybrid retrieval.
- Groq query rewriting and grounded answer generation.
- Persistent chat sessions in PostgreSQL/Supabase.
- FastAPI Swagger documentation at /docs.
- Retrieval hit-rate evaluation and RAGAS comparison of both retrieval modes.
- Browser client in web/.

## Architecture

~~~text
PDF -> page extraction -> overlapping chunks -> embeddings -> Qdrant

Question + history -> Groq query rewriting -> Qdrant search
    -> optional BM25 search -> optional reranking
    -> grounded prompt -> Groq answer
~~~

With hybrid retrieval enabled, Qdrant and BM25 results are merged and reranked. With hybrid retrieval disabled, the pipeline uses the top Qdrant results directly without BM25 or reranking.

## Project structure

~~~text
.
+-- app/
|   +-- main.py             FastAPI application and routes
|   +-- rag.py              End-to-end RAG pipeline
|   +-- llm.py              Groq answer-generation client
|   +-- query_rewriter.py   Groq standalone-query rewriter
|   +-- ingestion.py        PDF ingestion and Qdrant indexing
|   +-- pdf_loader.py       PDF text extraction
|   +-- chunking.py         Overlapping text chunking
|   +-- embeddings.py       Sentence Transformer embeddings
|   +-- qdrant_store.py     Qdrant operations
|   +-- bm25_search.py      BM25 keyword search
|   +-- reranker.py         Cross-encoder reranking
|   +-- database.py         PostgreSQL models and chat persistence
|   +-- prompt.py           Grounded-answer prompt
|   +-- evaluate.py         Retrieval source/page evaluation
+-- evaluation/
|   +-- dataset.json        RAGAS questions and ground truth
|   +-- run_ragas.py        Hybrid versus vector-only RAGAS evaluation
|   +-- groq_llm.py         Groq evaluator for RAGAS
+-- data/                   Evaluation data and local artifacts
+-- documents/              Source documents, including sample.pdf
+-- web/                    Static browser client
+-- docker-compose.yml      Qdrant service
+-- requirements.txt        Python dependencies
+-- .env                    Local secrets and configuration
~~~

## Requirements

- Python 3.10 or later.
- Docker Desktop or another Docker installation.
- A Groq API key.
- PostgreSQL/Supabase for persistent chat endpoints.

## Configuration

Create a root-level .env file:

~~~env
GROQ_API_KEY=your_groq_api_key_here
model_for_query=llama-3.3-70b-versatile
model_for_generation=llama-3.3-70b-versatile
ragas_evaluation_model=llama-3.3-70b-versatile
DATABASE_URL=postgresql://postgres.<project-ref>:<password>@<pooler-host>:5432/postgres
~~~

Use a model ID supported by your Groq account. URL-encode special characters in the database password; for example, @ becomes %40. Never commit .env or expose its secrets.

| Variable | Purpose |
|---|---|
| GROQ_API_KEY | Groq authentication |
| model_for_query | Model used for standalone query rewriting |
| model_for_generation | Model used for answer generation |
| ragas_evaluation_model | Model used by RAGAS evaluation |
| DATABASE_URL | PostgreSQL/Supabase connection string |

## Installation

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
~~~

## Start Qdrant

~~~powershell
docker compose up -d
~~~

Qdrant is exposed at localhost:6333 and uses a collection named documents.

## Index documents

The command-line ingestion entry point indexes documents/sample.pdf:

~~~powershell
python -m app.ingestion
~~~

Ingestion extracts PDF text, creates 500-character chunks with 50-character overlap, generates 384-dimensional embeddings, recreates the documents collection, and stores chunk text with source, page, and chunk index metadata.

Re-ingestion replaces the existing Qdrant collection. Uploading through the API has the same replacement behavior.

## Run the API

~~~powershell
uvicorn app.main:app --reload
~~~

The API is available at http://127.0.0.1:8000. Interactive documentation is at http://127.0.0.1:8000/docs.

The first run may download embedding and reranker models. Ingest at least one PDF before starting the API because the pipeline loads the Qdrant corpus into BM25 during startup.

## API endpoints

### POST /ask

Runs the RAG pipeline without persistent chat storage.

~~~powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/ask -Method Post -ContentType application/json -Body '{"question":"What are Docker volumes used for?"}'
~~~

Request fields:

| Field | Type | Default | Description |
|---|---|---|---|
| question | string | required | User question |
| history | array | [] | Previous role/content messages |
| temperature | number | 0.0 | Groq temperature from 0.0 to 2.0 |
| rerank_top_k | integer | 3 | Returned context chunks from 1 to 20 |
| hybrid_retrieval | boolean | true | Enables BM25 merging and reranking |

The response includes answer, rewritten_query, retrieval_mode, context, and sources. retrieval_mode is hybrid_reranked or qdrant_only.

### POST /chat

Runs the same pipeline and persists the conversation. Omit session_id to create a session, or provide an existing session_id to continue it.

~~~powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/chat -Method Post -ContentType application/json -Body '{"message":"What are Docker volumes used for?","hybrid_retrieval":true}'
~~~

The database must be reachable for /chat. If database initialization fails at startup, chat endpoints return 503 while /ask remains independent of PostgreSQL.

### Chat session routes

- POST /chat/session creates an empty session.
- GET /chat/sessions lists saved sessions.
- GET /chat/{session_id} returns messages for one session.

All three routes require a working DATABASE_URL.

### POST /ingest

Uploads and indexes a PDF, then refreshes BM25:

~~~powershell
curl.exe -X POST -F "file=@documents/sample.pdf" http://127.0.0.1:8000/ingest
~~~

Only PDF files are accepted. The response includes the filename and indexed chunk count.

## Browser client

The browser client expects the API at http://127.0.0.1:8000 and is configured for a frontend on port 5500.

~~~powershell
python -m http.server 5500 --directory web
~~~

Open http://127.0.0.1:5500. The interface supports chat sessions, PDF upload, temperature, rerank top-k, and the hybrid retrieval toggle.

## Evaluation

### Retrieval hit rate

~~~powershell
python -m app.evaluate
~~~

This reads data/evaluation.json, runs each question through the pipeline, and checks whether the expected source and page appear in the returned sources.

### RAGAS comparison

~~~powershell
python -m evaluation.run_ragas
~~~

The script runs evaluation/dataset.json twice:

1. hybrid_retrieval=True: Qdrant plus BM25, followed by reranking.
2. hybrid_retrieval=False: Qdrant vector results only.

Outputs:

- evaluation/results/ragas_results_hybrid.csv
- evaluation/results/ragas_results_vector_only.csv
- evaluation/results/ragas_comparison.csv

The comparison file contains mean faithfulness, answer_relevancy, context_precision, and context_recall for each strategy. RAGAS makes additional Groq requests and may consume significant quota.

## Troubleshooting

### Chat endpoints return 503

Check the startup log for Database initialization failed. Verify DATABASE_URL from Supabase Connect, URL-encode the password, and confirm that the pooler host and port are reachable. The /ask endpoint does not require PostgreSQL.

### Qdrant errors

Check Qdrant with:

~~~powershell
docker compose ps
~~~

Start it with:

~~~powershell
docker compose up -d
~~~

### Groq errors

Check GROQ_API_KEY and confirm both configured model IDs are valid Groq models. Restart the API after changing .env.

## Important notes

- The API allows local frontend origins on ports 5500 and 3000.
- Ingestion recreates the Qdrant collection and replaces previously indexed chunks.
- Answers are instructed to use only retrieved context and to say they do not know when the context is insufficient.
- Rotate secrets if they have been exposed or committed accidentally.
