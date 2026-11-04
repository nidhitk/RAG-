from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.rag import RAGPipeline
from app.ingestion import ingest_document
from app.database import (
    chat_session_exists,
    create_chat_session,
    get_messages,
    init_db,
    list_chat_sessions,
    save_message,
)
from dotenv import load_dotenv
import os
from pathlib import Path
from uuid import uuid4
from app.logging_config import get_logger

logger = get_logger(__name__)
load_dotenv()





app = FastAPI(
    title="RAG Documentation Assistant"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):

    question: str
    history: list = []
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    rerank_top_k: int = Field(default=3, ge=1, le=20)
    hybrid_retrieval: bool = True


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    rerank_top_k: int = Field(default=3, ge=1, le=20)
    hybrid_retrieval: bool = True


rag = RAGPipeline(
    api_key=os.environ.get("GROQ_API_KEY")
)
database_ready = False
try:
    init_db()
    database_ready = True
except Exception:
    logger.exception("Database initialization failed; chat endpoints will return 503")
logger.info("FastAPI application initialized")


def require_database():
    if not database_ready:
        raise HTTPException(
            status_code=503,
            detail="Chat database is unavailable. Check the Supabase DATABASE_URL.",
        )


@app.post("/ask",description="for testing purpose")
def ask_question(
    request: QuestionRequest
):
    logger.info("Received /ask request with %d history messages", len(request.history))
    result = rag.ask(
        question=request.question,
        history=request.history,
        temperature=request.temperature,
        rerank_top_k=request.rerank_top_k,
        hybrid_retrieval=request.hybrid_retrieval,
    )

    logger.info("Completed /ask request")
    return result

@app.post("/chat")
def chat(request: ChatRequest):
    """Chat with persisted conversation history."""
    require_database()
    session_id = request.session_id

    if session_id is None:
        session_id = create_chat_session()
    elif not chat_session_exists(session_id):
        raise HTTPException(status_code=404, detail="Chat session not found")

    history = get_messages(session_id)
    save_message(session_id, "user", request.message)

    result = rag.ask(
        question=request.message,
        history=history,
        temperature=request.temperature,
        rerank_top_k=request.rerank_top_k,
        hybrid_retrieval=request.hybrid_retrieval,
    )
    save_message(session_id, "assistant", result["answer"])

    return {
        "session_id": session_id,
        **result,
    }
@app.post("/chat/session")
def create_session():
    require_database()
    return {"session_id": create_chat_session()}

@app.get("/chat/sessions")
def chat_sessions():
    require_database()
    return {"sessions": list_chat_sessions()}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload a PDF, replace the Qdrant index, and refresh BM25."""
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = upload_dir / f"{uuid4().hex}_{filename}"

    try:
        with saved_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                destination.write(chunk)

        chunk_count = ingest_document(str(saved_path))
        rag.refresh_bm25()
        return {
            "message": "Document ingested successfully",
            "filename": filename,
            "chunks_indexed": chunk_count,
        }
    except Exception as error:
        logger.exception("Document ingestion failed for %s", filename)
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {error}")
    finally:
        saved_path.unlink(missing_ok=True)
@app.get("/chat/{session_id}")
def chat_history(session_id: str):
    require_database()
    if not chat_session_exists(session_id):
        raise HTTPException(status_code=404, detail="Chat session not found")

    return {
        "session_id": session_id,
        "messages": get_messages(session_id),
    }
