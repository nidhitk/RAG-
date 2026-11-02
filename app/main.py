from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.rag import RAGPipeline
from app.database import (
    chat_session_exists,
    create_chat_session,
    get_messages,
    init_db,
    save_message,
)
from dotenv import load_dotenv
import os
from app.logging_config import get_logger

logger = get_logger(__name__)
load_dotenv()





app = FastAPI(
    title="RAG Documentation Assistant"
)


class QuestionRequest(BaseModel):

    question: str
    history: list = []


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


rag = RAGPipeline(
    api_key=os.environ.get("GROQ_API_KEY")
)
init_db()
logger.info("FastAPI application initialized")


@app.post("/ask",description="for testing purpose")
def ask_question(
    request: QuestionRequest
):
    logger.info("Received /ask request with %d history messages", len(request.history))
    result = rag.ask(
        question=request.question,
        history=request.history
    )

    logger.info("Completed /ask request")
    return result


@app.post("/chat")
def chat(request: ChatRequest):
    """Chat with persisted conversation history."""
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
    )
    save_message(session_id, "assistant", result["answer"])

    return {
        "session_id": session_id,
        **result,
    }


@app.get("/chat/{session_id}")
def chat_history(session_id: str):
    if not chat_session_exists(session_id):
        raise HTTPException(status_code=404, detail="Chat session not found")

    return {
        "session_id": session_id,
        "messages": get_messages(session_id),
    }
