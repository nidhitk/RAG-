from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import RAGPipeline
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


rag = RAGPipeline(
    api_key=os.environ.get("GROQ_API_KEY")
)
logger.info("FastAPI application initialized")


@app.post("/ask")
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
