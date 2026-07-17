from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import RAGPipeline
from dotenv import load_dotenv
import os
load_dotenv()





app = FastAPI(
    title="RAG Documentation Assistant"
)


class QuestionRequest(BaseModel):

    question: str


rag = RAGPipeline(
    api_key=os.environ.get("GROQ_API_KEY")
)


@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    result = rag.ask(
        request.question
    )

    return result