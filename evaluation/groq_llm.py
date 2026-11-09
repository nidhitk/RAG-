import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from app.logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)

evaluator_llm = ChatGroq(
    model=os.getenv("ragas_evaluation_model", "llama-3.3-70b-versatile"),
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0,
)
logger.info("Initialized Groq evaluator LLM")
