import os
from langchain_ollama import ChatOllama
from app.logging_config import get_logger

logger = get_logger(__name__)

evaluator_llm = ChatOllama(
    model=os.getenv(
        "RAGAS_EVALUATION_MODEL_ollama",
        "qwen2.5:3b",
    ),
    base_url="http://localhost:11434",
    temperature=0,
)
logger.info("Initialized Ollama evaluator LLM")
