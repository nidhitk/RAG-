from groq import Groq
from langchain_ollama import ChatOllama
import os
from app.logging_config import get_logger

logger = get_logger(__name__)

class LLMService:

    def __init__(self, api_key):
        logger.info("Initializing Groq LLM client")
        self.client = Groq(
            api_key=api_key
        )

    def generate(self, prompt):
        logger.info("Generating response with Groq LLM")
        response = self.client.chat.completions.create(

            model=os.environ.get("model_for_generation"),

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response.choices[0].message.content
        logger.info("Groq LLM generated a %d-character response", len(answer))
        return answer


class Opensource_LLMService:

    def __init__(self):
        logger.info("Initializing local Ollama LLM client")
        self.client = ChatOllama(
            model=os.getenv(
                "OPEN_MODEL_FOR_GENERATION",
                "qwen2.5:3b"
            ),
            base_url="http://localhost:11434",
            temperature=0,
        )

    def generate(self, prompt: str) -> str:
        logger.info("Generating response with local Ollama LLM")
        response = self.client.invoke(prompt)
        answer = response.content
        logger.info("Local LLM generated a %d-character response", len(answer))
        return answer
