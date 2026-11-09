from groq import Groq
import os
from app.logging_config import get_logger

logger = get_logger(__name__)

class LLMService:

    def __init__(self, api_key):
        logger.info("Initializing Groq LLM client")
        self.client = Groq(
            api_key=api_key
        )

    def generate(self, prompt, temperature=0.0):
        logger.info("Generating response with Groq LLM")
        response = self.client.chat.completions.create(

            model=os.environ.get("model_for_generation"),

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
        )

        answer = response.choices[0].message.content
        logger.info("Groq LLM generated a %d-character response", len(answer))
        return answer

