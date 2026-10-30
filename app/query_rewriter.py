from groq import Groq
import os
from app.logging_config import get_logger

logger = get_logger(__name__)

class QueryRewriter:

    def __init__(self, api_key):
        logger.info("Initializing Groq query rewriter")
        self.client = Groq(
            api_key=api_key
        )

    def rewrite(self, question: str,history: list):
        logger.info("Rewriting query using %d history messages", len(history))
        conversation = ""

        for message in history:

                conversation += (
                    f"{message['role']}: "
                    f"{message['content']}\n"
                )

        prompt = f"""
        Rewrite the user's latest question
        as a standalone search query.

        Use the conversation history to resolve
        pronouns and references.

        Conversation history:
        {conversation}

        Latest question:
        {question}

        Return only the rewritten search query.
        """

#         prompt = f"""
# Rewrite the following user question
# into a clear, standalone search query.

# Do not answer the question.

# Original question:
# {question}

# Rewritten search query:
# """

        response = (
            self.client.chat.completions.create(
                model=os.environ.get("model_for_query"),

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0
            )
        )

        rewritten = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )
        logger.info("Query rewrite completed")
        return rewritten




import os
from langchain_ollama import ChatOllama


class Opensource_QueryRewriter:

    def __init__(self):
        logger.info("Initializing local Ollama query rewriter")
        self.client = ChatOllama(
            model=os.getenv(
                "open_MODEL_FOR_QUERY",
                "qwen2.5:3b"
            ),
            base_url="http://localhost:11434",
            temperature=0,
        )

    def rewrite(self, question: str, history: list) -> str:
        logger.info("Rewriting query locally using %d history messages", len(history))
        conversation = ""

        for message in history:
            conversation += (
                f"{message['role']}: "
                f"{message['content']}\n"
            )

        prompt = f"""
Rewrite the user's latest question
as a standalone search query.

Use the conversation history to resolve
pronouns and references.

Do not answer the question.
Return only the rewritten search query.

Conversation history:
{conversation}

Latest question:
{question}

Rewritten search query:
"""

        response = self.client.invoke(prompt)

        rewritten = response.content.strip()
        logger.info("Local query rewrite completed")
        return rewritten
