from app.logging_config import get_logger

logger = get_logger(__name__)


def build_prompt(
    question: str,
    context: str
):
    logger.debug("Building prompt with %d context characters", len(context))
    return f"""
You are a helpful AI assistant.

Answer the question using only the provided context.

If the answer cannot be found in the context,
say: "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
