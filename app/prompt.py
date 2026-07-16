def build_prompt(
    question: str,
    context: str
):

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