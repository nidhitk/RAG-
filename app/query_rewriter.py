from groq import Groq


class QueryRewriter:

    def __init__(self, api_key):

        self.client = Groq(
            api_key=api_key
        )

    def rewrite(self, question: str,history: list):
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
                model="llama-3.1-8b-instant",

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0
            )
        )

        return (
            response
            .choices[0]
            .message
            .content
            .strip()
        )