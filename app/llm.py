from groq import Groq
import os

class LLMService:

    def __init__(self, api_key):

        self.client = Groq(
            api_key=api_key
        )

    def generate(self, prompt):

        response = self.client.chat.completions.create(

            model=os.environ.get("model_for_generation"),

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content