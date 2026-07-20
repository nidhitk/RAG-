import json

from app.rag import RAGPipeline
from dotenv import load_dotenv
import os
load_dotenv()

def evaluate():

    with open(
        "data/evaluation.json",
        "r"
    ) as file:

        test_cases = json.load(file)

    rag = RAGPipeline(
        api_key=os.environ.get("GROQ_API_KEY")
    )

    for test_case in test_cases:

        result = rag.ask(
            question=test_case["question"]
        )

        print(
            "Question:",
            test_case["question"]
        )

        print(
            "Expected:",
            test_case["expected_answer"]
        )

        print(
            "Actual:",
            result["answer"]
        )

        print("-" * 50)


if __name__ == "__main__":

    evaluate()