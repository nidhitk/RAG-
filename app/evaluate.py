import json

from app.rag import RAGPipeline
from dotenv import load_dotenv
import os
load_dotenv()


def evaluate():

    with open(
        "data/evaluation.json",
        "r",
        encoding="utf-8"
    ) as file:

        test_cases = json.load(file)

    rag = RAGPipeline(
        api_key=os.environ.get("GROQ_API_KEY")
    )

    hits = 0

    for test_case in test_cases:

        result = rag.ask(
            question=test_case["question"]
        )

        sources = result["sources"]

        expected_source = (
            test_case["expected_source"]
        )

        expected_page = (
            test_case["expected_page"]
        )

        found = False
        # checks source example sample.pdf and page number of that source
        for source in sources:

            if (
                source["source"]
                == expected_source
                and
                source["page"]
                == expected_page
            ):

                found = True
                break

        if found:

            hits += 1

            print("✅ HIT")

        else:

            print("❌ MISS")

        print(
            test_case["question"]
        )

    hit_rate = (
        hits / len(test_cases)
    )

    print(
        f"Hit Rate: {hit_rate:.2%}"
    )


if __name__ == "__main__":

    evaluate()