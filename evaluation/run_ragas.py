import json
import os
import sys
from types import ModuleType
from pathlib import Path

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from evaluation.ollama_llm import evaluator_llm

# RAGAS 0.4.3 imports the retired LangChain Community Vertex AI module even
# when Vertex AI is not used.  RAGAS only references this class when checking
# whether an evaluator is a Vertex model; this evaluation uses ChatGroq.
try:
    from langchain_community.chat_models.vertexai import ChatVertexAI
except ModuleNotFoundError:
    vertexai_module = ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # pragma: no cover - compatibility for RAGAS import
        pass

    vertexai_module.ChatVertexAI = ChatVertexAI
    sys.modules[vertexai_module.__name__] = vertexai_module

from ragas import evaluate
from ragas.llms.base import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from app.rag import RAGPipeline
from app.logging_config import get_logger


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
logger = get_logger(__name__)


def load_test_cases():
    path = ROOT / "evaluation" / "dataset.json"

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_dataset(pipeline, test_cases):
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for case in test_cases:
        question = case["question"]
        expected_answer = case["ground_truth"]

        # Run your actual RAG pipeline.
        result = pipeline.ask(question)

        questions.append(question)
        answers.append(result["answer"])
        contexts.append(result["context"])
        ground_truths.append(expected_answer)

        logger.info("Evaluated question; retrieved %d chunks", len(result["context"]))

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def main():
    # Answers and retrieved passages may contain characters outside Windows'
    # legacy console code page; evaluation logging must not abort the run.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logger.info("Starting RAGAS evaluation")
    api_key = os.environ["GROQ_API_KEY"]

    pipeline = RAGPipeline(api_key=api_key)
    test_cases = load_test_cases()

    dataset = build_dataset(pipeline, test_cases)

    # Example evaluator: use a compatible Groq model.

    # evaluator_llm = ChatGroq(
    #     model=os.getenv(
    #         "ragas_evaluation_model",
    #         "llama-3.3-70b-versatile",
    #     ),
    #     api_key=api_key,
    #     temperature=0,
    # )

    # Groq accepts a single completion per request.  RAGAS requests several
    # completions for answer relevancy, so issue separate single-completion
    # requests instead of sending Groq an unsupported ``n > 1`` parameter.
    ragas_evaluator = LangchainLLMWrapper(evaluator_llm, bypass_n=True)
    # Groq supports only one completion per request.  One generated question
    # keeps answer-relevancy evaluation compatible and avoids tripling runtime.
    answer_relevancy.strictness = 1

    # Use an embedding model compatible with RAGAS.
    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=ragas_evaluator,
        embeddings=evaluator_embeddings,
        # Keep Groq calls within its tokens-per-minute quota.  RAGAS retries
        # transient 429 responses using exponential backoff.
        run_config=RunConfig(max_workers=1, timeout=300, max_retries=10),
    )

    output_dir = ROOT / "evaluation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = result.to_pandas()
    output_path = output_dir / "ragas_results.csv"
    df.to_csv(output_path, index=False)

    logger.info("RAGAS evaluation completed; results saved to %s", output_path)
    logger.info("RAGAS scores: %s", result)


if __name__ == "__main__":
    main()
