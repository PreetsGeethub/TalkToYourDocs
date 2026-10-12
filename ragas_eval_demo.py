
import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)


# Two controlled examples: one good case and one retrieval failure.
TEST_CASES = [
    {
        "name": "Good answer and relevant evidence",
        "user_input": "Where were the Docker images published?",
        "retrieved_contexts": [
            (
                "GitHub Actions builds a Docker image and publishes it "
                "to GitHub Container Registry (GHCR)."
            ),
            "The frontend uses Next.js and TypeScript.",
        ],
        "response": (
            "The Docker images were published to "
            "GitHub Container Registry (GHCR)."
        ),
        "reference": (
            "The Docker images were published to "
            "GitHub Container Registry (GHCR)."
        ),
    },
    {
        "name": "Relevant evidence is missing",
        "user_input": "Where were the Docker images published?",
        "retrieved_contexts": [
            "The backend was deployed on AWS EC2 using Docker and Nginx."
        ],
        "response": "The backend was deployed to AWS EC2.",
        "reference": (
            "The Docker images were published to "
            "GitHub Container Registry (GHCR)."
        ),
    },
]


async def main():
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found. Check your .env file."
        )

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    try:
        # LLM used by Ragas to judge the examples.
        evaluator_llm = llm_factory(
            "openrouter/free",
            client=client,
            max_tokens=4096,
        )

        # Local embeddings: no separate paid embedding API.
        evaluator_embeddings = embedding_factory(
            "huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
        )

        metrics = {
            "Faithfulness": Faithfulness(llm=evaluator_llm),
            "Answer Relevancy": AnswerRelevancy(
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
            ),
            "Context Precision": ContextPrecision(llm=evaluator_llm),
            "Context Recall": ContextRecall(llm=evaluator_llm),
        }

        for case in TEST_CASES:
            print("\n" + "=" * 60)
            print(case["name"])
            print("=" * 60)

            print("Question:", case["user_input"])
            print("Answer:", case["response"])

            for metric_name, metric in metrics.items():
                try:
                    if metric_name == "Faithfulness":
                        result = await metric.ascore(
                            user_input=case["user_input"],
                            response=case["response"],
                            retrieved_contexts=case["retrieved_contexts"],
                        )

                    elif metric_name == "Answer Relevancy":
                        result = await metric.ascore(
                            user_input=case["user_input"],
                            response=case["response"],
                        )

                    else:
                        result = await metric.ascore(
                            user_input=case["user_input"],
                            reference=case["reference"],
                            retrieved_contexts=case["retrieved_contexts"],
                        )

                    print(f"{metric_name}: {result.value:.2f}")

                except Exception as exc:
                    print(f"{metric_name}: FAILED ({exc})")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
