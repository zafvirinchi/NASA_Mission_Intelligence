import os
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional
import asyncio

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    if not question or not question.strip():
        return {"error": "Question is empty or invalid"}

    if not answer or not answer.strip():
        return {"error": "Answer is empty or invalid"}

    if not contexts or not isinstance(contexts, list):
        return {"error": "Contexts are empty or invalid"}

    contexts = [context for context in contexts if context and context.strip()]

    if not contexts:
        return {"error": "No valid context available for evaluation"}

    try:
        # TODO: Create evaluator LLM with model gpt-3.5-turbo
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                base_url="https://openai.vocareum.com/v1",
                api_key=os.getenv("OPENAI_API_KEY")
            )
        )

        # TODO: Create evaluator_embeddings with model test-embedding-3-small
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small",
                base_url="https://openai.vocareum.com/v1",
                api_key=os.getenv("OPENAI_API_KEY")
            )
        )

        # TODO: Define an instance for each metric to evaluate
        response_relevancy_metric = ResponseRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )

        faithfulness_metric = Faithfulness(
            llm=evaluator_llm
        )

        # TODO: Evaluate the response using the metrics
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts
        )

        async def run_evaluation():
            response_relevancy_score = await response_relevancy_metric.single_turn_ascore(sample)
            faithfulness_score = await faithfulness_metric.single_turn_ascore(sample)

            return {
                "response_relevancy": float(response_relevancy_score),
                "faithfulness": float(faithfulness_score)
            }

        try:
            results = asyncio.run(run_evaluation())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(run_evaluation())
            loop.close()

        # TODO: Return the evaluation results
        return results

    except Exception as e:
        return {"error": f"RAGAS evaluation failed: {str(e)}"}