from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import Dict, List
import os
import asyncio

try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, ResponseRelevancy, Faithfulness, RougeScore
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def get_openai_base_url():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key.startswith("voc"):
        return "https://openai.vocareum.com/v1"
    return None


def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    if not question or not answer or not contexts:
        return {"error": "Question, answer, and contexts are required"}

    contexts = [c for c in contexts if c and c.strip()]

    if not contexts:
        return {"error": "No valid contexts provided"}

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = get_openai_base_url()

        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                api_key=api_key,
                base_url=base_url
            )
        )

        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=api_key,
                base_url=base_url
            )
        )

        response_relevancy_metric = ResponseRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )

        faithfulness_metric = Faithfulness(llm=evaluator_llm)
        bleu_metric = BleuScore()
        rouge_metric = RougeScore()

        reference_text = " ".join(contexts)[:2000]

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference_text
        )

        async def run_evaluation():
            return {
                "response_relevancy": float(await response_relevancy_metric.single_turn_ascore(sample)),
                "faithfulness": float(await faithfulness_metric.single_turn_ascore(sample)),
                "bleu_score": float(await bleu_metric.single_turn_ascore(sample)),
                "rouge_score": float(await rouge_metric.single_turn_ascore(sample))
            }

        try:
            return asyncio.run(run_evaluation())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(run_evaluation())
            loop.close()
            return result

    except Exception as e:
        return {"error": f"RAGAS evaluation failed: {str(e)}"}