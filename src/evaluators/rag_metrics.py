"""
RAG Evaluator - RAGAS (Faithfulness, Answer Relevancy)
Uses Gemini via LangChain. Supports mock mode for CI dry-runs.
"""
import os
import sys
import re
import time
import types

from dotenv import load_dotenv

load_dotenv(override=True)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"]    = "false"

# Patch missing vertexai module
try:
    import langchain_community.chat_models
    if not hasattr(langchain_community.chat_models, "vertexai"):
        mock_vertexai = types.ModuleType("vertexai")
        try:
            from langchain_google_vertexai import ChatVertexAI
            mock_vertexai.ChatVertexAI = ChatVertexAI
        except ImportError:
            class DummyChatVertexAI:
                pass
            mock_vertexai.ChatVertexAI = DummyChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = mock_vertexai
        setattr(langchain_community.chat_models, "vertexai", mock_vertexai)
except Exception:
    pass

from src.core.config import settings


def _is_mock() -> bool:
    return settings.MOCK_EVALUATORS or os.getenv("MOCK_EVALUATORS", "false").lower() == "true"


def _clean_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _build_evaluator():
    from ragas.llms import LangchainLLMWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    base_llm = ChatGoogleGenerativeAI(
        model=settings.DEFAULT_EVAL_MODEL,
        google_api_key=api_key,
        temperature=0.0,
    )
    evaluator_llm = LangchainLLMWrapper(base_llm)

    orig = evaluator_llm.generate_text

    async def _patched_generate(*args, **kwargs):
        if kwargs.get("n", 1) > 1:
            kwargs["n"] = 1
        result = await orig(*args, **kwargs)
        for gen_list in result.generations:
            for gen in gen_list:
                if hasattr(gen, "text") and gen.text:
                    gen.text = _clean_json(gen.text)
                if hasattr(gen, "message") and hasattr(gen.message, "content"):
                    if isinstance(gen.message.content, str):
                        gen.message.content = _clean_json(gen.message.content)
        return result

    evaluator_llm.generate_text = _patched_generate

    evaluator_embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=api_key,
    )
    return evaluator_llm, evaluator_embeddings


async def evaluate_rag_pipeline(
    query:    str,
    response: str,
    context:  list[str],
) -> dict:
    """
    Evaluate RAG output using RAGAS. Supports mock mode.
    """
    if _is_mock():
        import asyncio
        await asyncio.sleep(0.05)
        return {
            "faithfulness_score": 0.92,
            "relevancy_score":    0.88,
            "rag_passed":         True,
            "latency_ms":         50.0,
        }

    from ragas import SingleTurnSample
    from ragas.metrics import Faithfulness, AnswerRelevancy
    from src.core.rate_limiter import with_semaphore, retry_with_backoff

    start = time.monotonic()

    evaluator_llm, evaluator_embeddings = _build_evaluator()

    sample = SingleTurnSample(
        user_input=query,
        response=response,
        retrieved_contexts=context,
    )

    faithfulness_scorer = Faithfulness(llm=evaluator_llm)
    relevancy_scorer    = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    if hasattr(relevancy_scorer, "strictness"):
        relevancy_scorer.strictness = 1

    faithfulness_score = await with_semaphore(
        retry_with_backoff(faithfulness_scorer.single_turn_ascore, sample)
    )
    relevancy_score = await with_semaphore(
        retry_with_backoff(relevancy_scorer.single_turn_ascore, sample)
    )

    latency_ms = (time.monotonic() - start) * 1000

    return {
        "faithfulness_score": float(faithfulness_score),
        "relevancy_score":    float(relevancy_score),
        "rag_passed":         bool(
            faithfulness_score >= settings.FAITHFULNESS_THRESHOLD
            and relevancy_score >= settings.RELEVANCY_THRESHOLD
        ),
        "latency_ms": round(latency_ms, 2),
    }