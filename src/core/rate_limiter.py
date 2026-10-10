"""
Rate Limiter — asyncio.Semaphore-based concurrency control for external API calls.
Includes exponential backoff with jitter for 429 / rate-limit failures.
"""
import asyncio
import random
import logging
import time
from functools import wraps

from src.core.config import settings

logger = logging.getLogger(__name__)

# One shared semaphore for all Gemini calls
_gemini_semaphore = asyncio.Semaphore(settings.GEMINI_MAX_CONCURRENT)


async def with_semaphore(coro):
    """Wrap a coroutine with the Gemini semaphore."""
    async with _gemini_semaphore:
        return await coro


async def retry_with_backoff(fn, *args, max_retries: int | None = None, **kwargs):
    """
    Call async fn(*args, **kwargs) with exponential backoff + jitter.
    Retries on any exception (typically 429 / rate limit).

    Args:
        fn: async callable
        max_retries: overrides settings.GEMINI_MAX_RETRIES
    """
    retries   = max_retries if max_retries is not None else settings.GEMINI_MAX_RETRIES
    last_exc  = None

    for attempt in range(retries + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt == retries:
                break
            wait = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(
                f"[rate_limiter] attempt {attempt + 1}/{retries} failed: {exc}. "
                f"Retrying in {wait:.1f}s..."
            )
            await asyncio.sleep(wait)

    raise last_exc


class MockEvaluator:
    """
    Returns deterministic mock scores for local tests and CI dry-runs.
    Activated when settings.MOCK_EVALUATORS = true.
    """

    @staticmethod
    async def safety() -> dict:
        await asyncio.sleep(0.05)   # simulate latency
        return {
            "toxicity_score":      0.02,
            "bias_score":          0.03,
            "hallucination_score": 0.05,
            "overall_passed":      True,
            "latency_ms":          50.0,
        }

    @staticmethod
    async def rag() -> dict:
        await asyncio.sleep(0.05)
        return {
            "faithfulness_score": 0.92,
            "relevancy_score":    0.88,
            "rag_passed":         True,
            "latency_ms":         50.0,
        }