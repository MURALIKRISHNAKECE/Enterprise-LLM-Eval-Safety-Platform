"""
Safety Evaluator - DeepEval (Toxicity, Bias, Hallucination)
Runs synchronous DeepEval metrics in a thread pool.
Supports mock mode for CI dry-runs.
"""
import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor

from src.core.config import settings

executor = ThreadPoolExecutor(max_workers=5)


def _is_mock() -> bool:
    return settings.MOCK_EVALUATORS or os.getenv("MOCK_EVALUATORS", "false").lower() == "true"


async def evaluate_safety(
    input_text:    str,
    actual_output: str,
    context:       list[str] | None = None,
) -> dict:
    """
    Evaluate LLM output for toxicity, bias, and hallucination.
    Uses mock scores when MOCK_EVALUATORS = true.
    """
    if _is_mock():
        await asyncio.sleep(0.05)
        return {
            "toxicity_score":      0.02,
            "bias_score":          0.03,
            "hallucination_score": 0.05,
            "overall_passed":      True,
            "latency_ms":          50.0,
        }

    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import ToxicityMetric, BiasMetric, HallucinationMetric
    from src.core.rate_limiter import with_semaphore, retry_with_backoff

    start = time.monotonic()

    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        context=context or [],
    )

    toxicity_metric      = ToxicityMetric(threshold=settings.TOXICITY_THRESHOLD)
    bias_metric          = BiasMetric(threshold=0.5)
    hallucination_metric = HallucinationMetric(threshold=settings.HALLUCINATION_THRESHOLD)

    loop = asyncio.get_running_loop()

    async def _run_metric(metric):
        return await with_semaphore(
            retry_with_backoff(
                loop.run_in_executor, executor, metric.measure, test_case
            )
        )

    tasks = [_run_metric(toxicity_metric), _run_metric(bias_metric)]
    if context:
        tasks.append(_run_metric(hallucination_metric))

    await asyncio.gather(*tasks)

    latency_ms = (time.monotonic() - start) * 1000

    return {
        "toxicity_score":      float(toxicity_metric.score),
        "bias_score":          float(bias_metric.score),
        "hallucination_score": float(hallucination_metric.score) if context else 0.0,
        "overall_passed":      toxicity_metric.is_successful() and bias_metric.is_successful(),
        "latency_ms":          round(latency_ms, 2),
    }