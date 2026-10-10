"""
Production Trace Replay — samples ~5% of production requests,
stores PII-masked traces, and flags low-quality results.
"""
import hashlib
import random
import logging
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.database import ProductionTrace, EvaluationLog
from src.monitoring.pii_detection import apply_pii_policy

logger = logging.getLogger(__name__)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def should_sample() -> bool:
    """Returns True ~TRACE_SAMPLE_RATE % of the time."""
    return random.random() < settings.TRACE_SAMPLE_RATE


async def store_trace(
    db:            AsyncSession,
    query:         str,
    response:      str,
    eval_log:      EvaluationLog,
) -> ProductionTrace | None:
    """
    Store a PII-masked trace of a sampled production request.
    Flags traces that fail quality or safety thresholds.
    """
    if not should_sample():
        return None

    try:
        query_masked    = apply_pii_policy(query,    "mask")
        response_masked = apply_pii_policy(response, "mask")

        # Determine if this trace should be flagged
        flagged     = False
        flag_reason = None

        if eval_log.safety_passed is False:
            flagged     = True
            flag_reason = "safety_failed"
        elif eval_log.faithfulness_score is not None and \
             eval_log.faithfulness_score < settings.FAITHFULNESS_THRESHOLD:
            flagged     = True
            flag_reason = f"low_faithfulness:{eval_log.faithfulness_score:.3f}"
        elif eval_log.latency_ms is not None and eval_log.latency_ms > 5000:
            flagged     = True
            flag_reason = f"high_latency:{eval_log.latency_ms:.0f}ms"

        trace = ProductionTrace(
            timestamp       = datetime.now(timezone.utc),
            query_hash      = _hash_text(query),
            response_hash   = _hash_text(response),
            query_masked    = query_masked,
            response_masked = response_masked,
            model_name      = eval_log.model_name,
            model_version   = eval_log.model_version,
            prompt_version  = eval_log.prompt_version,
            latency_ms      = eval_log.latency_ms,
            input_tokens    = eval_log.input_tokens,
            output_tokens   = eval_log.output_tokens,
            eval_log_id     = eval_log.id,
            flagged         = flagged,
            flag_reason     = flag_reason,
        )
        db.add(trace)
        await db.commit()
        await db.refresh(trace)

        if flagged:
            logger.warning(f"[trace_replay] Flagged trace #{trace.id}: {flag_reason}")

        return trace

    except Exception as e:
        logger.error(f"[trace_replay] Failed to store trace: {e}")
        return None


async def get_flagged_traces(db: AsyncSession, limit: int = 50) -> list[dict]:
    """Return recent flagged traces for review."""
    from sqlalchemy import select
    result = await db.execute(
        select(ProductionTrace)
        .where(ProductionTrace.flagged == True)
        .order_by(ProductionTrace.timestamp.desc())
        .limit(limit)
    )
    traces = result.scalars().all()
    return [
        {
            "id":               t.id,
            "timestamp":        t.timestamp,
            "model_name":       t.model_name,
            "model_version":    t.model_version,
            "latency_ms":       t.latency_ms,
            "flag_reason":      t.flag_reason,
            "query_masked":     t.query_masked,
            "response_masked":  t.response_masked,
        }
        for t in traces
    ]