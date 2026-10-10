"""
Alerting module — sends Slack webhook alerts when evaluation thresholds are breached.
Configure ALERT_WEBHOOK_URL in .env to enable. Fails silently if not configured.
"""
import httpx
import logging
from src.core.config import settings

logger = logging.getLogger(__name__)


def _build_slack_message(violations: list[dict]) -> dict:
    """Build a Slack Block Kit message payload from a list of violations."""
    header = {
        "type": "header",
        "text": {"type": "plain_text", "text": "🚨 LLM Safety Platform — Threshold Alert", "emoji": True}
    }

    blocks = [header, {"type": "divider"}]

    for v in violations:
        blocks.append({
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Metric:*\n{v['metric']}"},
                {"type": "mrkdwn", "text": f"*Value:*\n`{v['value']:.4f}`"},
                {"type": "mrkdwn", "text": f"*Threshold:*\n`{v['threshold']:.4f}`"},
                {"type": "mrkdwn", "text": f"*Status:*\n{v['status']}"},
            ]
        })

    blocks.append({"type": "divider"})
    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": "Enterprise LLM Evaluation & AI Safety Platform"}]
    })

    return {"blocks": blocks}


async def send_alert(violations: list[dict]) -> None:
    """
    Send alert to Slack if ALERT_WEBHOOK_URL is configured.

    Args:
        violations: list of dicts with keys: metric, value, threshold, status
    """
    if not settings.ALERT_WEBHOOK_URL:
        logger.debug("ALERT_WEBHOOK_URL not set — skipping alert.")
        return

    if not violations:
        return

    payload = _build_slack_message(violations)

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(settings.ALERT_WEBHOOK_URL, json=payload)
            if resp.status_code != 200:
                logger.warning(f"Slack alert returned HTTP {resp.status_code}: {resp.text}")
            else:
                logger.info(f"Alert sent: {len(violations)} violation(s).")
    except Exception as e:
        logger.error(f"Failed to send Slack alert: {e}")


def check_thresholds(safety: dict, rag: dict) -> list[dict]:
    """
    Compare evaluation scores against configured thresholds.

    Returns a list of violation dicts (empty if all pass).
    """
    violations = []

    checks = [
        ("Toxicity",         safety.get("toxicity_score", 0.0),        settings.TOXICITY_THRESHOLD,      "above"),
        ("Hallucination",    safety.get("hallucination_score", 0.0),   settings.HALLUCINATION_THRESHOLD, "above"),
        ("Faithfulness",     rag.get("faithfulness_score", 1.0),        settings.FAITHFULNESS_THRESHOLD,  "below"),
        ("Answer Relevancy", rag.get("relevancy_score", 1.0),           settings.RELEVANCY_THRESHOLD,     "below"),
    ]

    for metric, value, threshold, direction in checks:
        breached = (direction == "above" and value > threshold) or \
                   (direction == "below" and value < threshold)
        if breached:
            violations.append({
                "metric":    metric,
                "value":     value,
                "threshold": threshold,
                "status":    f"⚠️ {'Exceeds' if direction == 'above' else 'Below'} threshold",
            })

    return violations