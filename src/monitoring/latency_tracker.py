"""
Latency Tracker — computes p50, p95, p99 latency percentiles per evaluation engine.
"""
import statistics


def compute_percentiles(latencies: list[float]) -> dict:
    """
    Compute p50, p95, p99 from a list of latency values in ms.

    Returns dict with percentile stats or zeros if empty.
    """
    if not latencies:
        return {
            "count": 0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "avg_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    def percentile(data: list[float], pct: float) -> float:
        idx = (pct / 100) * (len(data) - 1)
        lower = int(idx)
        upper = min(lower + 1, len(data) - 1)
        frac  = idx - lower
        return round(data[lower] + frac * (data[upper] - data[lower]), 2)

    return {
        "count":  n,
        "min_ms": round(min(sorted_latencies), 2),
        "max_ms": round(max(sorted_latencies), 2),
        "avg_ms": round(statistics.mean(sorted_latencies), 2),
        "p50_ms": percentile(sorted_latencies, 50),
        "p95_ms": percentile(sorted_latencies, 95),
        "p99_ms": percentile(sorted_latencies, 99),
    }


def analyze_latency(logs: list[dict]) -> dict:
    """
    Break down latency stats from evaluation logs.

    Args:
        logs: List of dicts with latency_ms field

    Returns:
        Dict with overall and per-model latency percentiles
    """
    all_latencies = [
        l["latency_ms"] for l in logs
        if l.get("latency_ms") is not None
    ]

    # Per-model breakdown
    model_latencies: dict[str, list[float]] = {}
    for l in logs:
        if l.get("latency_ms") is not None and l.get("model_name"):
            model = l["model_name"]
            model_latencies.setdefault(model, []).append(l["latency_ms"])

    per_model = {
        model: compute_percentiles(lats)
        for model, lats in model_latencies.items()
    }

    return {
        "overall": compute_percentiles(all_latencies),
        "per_model": per_model,
    } 