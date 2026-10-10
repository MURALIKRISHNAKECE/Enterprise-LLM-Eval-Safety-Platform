"""
Drift Detection — baseline vs recent comparison with statistical methods.
Includes PSI, KS test, and basic embedding-distribution drift.
"""
import math
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    metric:       str
    baseline_avg: float
    recent_avg:   float
    drift_pct:    float
    drifted:      bool
    direction:    str
    ks_statistic: float = 0.0
    ks_p_value:   float = 1.0
    psi:          float = 0.0


# ---------------------------------------------------------------------------
# PSI (Population Stability Index)
# ---------------------------------------------------------------------------

def compute_psi(baseline: list[float], recent: list[float], bins: int = 10) -> float:
    """
    PSI < 0.1  : stable
    PSI 0.1-0.2: slight change
    PSI > 0.2  : significant shift
    """
    if not baseline or not recent:
        return 0.0

    all_vals = baseline + recent
    min_v, max_v = min(all_vals), max(all_vals)
    if min_v == max_v:
        return 0.0

    def bin_counts(data):
        counts = [0] * bins
        for v in data:
            idx = min(int((v - min_v) / (max_v - min_v) * bins), bins - 1)
            counts[idx] += 1
        return counts

    b_counts = bin_counts(baseline)
    r_counts = bin_counts(recent)
    n_b, n_r = len(baseline), len(recent)
    eps      = 1e-8

    psi = 0.0
    for b, r in zip(b_counts, r_counts):
        b_pct = (b / n_b) + eps
        r_pct = (r / n_r) + eps
        psi  += (r_pct - b_pct) * math.log(r_pct / b_pct)

    return round(psi, 4)


# ---------------------------------------------------------------------------
# KS Test (pure Python — no scipy needed)
# ---------------------------------------------------------------------------

def compute_ks(baseline: list[float], recent: list[float]) -> tuple[float, float]:
    """
    Two-sample KS statistic.
    Returns (statistic, approximate_p_value).
    p < 0.05 = statistically significant difference.
    """
    if not baseline or not recent:
        return 0.0, 1.0

    combined = sorted(set(baseline + recent))
    n1, n2   = len(baseline), len(recent)
    b_sorted = sorted(baseline)
    r_sorted = sorted(recent)

    def ecdf(data, x):
        count = sum(1 for v in data if v <= x)
        return count / len(data)

    ks_stat = max(abs(ecdf(b_sorted, x) - ecdf(r_sorted, x)) for x in combined)

    n     = (n1 * n2) / (n1 + n2)
    z     = ks_stat * math.sqrt(n)
    p_val = 2 * math.exp(-2 * z * z) if z > 0 else 1.0
    p_val = max(0.0, min(1.0, p_val))

    return round(ks_stat, 4), round(p_val, 4)


# ---------------------------------------------------------------------------
# Core drift calculation
# ---------------------------------------------------------------------------

def calculate_drift(
    baseline_scores:     list[float],
    recent_scores:       list[float],
    metric_name:         str,
    drift_threshold_pct: float = 5.0,
    higher_is_better:    bool  = True,
) -> DriftResult:
    if not baseline_scores or not recent_scores:
        return DriftResult(
            metric=metric_name, baseline_avg=0.0, recent_avg=0.0,
            drift_pct=0.0, drifted=False, direction="stable",
        )

    baseline_avg = sum(baseline_scores) / len(baseline_scores)
    recent_avg   = sum(recent_scores)   / len(recent_scores)

    drift_pct = 0.0 if baseline_avg == 0 else \
        ((recent_avg - baseline_avg) / baseline_avg) * 100

    if higher_is_better:
        drifted   = drift_pct < -drift_threshold_pct
        direction = "degraded" if drift_pct < 0 else "improved"
    else:
        drifted   = drift_pct > drift_threshold_pct
        direction = "degraded" if drift_pct > 0 else "improved"

    ks_stat, ks_p = compute_ks(baseline_scores, recent_scores)
    psi           = compute_psi(baseline_scores, recent_scores)

    # Only apply statistical tests when sample size is sufficient (>=30 per window)
    if len(baseline_scores) >= 30 and len(recent_scores) >= 30:
        if ks_p < 0.05 or psi > 0.2:
            drifted = True

    return DriftResult(
        metric       = metric_name,
        baseline_avg = round(baseline_avg, 4),
        recent_avg   = round(recent_avg,   4),
        drift_pct    = round(drift_pct,    2),
        drifted      = drifted,
        direction    = direction,
        ks_statistic = ks_stat,
        ks_p_value   = ks_p,
        psi          = psi,
    )


# ---------------------------------------------------------------------------
# Analyze all metrics across time windows
# ---------------------------------------------------------------------------

def analyze_drift(logs: list[dict], baseline_days: int = 7, recent_days: int = 1) -> list[dict]:
    now             = datetime.now(timezone.utc)
    recent_cutoff   = now - timedelta(days=recent_days)
    baseline_cutoff = now - timedelta(days=baseline_days)

    baseline_logs = [
        l for l in logs
        if l.get("timestamp") and baseline_cutoff <= _parse_ts(l["timestamp"]) < recent_cutoff
    ]
    recent_logs = [
        l for l in logs
        if l.get("timestamp") and _parse_ts(l["timestamp"]) >= recent_cutoff
    ]

    if not baseline_logs or not recent_logs:
        return []

    metrics_config = [
        ("toxicity_score",      False),
        ("bias_score",          False),
        ("hallucination_score", False),
        ("faithfulness_score",  True),
        ("relevancy_score",     True),
    ]

    results = []
    for metric, higher_is_better in metrics_config:
        baseline_scores = [l[metric] for l in baseline_logs if l.get(metric) is not None]
        recent_scores   = [l[metric] for l in recent_logs   if l.get(metric) is not None]

        if not baseline_scores or not recent_scores:
            continue

        drift = calculate_drift(
            baseline_scores     = baseline_scores,
            recent_scores       = recent_scores,
            metric_name         = metric,
            drift_threshold_pct = 5.0,
            higher_is_better    = higher_is_better,
        )
        results.append({
            "metric":         drift.metric,
            "baseline_avg":   drift.baseline_avg,
            "recent_avg":     drift.recent_avg,
            "drift_pct":      drift.drift_pct,
            "drifted":        drift.drifted,
            "direction":      drift.direction,
            "ks_statistic":   drift.ks_statistic,
            "ks_p_value":     drift.ks_p_value,
            "psi":            drift.psi,
            "psi_status":     "stable" if drift.psi < 0.1 else "slight" if drift.psi < 0.2 else "significant",
            "baseline_days":  baseline_days,
            "recent_days":    recent_days,
            "baseline_count": len(baseline_scores),
            "recent_count":   len(recent_scores),
        })

    return results


def _parse_ts(ts) -> datetime:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(ts)).replace(tzinfo=timezone.utc)