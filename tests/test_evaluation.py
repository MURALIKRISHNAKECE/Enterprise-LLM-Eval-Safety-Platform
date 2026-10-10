"""
Unit tests for evaluation pipeline components.
Run with: python -m pytest tests/ -v --ignore=tests/load_test.py
"""
import pytest
from src.monitoring.pii_detection import (
    scan_for_pii, pii_summary, apply_pii_policy, process_inputs_with_policy
)
from src.monitoring.drift_detection import (
    analyze_drift, calculate_drift, compute_psi, compute_ks
)
from src.monitoring.latency_tracker import compute_percentiles, analyze_latency


# ---------------------------------------------------------------------------
# PII Detection Tests
# ---------------------------------------------------------------------------

class TestPIIDetection:

    def test_clean_input_passes(self):
        matches = scan_for_pii("What is dehydration?", "Loss of fluids.", [])
        assert matches == []

    def test_detects_email(self):
        matches = scan_for_pii("Contact john@example.com", "OK", [])
        assert any(m.pii_type == "email" for m in matches)

    def test_detects_ssn(self):
        matches = scan_for_pii("My SSN is 123-45-6789", "OK", [])
        assert any(m.pii_type == "ssn" for m in matches)

    def test_detects_credit_card(self):
        matches = scan_for_pii("Card: 4111111111111111", "OK", [])
        assert any(m.pii_type == "credit_card" for m in matches)

    def test_pii_summary_format(self):
        matches = scan_for_pii("Email: test@test.com", "OK", [])
        summary = pii_summary(matches)
        assert summary["pii_detected"] is True
        assert summary["count"] > 0
        assert "findings" in summary

    def test_context_scanned(self):
        matches = scan_for_pii("Query", "Response", ["Call john@example.com"])
        assert any(m.field.startswith("context") for m in matches)

    def test_mask_policy(self):
        result = apply_pii_policy("Email me at test@test.com please", policy="mask")
        assert "test@test.com" not in result
        assert "[EMAIL]" in result

    def test_redact_policy(self):
        result = apply_pii_policy("Email me at test@test.com please", policy="redact")
        assert "test@test.com" not in result

    def test_process_inputs_block(self):
        _, _, _, matches, action = process_inputs_with_policy(
            "My email is test@test.com", "OK", [], policy="block"
        )
        assert action == "block"
        assert len(matches) > 0

    def test_process_inputs_mask(self):
        clean_q, _, _, matches, action = process_inputs_with_policy(
            "Email: test@test.com", "OK", [], policy="mask"
        )
        assert action == "mask"
        assert "[EMAIL]" in clean_q

    def test_process_inputs_clean(self):
        _, _, _, matches, action = process_inputs_with_policy(
            "What is the weather?", "It is sunny.", []
        )
        assert action == "none"
        assert matches == []


# ---------------------------------------------------------------------------
# Drift Detection Tests
# ---------------------------------------------------------------------------

class TestDriftDetection:

    def test_stable_scores_no_drift(self):
        baseline = [0.88, 0.89, 0.90, 0.91, 0.90, 0.89, 0.90, 0.91, 0.89, 0.90]
        recent   = [0.90, 0.91, 0.89, 0.90, 0.91, 0.90, 0.89, 0.91, 0.90, 0.89]
        result = calculate_drift(
            baseline_scores=baseline,
            recent_scores=recent,
            metric_name="faithfulness_score",
            drift_threshold_pct=5.0,
            higher_is_better=True,
        )
        assert result.drifted is False

    def test_detects_faithfulness_drop(self):
        result = calculate_drift(
            baseline_scores=[0.9, 0.9, 0.9],
            recent_scores=[0.5, 0.5, 0.5],
            metric_name="faithfulness_score",
            drift_threshold_pct=5.0,
            higher_is_better=True,
        )
        assert result.drifted is True
        assert result.direction == "degraded"

    def test_detects_toxicity_increase(self):
        result = calculate_drift(
            baseline_scores=[0.01, 0.01, 0.01],
            recent_scores=[0.50, 0.50, 0.50],
            metric_name="toxicity_score",
            drift_threshold_pct=5.0,
            higher_is_better=False,
        )
        assert result.drifted is True
        assert result.direction == "degraded"

    def test_empty_returns_stable(self):
        result = calculate_drift([], [], "test_metric")
        assert result.drifted is False
        assert result.baseline_avg == 0.0

    def test_analyze_drift_no_data(self):
        assert analyze_drift(logs=[], baseline_days=7, recent_days=1) == []

    def test_psi_stable(self):
        baseline = [0.85 + i * 0.001 for i in range(50)]
        recent   = [0.85 + i * 0.001 for i in range(50)]
        psi = compute_psi(baseline, recent)
        assert psi < 0.1

    def test_psi_significant_shift(self):
        psi = compute_psi([0.1] * 20, [0.9] * 20)
        assert psi > 0.2

    def test_ks_identical_distributions(self):
        stat, p = compute_ks([0.9] * 20, [0.9] * 20)
        assert stat == 0.0

    def test_ks_different_distributions(self):
        stat, p = compute_ks([0.1] * 30, [0.9] * 30)
        assert stat > 0.5
        assert p < 0.05


# ---------------------------------------------------------------------------
# Latency Tracker Tests
# ---------------------------------------------------------------------------

class TestLatencyTracker:

    def test_empty_returns_zeros(self):
        result = compute_percentiles([])
        assert result["count"] == 0
        assert result["p50_ms"] == 0.0

    def test_single_value(self):
        result = compute_percentiles([100.0])
        assert result["p50_ms"] == 100.0
        assert result["p95_ms"] == 100.0

    def test_percentiles_ordered(self):
        latencies = [float(i) for i in range(1, 101)]
        result = compute_percentiles(latencies)
        assert result["p50_ms"] <= result["p95_ms"] <= result["p99_ms"]

    def test_per_model_breakdown(self):
        logs = [
            {"latency_ms": 100.0, "model_name": "model-a", "timestamp": None},
            {"latency_ms": 200.0, "model_name": "model-a", "timestamp": None},
            {"latency_ms": 300.0, "model_name": "model-b", "timestamp": None},
        ]
        result = analyze_latency(logs)
        assert "model-a" in result["per_model"]
        assert "model-b" in result["per_model"]
        assert result["per_model"]["model-a"]["count"] == 2

    def test_skips_none_latency(self):
        logs = [
            {"latency_ms": None,  "model_name": "model-a", "timestamp": None},
            {"latency_ms": 100.0, "model_name": "model-a", "timestamp": None},
        ]
        result = analyze_latency(logs)
        assert result["overall"]["count"] == 1