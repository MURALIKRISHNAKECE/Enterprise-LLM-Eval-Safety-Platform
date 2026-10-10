"""
Tests for evaluator modules and alerting thresholds.
Run with: python -m pytest tests/ -v --ignore=tests/load_test.py
"""
import asyncio
from unittest.mock import patch

from src.monitoring.pii_detection import scan_for_pii, pii_summary
from src.monitoring.alerting import check_thresholds


# ---------------------------------------------------------------------------
# PII Tests
# ---------------------------------------------------------------------------

def test_pii_clean_input():
    matches = scan_for_pii("What is machine learning?", "It is a subset of AI.", [])
    assert matches == []


def test_pii_detects_email():
    matches = scan_for_pii("Send results to admin@company.com", "OK", [])
    assert any(m.pii_type == "email" for m in matches)


def test_pii_detects_ssn():
    matches = scan_for_pii("SSN: 123-45-6789", "OK", [])
    assert any(m.pii_type == "ssn" for m in matches)


def test_pii_summary_structure():
    matches = scan_for_pii("Email: test@test.com", "OK", [])
    summary = pii_summary(matches)
    assert "pii_detected" in summary
    assert "count" in summary
    assert "findings" in summary
    assert isinstance(summary["findings"], list)


# ---------------------------------------------------------------------------
# Threshold / Alerting Tests
# ---------------------------------------------------------------------------

def test_no_violations_when_scores_are_good():
    safety = {"toxicity_score": 0.01, "hallucination_score": 0.05}
    rag    = {"faithfulness_score": 0.95, "relevancy_score": 0.90}
    assert check_thresholds(safety, rag) == []


def test_toxicity_violation():
    safety = {"toxicity_score": 0.50, "hallucination_score": 0.05}
    rag    = {"faithfulness_score": 0.95, "relevancy_score": 0.90}
    violations = check_thresholds(safety, rag)
    assert any(v["metric"] == "Toxicity" for v in violations)


def test_faithfulness_violation():
    safety = {"toxicity_score": 0.01, "hallucination_score": 0.05}
    rag    = {"faithfulness_score": 0.50, "relevancy_score": 0.90}
    violations = check_thresholds(safety, rag)
    assert any(v["metric"] == "Faithfulness" for v in violations)


# ---------------------------------------------------------------------------
# Evaluator Tests — patch settings.MOCK_EVALUATORS=True, no deepeval/ragas
# ---------------------------------------------------------------------------

def test_safety_evaluator():
    from src.evaluators import safety_metrics
    with patch.object(safety_metrics.settings, "MOCK_EVALUATORS", True):
        results = asyncio.run(safety_metrics.evaluate_safety(
            input_text="How can I make money fast online?",
            actual_output="You can offer freelance services, write blogs, or do online surveys.",
        ))
    assert "toxicity_score" in results
    assert "bias_score" in results
    assert "hallucination_score" in results
    assert "overall_passed" in results
    assert results["overall_passed"] is True


def test_rag_evaluator():
    from src.evaluators import rag_metrics
    with patch.object(rag_metrics.settings, "MOCK_EVALUATORS", True):
        results = asyncio.run(rag_metrics.evaluate_rag_pipeline(
            query="What is Deloitte's primary focus for LLMs?",
            response="Deloitte focuses on LLM benchmarking, model safety, and AI governance.",
            context=["Deloitte emphasizes LLM benchmarking, safety, evaluation, and AI governance."],
        ))
    assert "faithfulness_score" in results
    assert "relevancy_score" in results
    assert "rag_passed" in results
    assert results["rag_passed"] is True