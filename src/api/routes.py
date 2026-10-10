"""
FastAPI Routes — full evaluation pipeline with JWT auth, PII policy,
trace replay, model comparison, drift detection, latency tracking.
"""
import asyncio
import time
import json
import pathlib
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.api.schemas import (
    FullEvalRequest, EvalResponse, EvalHistoryItem,
    HumanFeedbackRequest, GoldenEvalRequest,
)
from src.core.database import get_db, EvaluationLog, User
from src.core.config import settings
from src.core.rbac import Role, check_permission, PERMISSIONS
from src.core.auth import get_current_user
from src.evaluators.safety_metrics import evaluate_safety
from src.evaluators.rag_metrics import evaluate_rag_pipeline
from src.monitoring.alerting import check_thresholds, send_alert
from src.monitoring.pii_detection import process_inputs_with_policy, pii_summary
from src.monitoring.drift_detection import analyze_drift
from src.monitoring.latency_tracker import analyze_latency
from src.monitoring.trace_replay import store_trace, get_flagged_traces

router = APIRouter(prefix="/api/v1/eval", tags=["Evaluation"])


def _role(user: User) -> Role:
    try:
        return Role(user.role)
    except ValueError:
        return Role.VIEWER


# ---------------------------------------------------------------------------
# POST /full
# ---------------------------------------------------------------------------
@router.post("/full", response_model=EvalResponse)
async def run_full_eval(
    payload:      FullEvalRequest,
    db:           AsyncSession = Depends(get_db),
    current_user: User         = Depends(get_current_user),
):
    role = _role(current_user)
    if not check_permission(role, "run_evaluation"):
        raise HTTPException(status_code=403, detail=f"Role '{role}' cannot run evaluations.")

    wall_start = time.monotonic()

    # PII policy enforcement
    clean_query, clean_response, clean_contexts, pii_matches, pii_action = \
        process_inputs_with_policy(
            payload.query,
            payload.response,
            payload.retrieved_contexts or [],
        )

    if pii_action == "block" and pii_matches:
        raise HTTPException(
            status_code=422,
            detail={"error": "PII detected — request blocked.", **pii_summary(pii_matches)},
        )

    try:
        safety_task = evaluate_safety(
            input_text=clean_query,
            actual_output=clean_response,
            context=clean_contexts,
        )
        rag_task = evaluate_rag_pipeline(
            query=clean_query,
            response=clean_response,
            context=clean_contexts,
        )
        safety_results, rag_results = await asyncio.gather(safety_task, rag_task)

        total_latency_ms = (time.monotonic() - wall_start) * 1000

        model_name     = payload.model_name       or settings.DEFAULT_EVAL_MODEL
        model_version  = payload.model_version    or settings.MODEL_VERSION
        prompt_version = payload.prompt_version   or settings.PROMPT_VERSION
        retriever_ver  = payload.retriever_version or settings.RETRIEVER_VERSION

        log = EvaluationLog(
            query               = clean_query,
            response            = clean_response,
            user_id             = current_user.id,
            username            = current_user.username,
            toxicity_score      = safety_results.get("toxicity_score"),
            bias_score          = safety_results.get("bias_score"),
            hallucination_score = safety_results.get("hallucination_score"),
            safety_passed       = safety_results.get("overall_passed"),
            faithfulness_score  = rag_results.get("faithfulness_score"),
            relevancy_score     = rag_results.get("relevancy_score"),
            rag_passed          = rag_results.get("rag_passed"),
            model_name          = model_name,
            model_version       = model_version,
            prompt_version      = prompt_version,
            retriever_version   = retriever_ver,
            embedding_model     = settings.EMBEDDING_MODEL,
            latency_ms          = round(total_latency_ms, 2),
            pii_action          = pii_action if pii_action != "none" else None,
            raw_payload         = {"safety": safety_results, "rag": rag_results},
        )
        db.add(log)
        await db.commit()
        await db.refresh(log)

        # Non-blocking: alerting + trace sampling
        violations = check_thresholds(safety_results, rag_results)
        if violations:
            asyncio.create_task(send_alert(violations))

        asyncio.create_task(store_trace(db, clean_query, clean_response, log))

        return EvalResponse(
            status         = "success",
            eval_id        = log.id,
            safety_metrics = safety_results,
            rag_metrics    = rag_results,
            model_name     = model_name,
            model_version  = model_version,
            prompt_version = prompt_version,
            latency_ms     = round(total_latency_ms, 2),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


# ---------------------------------------------------------------------------
# GET /history
# ---------------------------------------------------------------------------
@router.get("/history", response_model=list[EvalHistoryItem])
async def get_eval_history(
    limit:        int          = 50,
    db:           AsyncSession = Depends(get_db),
    current_user: User         = Depends(get_current_user),
):
    role = _role(current_user)
    if not check_permission(role, "view_history"):
        raise HTTPException(status_code=403, detail=f"Role '{role}' cannot view history.")

    result = await db.execute(
        select(EvaluationLog).order_by(EvaluationLog.timestamp.desc()).limit(limit)
    )
    return result.scalars().all()


# ---------------------------------------------------------------------------
# POST /feedback/{eval_id}
# ---------------------------------------------------------------------------
@router.post("/feedback/{eval_id}")
async def submit_human_feedback(
    eval_id:      int,
    feedback:     HumanFeedbackRequest,
    db:           AsyncSession = Depends(get_db),
    current_user: User         = Depends(get_current_user),
):
    role = _role(current_user)
    if not check_permission(role, "submit_feedback"):
        raise HTTPException(status_code=403, detail=f"Role '{role}' cannot submit feedback.")

    result = await db.execute(select(EvaluationLog).where(EvaluationLog.id == eval_id))
    log = result.scalar_one_or_none()
    if not log:
        raise HTTPException(status_code=404, detail=f"Evaluation ID {eval_id} not found.")

    valid_labels = {"Correct", "Incorrect", "Unsafe", "Irrelevant", "Partial"}
    if feedback.human_label not in valid_labels:
        raise HTTPException(status_code=422, detail=f"Label must be one of: {valid_labels}")

    log.human_label      = feedback.human_label
    log.reviewer_id      = feedback.reviewer_id or current_user.username
    log.reviewer_comment = feedback.reviewer_comment
    log.review_timestamp = datetime.now(timezone.utc)
    await db.commit()

    auto_passed = log.safety_passed and log.rag_passed
    human_pass  = feedback.human_label in {"Correct", "Partial"}

    return {
        "status":                 "feedback_recorded",
        "eval_id":                eval_id,
        "human_label":            feedback.human_label,
        "automated_passed":       auto_passed,
        "agreement":              (auto_passed == human_pass),
        "automated_faithfulness": log.faithfulness_score,
    }


# ---------------------------------------------------------------------------
# POST /golden-run
# ---------------------------------------------------------------------------
@router.post("/golden-run")
async def run_golden_dataset(
    payload:      GoldenEvalRequest,
    db:           AsyncSession = Depends(get_db),
    current_user: User         = Depends(get_current_user),
):
    role = _role(current_user)
    if not check_permission(role, "run_golden"):
        raise HTTPException(status_code=403, detail=f"Role '{role}' cannot run golden dataset.")

    dataset_path = pathlib.Path(payload.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Golden dataset not found: {dataset_path}")

    with open(dataset_path) as f:
        cases = json.load(f)

    results = []
    passed = failed = 0

    for case in cases:
        try:
            safety = await evaluate_safety(
                input_text=case["question"],
                actual_output=case["answer"],
                context=case.get("contexts", [case.get("context", "")]),
            )
            rag = await evaluate_rag_pipeline(
                query=case["question"],
                response=case["answer"],
                context=case.get("contexts", [case.get("context", "")]),
            )
            overall = safety["overall_passed"] and rag["rag_passed"]
            passed += 1 if overall else 0
            failed += 0 if overall else 1
            results.append({
                "id": case.get("id"), "safety": safety, "rag": rag, "passed": overall,
            })
        except Exception as e:
            failed += 1
            results.append({"id": case.get("id"), "error": str(e), "passed": False})

    total     = len(cases)
    pass_rate = passed / total if total else 0

    return {
        "total":     total,
        "passed":    passed,
        "failed":    failed,
        "pass_rate": round(pass_rate, 4),
        "ci_passed": pass_rate >= 0.80,
        "results":   results,
    }


# ---------------------------------------------------------------------------
# GET /compare
# ---------------------------------------------------------------------------
@router.get("/compare")
async def compare_versions(
    model_a:      str,
    model_b:      str,
    db:           AsyncSession = Depends(get_db),
    current_user: User         = Depends(get_current_user),
):
    role = _role(current_user)
    if not check_permission(role, "compare_models"):
        raise HTTPException(status_code=403, detail=f"Role '{role}' cannot compare models.")

    async def _avg(model: str):
        result = await db.execute(
            select(EvaluationLog).where(EvaluationLog.model_name == model)
        )
        logs = result.scalars().all()
        if not logs:
            return None
        n = len(logs)
        return {
            "model":             model,
            "count":             n,
            "avg_toxicity":      round(sum(l.toxicity_score      or 0 for l in logs) / n, 4),
            "avg_bias":          round(sum(l.bias_score          or 0 for l in logs) / n, 4),
            "avg_hallucination": round(sum(l.hallucination_score or 0 for l in logs) / n, 4),
            "avg_faithfulness":  round(sum(l.faithfulness_score  or 0 for l in logs) / n, 4),
            "avg_relevancy":     round(sum(l.relevancy_score     or 0 for l in logs) / n, 4),
            "avg_latency_ms":    round(sum(l.latency_ms          or 0 for l in logs) / n, 2),
            "safety_pass_rate":  round(sum(1 for l in logs if l.safety_passed) / n, 4),
        }

    a_stats, b_stats = await asyncio.gather(_avg(model_a), _avg(model_b))
    return {
        "model_a": a_stats or {"error": f"No data for '{model_a}'"},
        "model_b": b_stats or {"error": f"No data for '{model_b}'"},
    }


# ---------------------------------------------------------------------------
# GET /drift
# ---------------------------------------------------------------------------
@router.get("/drift")
async def get_drift_analysis(
    baseline_days: int          = 7,
    recent_days:   int          = 1,
    db:            AsyncSession = Depends(get_db),
    current_user:  User         = Depends(get_current_user),
):
    role = _role(current_user)
    if not check_permission(role, "view_drift"):
        raise HTTPException(status_code=403, detail=f"Role '{role}' cannot view drift.")

    result = await db.execute(
        select(EvaluationLog).order_by(EvaluationLog.timestamp.desc()).limit(500)
    )
    logs = result.scalars().all()

    logs_as_dicts = [
        {
            "timestamp":           l.timestamp,
            "toxicity_score":      l.toxicity_score,
            "bias_score":          l.bias_score,
            "hallucination_score": l.hallucination_score,
            "faithfulness_score":  l.faithfulness_score,
            "relevancy_score":     l.relevancy_score,
        }
        for l in logs
    ]

    drift_results = analyze_drift(logs_as_dicts, baseline_days, recent_days)
    drifted_count = sum(1 for r in drift_results if r["drifted"])

    return {
        "baseline_days":  baseline_days,
        "recent_days":    recent_days,
        "total_metrics":  len(drift_results),
        "drifted_count":  drifted_count,
        "drift_detected": drifted_count > 0,
        "results":        drift_results,
    }


# ---------------------------------------------------------------------------
# GET /latency
# ---------------------------------------------------------------------------
@router.get("/latency")
async def get_latency_stats(
    limit:        int          = 500,
    db:           AsyncSession = Depends(get_db),
    current_user: User         = Depends(get_current_user),
):
    role = _role(current_user)
    if not check_permission(role, "view_latency"):
        raise HTTPException(status_code=403, detail=f"Role '{role}' cannot view latency.")

    result = await db.execute(
        select(EvaluationLog).order_by(EvaluationLog.timestamp.desc()).limit(limit)
    )
    logs = result.scalars().all()

    logs_as_dicts = [
        {"latency_ms": l.latency_ms, "model_name": l.model_name, "timestamp": l.timestamp}
        for l in logs
    ]

    stats     = analyze_latency(logs_as_dicts)
    p95_alert = stats["overall"]["p95_ms"] > 5000

    return {
        "p95_alert": p95_alert,
        "alert_msg": "p95 latency exceeds 5000ms" if p95_alert else "Latency within threshold",
        **stats,
    }


# ---------------------------------------------------------------------------
# GET /traces/flagged
# ---------------------------------------------------------------------------
@router.get("/traces/flagged")
async def get_flagged_traces_endpoint(
    limit:        int          = 50,
    db:           AsyncSession = Depends(get_db),
    current_user: User         = Depends(get_current_user),
):
    role = _role(current_user)
    if not check_permission(role, "view_history"):
        raise HTTPException(status_code=403, detail="Insufficient permissions.")

    return await get_flagged_traces(db, limit)


# ---------------------------------------------------------------------------
# GET /whoami
# ---------------------------------------------------------------------------
@router.get("/whoami")
async def whoami(current_user: User = Depends(get_current_user)):
    role = _role(current_user)
    return {
        "username":    current_user.username,
        "role":        role,
        "permissions": sorted(PERMISSIONS.get(role, set())),
    }