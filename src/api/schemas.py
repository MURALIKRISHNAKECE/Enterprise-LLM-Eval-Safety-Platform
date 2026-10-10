from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class SafetyEvalRequest(BaseModel):
    input_text: str = Field(..., description="User prompt or input text")
    actual_output: str = Field(..., description="LLM response to evaluate")
    context: Optional[List[str]] = Field(default=None, description="Reference context for hallucination check")


class RAGEvalRequest(BaseModel):
    query: str = Field(..., description="User query")
    response: str = Field(..., description="Generated RAG response")
    retrieved_contexts: List[str] = Field(..., description="Chunks fetched from vector search")


class FullEvalRequest(BaseModel):
    query: str = Field(..., description="User query or prompt")
    response: str = Field(..., description="Model generated response")
    retrieved_contexts: Optional[List[str]] = Field(default=[], description="Contexts for RAG evaluation")

    # Optional version metadata
    model_name: Optional[str] = Field(default=None, description="Model identifier e.g. gemini-2.0-flash-lite")
    model_version: Optional[str] = Field(default=None, description="Model version e.g. 1.0.0")
    prompt_version: Optional[str] = Field(default=None, description="Prompt template version e.g. v1")
    retriever_version: Optional[str] = Field(default=None, description="Retriever version e.g. v1")


class HumanFeedbackRequest(BaseModel):
    human_label: str = Field(..., description="Correct | Incorrect | Unsafe | Irrelevant | Partial")
    reviewer_id: Optional[str] = Field(default="anonymous", description="Reviewer identifier")
    reviewer_comment: Optional[str] = Field(default=None, description="Free-text reviewer notes")


class GoldenEvalRequest(BaseModel):
    dataset_path: Optional[str] = Field(
        default="data/golden_dataset/golden_dataset.json",
        description="Path to the golden dataset JSON file"
    )


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------

class EvalResponse(BaseModel):
    status: str
    eval_id: Optional[int] = None
    safety_metrics: Optional[Dict[str, Any]] = None
    rag_metrics: Optional[Dict[str, Any]] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    prompt_version: Optional[str] = None
    latency_ms: Optional[float] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())


class EvalHistoryItem(BaseModel):
    id: int
    timestamp: Optional[datetime]
    query: str
    response: str
    toxicity_score: Optional[float]
    bias_score: Optional[float]
    hallucination_score: Optional[float]
    safety_passed: Optional[bool]
    faithfulness_score: Optional[float]
    relevancy_score: Optional[float]
    rag_passed: Optional[bool]
    model_name: Optional[str]
    model_version: Optional[str]
    prompt_version: Optional[str]
    latency_ms: Optional[float]
    estimated_cost_usd: Optional[float]
    human_label: Optional[str]
    reviewer_id: Optional[str]
    reviewer_comment: Optional[str]

    class Config:
        from_attributes = True