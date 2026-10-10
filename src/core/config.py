import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "LLM Evaluation & AI Safety Platform"

    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    # Model & Version Tracking
    DEFAULT_EVAL_MODEL: str = os.getenv("DEFAULT_EVAL_MODEL", "gemini-2.0-flash-lite")
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "1.0.0")
    PROMPT_VERSION: str = os.getenv("PROMPT_VERSION", "v1")
    RETRIEVER_VERSION: str = os.getenv("RETRIEVER_VERSION", "v1")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

    # Evaluation Thresholds
    TOXICITY_THRESHOLD: float = float(os.getenv("TOXICITY_THRESHOLD", "0.10"))
    HALLUCINATION_THRESHOLD: float = float(os.getenv("HALLUCINATION_THRESHOLD", "0.15"))
    FAITHFULNESS_THRESHOLD: float = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.85"))
    RELEVANCY_THRESHOLD: float = float(os.getenv("RELEVANCY_THRESHOLD", "0.80"))

    # Alerting
    ALERT_WEBHOOK_URL: str = os.getenv("ALERT_WEBHOOK_URL", "")

    # JWT Auth
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "change-me-in-production-use-32-chars")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

    # Rate Limiting
    GEMINI_MAX_CONCURRENT: int = int(os.getenv("GEMINI_MAX_CONCURRENT", "5"))
    GEMINI_MAX_RETRIES: int = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
    MOCK_EVALUATORS: bool = os.getenv("MOCK_EVALUATORS", "false").lower() == "true"

    # PII Policy: block | mask | redact
    PII_POLICY: str = os.getenv("PII_POLICY", "block")

    # Production Trace Replay
    TRACE_SAMPLE_RATE: float = float(os.getenv("TRACE_SAMPLE_RATE", "0.05"))

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()