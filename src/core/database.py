import os
from typing import AsyncGenerator
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, Boolean,
    DateTime, JSON, Text, ForeignKey
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, relationship
from dotenv import load_dotenv

load_dotenv(override=False)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:12345@localhost:5432/llm_eval_db"
)

engine = create_async_engine(DATABASE_URL, echo=False, pool_size=10, max_overflow=20)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Users table — JWT auth
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username      = Column(String(100), unique=True, nullable=False, index=True)
    email         = Column(String(200), unique=True, nullable=False, index=True)
    hashed_password = Column(String(200), nullable=False)
    role          = Column(String(50), nullable=False, default="viewer")
    is_active     = Column(Boolean, default=True)
    created_at    = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_login    = Column(DateTime(timezone=True), nullable=True)

    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# API Keys table
# ---------------------------------------------------------------------------
class APIKey(Base):
    __tablename__ = "api_keys"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    key_hash   = Column(String(200), nullable=False, unique=True)
    key_prefix = Column(String(20), nullable=False)   # first 8 chars for display
    role       = Column(String(50), nullable=False, default="viewer")
    is_active  = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="api_keys")


# ---------------------------------------------------------------------------
# Evaluation Logs table
# ---------------------------------------------------------------------------
class EvaluationLog(Base):
    __tablename__ = "evaluation_logs"

    id        = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)

    # Identity — who ran this evaluation
    user_id   = Column(Integer, ForeignKey("users.id"), nullable=True)
    username  = Column(String(100), nullable=True)

    # Input
    query    = Column(String, nullable=False)
    response = Column(String, nullable=False)

    # Safety Metrics (DeepEval)
    toxicity_score      = Column(Float, nullable=True)
    bias_score          = Column(Float, nullable=True)
    hallucination_score = Column(Float, nullable=True)
    safety_passed       = Column(Boolean, nullable=True)

    # RAG Metrics (RAGAS)
    faithfulness_score = Column(Float, nullable=True)
    relevancy_score    = Column(Float, nullable=True)
    rag_passed         = Column(Boolean, nullable=True)

    # Model & Prompt Versioning
    model_name        = Column(String(100), nullable=True)
    model_version     = Column(String(50),  nullable=True)
    prompt_version    = Column(String(50),  nullable=True)
    retriever_version = Column(String(50),  nullable=True)
    embedding_model   = Column(String(100), nullable=True)

    # Cost & Latency
    input_tokens       = Column(Integer, nullable=True)
    output_tokens      = Column(Integer, nullable=True)
    estimated_cost_usd = Column(Float,   nullable=True)
    latency_ms         = Column(Float,   nullable=True)

    # Human Review
    human_label      = Column(String(50),  nullable=True)
    reviewer_id      = Column(String(100), nullable=True)
    reviewer_comment = Column(Text,        nullable=True)
    review_timestamp = Column(DateTime(timezone=True), nullable=True)

    # Trace / sampling
    is_sampled_trace = Column(Boolean, default=False)
    pii_action       = Column(String(20), nullable=True)  # block | mask | redact

    # Raw debug payload
    raw_payload = Column(JSON, nullable=True)


# ---------------------------------------------------------------------------
# Production Traces table
# ---------------------------------------------------------------------------
class ProductionTrace(Base):
    __tablename__ = "production_traces"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    timestamp      = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    query_hash     = Column(String(64),  nullable=False)   # SHA-256 of query
    response_hash  = Column(String(64),  nullable=False)
    query_masked   = Column(Text,        nullable=True)    # PII-masked version
    response_masked= Column(Text,        nullable=True)
    model_name     = Column(String(100), nullable=True)
    model_version  = Column(String(50),  nullable=True)
    prompt_version = Column(String(50),  nullable=True)
    latency_ms     = Column(Float,       nullable=True)
    input_tokens   = Column(Integer,     nullable=True)
    output_tokens  = Column(Integer,     nullable=True)
    eval_log_id    = Column(Integer, ForeignKey("evaluation_logs.id"), nullable=True)
    flagged        = Column(Boolean, default=False)
    flag_reason    = Column(String(200), nullable=True)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise