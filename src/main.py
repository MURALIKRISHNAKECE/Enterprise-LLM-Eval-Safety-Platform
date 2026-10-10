from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router as eval_router
from src.api.auth_routes import router as auth_router
from src.core.database import init_db, AsyncSessionLocal
from src.core.auth import seed_default_admin


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    async with AsyncSessionLocal() as db:
        await seed_default_admin(db)
    yield


app = FastAPI(
    title="Enterprise LLM Evaluation & AI Safety Platform",
    version="3.0.0",
    description=(
        "Production-grade API for safety benchmarking, RAG quality evaluation, "
        "JWT auth, PII policy enforcement, trace replay, statistical drift detection, "
        "rate limiting, and CI/CD evaluation gates."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(eval_router)


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status":  "healthy",
        "service": "llm-eval-safety-platform",
        "version": "3.0.0",
    }