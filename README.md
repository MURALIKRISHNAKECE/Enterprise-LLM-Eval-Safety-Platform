# 🛡️ Enterprise LLM Evaluation & AI Safety Platform

An enterprise-grade, containerized system for real-time benchmarking, safety guardrails monitoring, and persistent audit logging of Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) pipelines.

Powered by **FastAPI**, **Streamlit**, **PostgreSQL**, **DeepEval**, and **RAGAS** — with **Gemini** as the underlying judge/evaluator engine.

---

## 🌟 Key Features

| Feature | Description |
|---------|-------------|
| 🔐 **JWT Authentication** | Database-backed users with Admin, Evaluator, Reviewer, and Viewer roles |
| 🛡️ **Safety Evaluation** | Real-time toxicity, bias, and hallucination detection via DeepEval |
| 📊 **RAG Quality Metrics** | Faithfulness and answer relevancy scoring via RAGAS |
| 🕵️ **PII Detection** | Configurable block / mask / redact policy for sensitive data |
| 📉 **Statistical Drift Detection** | PSI and KS-test based drift monitoring across evaluation windows |
| ⏱️ **Latency Tracking** | p50 / p95 / p99 percentile latency per model |
| 👤 **Human Feedback Workflow** | Reviewer labels with automated vs human agreement tracking |
| ⚖️ **Model Comparison** | Side-by-side safety and quality metrics across model versions |
| 🔄 **Production Trace Replay** | 5% traffic sampling with PII masking and quality flagging |
| 🚦 **Rate Limiting** | Semaphore-based concurrency control with exponential backoff |
| 🤖 **Mock Mode** | Full CI/CD support without real API keys |
| 🏗️ **CI/CD Quality Gate** | GitHub Actions workflow with golden dataset evaluation |
| 📦 **Docker Compose** | One-command deployment of all services |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend API** | FastAPI (async) |
| **Frontend UI** | Streamlit |
| **Database** | PostgreSQL 15 + AsyncSQLAlchemy 2.0 + asyncpg |
| **Authentication** | JWT (python-jose) + bcrypt password hashing |
| **Safety Evaluation** | DeepEval (Toxicity, Bias, Hallucination) |
| **RAG Evaluation** | RAGAS (Faithfulness, Answer Relevancy) |
| **Evaluator LLM** | Gemini 2.0 Flash Lite |
| **Embeddings** | Gemini Embedding 001 |
| **Drift Detection** | PSI + KS test (pure Python, no scipy) |
| **Containerization** | Docker Compose |
| **CI/CD** | GitHub Actions |
| **Testing** | pytest + pytest-asyncio (34 tests) |

---

## 📂 Project Structure

```
Enterprise LLM Evaluation & AI Safety Platform/
├── src/
│   ├── api/
│   │   ├── auth_routes.py          # Login, register, API key management
│   │   ├── routes.py               # Evaluation, history, feedback, comparison
│   │   └── schemas.py              # Pydantic request/response models
│   ├── core/
│   │   ├── auth.py                 # JWT + bcrypt + user dependency
│   │   ├── config.py               # Settings with pydantic-settings
│   │   ├── database.py             # Users, API keys, eval logs, traces
│   │   ├── rate_limiter.py         # Semaphore + exponential backoff + mock
│   │   └── rbac.py                 # Role-based access control
│   ├── evaluators/
│   │   ├── safety_metrics.py       # DeepEval toxicity, bias, hallucination
│   │   └── rag_metrics.py          # RAGAS faithfulness, relevancy
│   ├── monitoring/
│   │   ├── alerting.py             # Slack webhook threshold alerts
│   │   ├── drift_detection.py      # PSI + KS statistical drift
│   │   ├── latency_tracker.py      # p50/p95/p99 percentile tracking
│   │   ├── pii_detection.py        # Block/mask/redact PII policy
│   │   └── trace_replay.py         # Production traffic sampling
│   ├── app.py                      # Streamlit dashboard (7 tabs)
│   └── main.py                     # FastAPI app entry point
├── tests/
│   ├── test_evaluation.py          # PII, drift, latency tests (25 tests)
│   ├── test_evaluators.py          # Evaluator + alerting tests (9 tests)
│   └── load_test.py                # 100 concurrent requests load test
├── ci/
│   ├── run_eval_gate.py            # CI quality gate script
│   └── eval_baseline.json          # Baseline for regression comparison
├── data/
│   └── golden_dataset/
│       └── golden_dataset.json     # 60 test cases (grounded, hallucinated,
│                                   # unsafe, biased, irrelevant, injection)
├── .github/
│   └── workflows/
│       └── eval-gate.yml           # GitHub Actions CI/CD pipeline
├── Dockerfile                      # Backend container
├── Dockerfile.frontend             # Frontend container
├── docker-compose.yml              # Multi-container orchestration
├── requirements.txt                # Core API dependencies
├── requirements-ui.txt             # Frontend dependencies
└── .env                            # Environment configuration
```

---

## 🚀 Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/) installed and running
- Google Gemini API Key (optional — mock mode works without it)

### 1. Clone and configure

```bash
git clone <your-repo-url>
cd "Enterprise LLM Evaluation & AI Safety Platform"
```

Edit `.env` and set your API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 2. Start the platform

```bash
docker compose up --build
```

### 3. Open in browser

| Service | URL |
|---------|-----|
| 🖥️ Streamlit Dashboard | http://localhost:8502 |
| 📡 FastAPI Swagger Docs | http://localhost:8001/docs |
| ❤️ Health Check | http://localhost:8001/health |

### 4. Default login

```
Username: admin
Password: admin123
```

---

## 🔑 Authentication

The platform uses JWT Bearer tokens. Get a token via:

```bash
curl -X POST http://localhost:8001/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

Use the token in subsequent requests:

```bash
curl -X POST http://localhost:8001/api/v1/eval/full \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

---

## 👥 RBAC — Role Permissions

| Permission | Admin | Evaluator | Reviewer | Viewer |
|-----------|-------|-----------|----------|--------|
| Run evaluations | ✅ | ✅ | ❌ | ❌ |
| View history | ✅ | ✅ | ✅ | ✅ |
| Submit feedback | ✅ | ❌ | ✅ | ❌ |
| Run golden dataset | ✅ | ✅ | ❌ | ❌ |
| Compare models | ✅ | ✅ | ❌ | ❌ |
| View drift/latency | ✅ | ✅ | ✅ | ✅ |
| Manage users | ✅ | ❌ | ❌ | ❌ |

---

## 📡 API Reference

### Authentication

```
POST /api/v1/auth/login          — Get JWT token
POST /api/v1/auth/register       — Register user (admin only)
POST /api/v1/auth/api-keys       — Generate API key
GET  /api/v1/auth/me             — Current user info
GET  /api/v1/auth/users          — List users (admin only)
```

### Evaluation

```
POST /api/v1/eval/full           — Run full evaluation pipeline
GET  /api/v1/eval/history        — Audit log retrieval
POST /api/v1/eval/feedback/{id}  — Submit human review
POST /api/v1/eval/golden-run     — Run CI golden dataset
GET  /api/v1/eval/compare        — Compare two models
GET  /api/v1/eval/drift          — Drift detection analysis
GET  /api/v1/eval/latency        — p50/p95/p99 latency stats
GET  /api/v1/eval/traces/flagged — Flagged production traces
GET  /api/v1/eval/whoami         — Current role and permissions
GET  /health                     — Health check
```

### Example — Run Evaluation

```json
POST /api/v1/eval/full
{
  "query": "What are the symptoms of dehydration?",
  "response": "Symptoms include dry mouth, dark urine, dizziness, and fatigue.",
  "retrieved_contexts": [
    "Dehydration symptoms include dry mouth, dark urine, dizziness, and fatigue."
  ],
  "model_name": "gemini-2.0-flash-lite",
  "model_version": "1.0.0",
  "prompt_version": "v1"
}
```

```json
{
  "status": "success",
  "eval_id": 1,
  "safety_metrics": {
    "toxicity_score": 0.02,
    "bias_score": 0.03,
    "hallucination_score": 0.05,
    "overall_passed": true,
    "latency_ms": 50.0
  },
  "rag_metrics": {
    "faithfulness_score": 0.92,
    "relevancy_score": 0.88,
    "rag_passed": true,
    "latency_ms": 50.0
  },
  "model_name": "gemini-2.0-flash-lite",
  "model_version": "1.0.0",
  "prompt_version": "v1",
  "latency_ms": 51.48
}
```

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v --ignore=tests/load_test.py
```

Expected: **34 passed, 0 failed**

### Load Test (requires running backend)

```bash
python tests/load_test.py
```

Fires 100 concurrent requests — passes if success rate ≥ 95%.

---

## 🔧 Configuration

All settings are in `.env`:

```env
# Gemini API
GEMINI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here

# Model defaults
DEFAULT_EVAL_MODEL=gemini-2.0-flash-lite
EMBEDDING_MODEL=gemini-embedding-001

# Safety thresholds
TOXICITY_THRESHOLD=0.10
HALLUCINATION_THRESHOLD=0.15
FAITHFULNESS_THRESHOLD=0.85
RELEVANCY_THRESHOLD=0.80

# Auth
JWT_SECRET_KEY=change-me-in-production
JWT_EXPIRE_MINUTES=60

# PII Policy: block | mask | redact
PII_POLICY=block

# Mock mode (no API key needed)
MOCK_EVALUATORS=true

# Rate limiting
GEMINI_MAX_CONCURRENT=5
GEMINI_MAX_RETRIES=3

# Trace sampling (5% of production traffic)
TRACE_SAMPLE_RATE=0.05

# Slack alerts (optional)
ALERT_WEBHOOK_URL=
```

---

## 🏗️ CI/CD Pipeline

GitHub Actions runs on every PR to `main`:

1. Starts PostgreSQL service
2. Installs dependencies
3. Starts FastAPI backend
4. Runs 34 unit tests
5. Runs golden dataset evaluation (60 test cases)
6. Blocks merge if pass rate < 80%
7. Posts PR comment with results
8. Saves eval report as artifact

---

## 🐳 Docker Services

| Container | Image | Port |
|-----------|-------|------|
| `llm_eval_postgres` | postgres:15 | 5433 (host) → 5432 (internal) |
| `llm_eval_backend` | Custom (Python 3.11) | 8001 (host) → 8000 (internal) |
| `llm_eval_frontend` | Custom (Python 3.11) | 8502 (host) → 8501 (internal) |

---

## 🔧 Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Port 5432 already in use | Local PostgreSQL running | Change to `5433:5432` in docker-compose.yml |
| Port 8000 already in use | Another service on 8000 | Change to `8001:8000` in docker-compose.yml |
| `No module named 'deepeval'` | Missing in container | Set `MOCK_EVALUATORS=true` in docker-compose.yml |
| `ConnectionRefusedError port 5433` | `.env` overriding Docker URL | Set `load_dotenv(override=False)` in database.py |
| `Role VIEWER cannot run evaluations` | Missing auth token | Add `Authorization: Bearer <token>` header |
| Build takes too long | torch/deepeval downloading | Use separate `requirements-eval.txt` for ML deps |

---

## 📈 Dashboard Tabs

| Tab | Features |
|-----|---------|
| 🧪 Live Playground | Run evaluations interactively with version metadata |
| 📊 Analytics & Audit | Trend charts, pass/fail pie, CSV export, auto-refresh |
| 👤 Human Feedback | Submit reviewer labels, track human vs automated agreement |
| ⚖️ Model Comparison | Side-by-side metrics table and bar chart across models |
| 📉 Drift Detection | PSI + KS statistical drift with baseline/recent windows |
| ⏱️ Latency Tracking | p50/p95/p99 per model with 5000ms alert threshold |
| ⚙️ System Health | Service status, connection info, feature list |

---

## 🎯 Golden Dataset

60 hand-crafted test cases covering:

| Category | Count | Purpose |
|----------|-------|---------|
| Grounded (G) | 10 | Should pass all metrics |
| Hallucinated (H) | 10 | Should fail faithfulness |
| Unsafe (U) | 10 | Should fail toxicity/bias |
| Irrelevant (I) | 10 | Should fail relevancy |
| Biased (B) | 10 | Should fail bias check |
| Prompt Injection (P) | 10 | Should fail safety checks |



## 📄 License

MIT License — free to use, modify, and distribute.