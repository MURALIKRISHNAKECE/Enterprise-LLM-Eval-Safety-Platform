import os
import time
import requests
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise LLM Evaluation & AI Safety Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL     = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api/v1/eval").rstrip("/")
HEALTH_CHECK_URL = API_BASE_URL.replace("/api/v1/eval", "") + "/health"
REQUEST_TIMEOUT  = 300


def get_admin_token() -> str:
    try:
        base = API_BASE_URL.replace("/api/v1/eval", "")
        res = requests.post(
            f"{base}/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"},
            timeout=10,
        )
        if res.status_code == 200:
            return res.json().get("access_token", "")
    except Exception:
        pass
    return ""


AUTH_TOKEN   = get_admin_token()
AUTH_HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}"} if AUTH_TOKEN else {}

st.markdown("""
<style>
.main-header{font-size:2.2rem;font-weight:700;color:#1E293B;margin-bottom:.2rem}
.sub-header{font-size:1.0rem;color:#64748B;margin-bottom:1.5rem}
.pass-badge{background:#DCFCE7;color:#166534;font-weight:600;padding:4px 12px;border-radius:12px}
.fail-badge{background:#FEE2E2;color:#991B1B;font-weight:600;padding:4px 12px;border-radius:12px}
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🛡️ AI Safety Platform")
st.sidebar.markdown("**Engine**: Gemini 2.0 Flash Lite")
st.sidebar.markdown("**Embeddings**: Gemini Embedding 001")
st.sidebar.markdown("---")

backend_online = False
try:
    h = requests.get(HEALTH_CHECK_URL, timeout=5)
    if h.status_code == 200:
        backend_online = True
        st.sidebar.success("✅ Backend API: Online")
    else:
        st.sidebar.error(f"Backend API Error: HTTP {h.status_code}")
except Exception:
    st.sidebar.error("❌ Backend API: Offline")

st.sidebar.markdown("---")
st.sidebar.info(
    "Real-time toxicity, bias, hallucination, and RAG metric evaluation "
    "with human feedback, model comparison, drift detection, latency tracking, and PII detection."
)

if AUTH_TOKEN:
    st.sidebar.success("🔐 Authenticated as admin")
else:
    st.sidebar.warning("⚠️ Not authenticated")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">Enterprise LLM Evaluation & Safety Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Safety guardrails · RAG benchmarking · Human review · Model comparison · Drift detection · Latency tracking · Audit logs</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "🧪 Live Playground",
    "📊 Analytics & Audit",
    "👤 Human Feedback",
    "⚖️ Model Comparison",
    "📉 Drift Detection",
    "⏱️ Latency Tracking",
    "⚙️ System Health",
])

# =============================================================================
# TAB 1 - Live Playground
# =============================================================================
with tabs[0]:
    st.subheader("Run Interactive Evaluation")
    st.write("Submit a prompt, response, and retrieved contexts to run DeepEval + RAGAS concurrently.")

    col_l, col_r = st.columns(2)

    with col_l:
        query_input = st.text_area("User Query / Prompt", height=100,
            value="What are the symptoms of acute dehydration and how should it be treated?")
        response_input = st.text_area("LLM Output / Candidate Response", height=140,
            value="Symptoms include extreme thirst, dry mouth, dizziness, and confusion. Severe cases require IV fluid replacement.")
        context_input = st.text_area("Retrieved Contexts (one per line)", height=120,
            value="Dehydration occurs when your body loses more fluid than you take in.\nSymptoms include thirst, dark urine, dizziness, and fatigue. Severe dehydration requires IV treatment.")

        with st.expander("⚙️ Version Metadata (optional)"):
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                model_name = st.text_input("Model Name",     value="gemini-2.0-flash-lite")
                prompt_ver = st.text_input("Prompt Version", value="v1")
            with col_v2:
                model_ver     = st.text_input("Model Version",    value="1.0.0")
                retriever_ver = st.text_input("Retriever Version", value="v1")

        retrieved_contexts = [l.strip() for l in context_input.split("\n") if l.strip()]
        run_btn = st.button("🚀 Run Full Evaluation Pipeline", type="primary", use_container_width=True)

    with col_r:
        st.subheader("Evaluation Results")

        if run_btn:
            if not query_input.strip() or not response_input.strip():
                st.warning("Please provide both a Query and a Response.")
            elif not backend_online:
                st.error("Backend API is offline.")
            else:
                payload = {
                    "query":              query_input,
                    "response":           response_input,
                    "retrieved_contexts": retrieved_contexts,
                    "model_name":         model_name,
                    "model_version":      model_ver,
                    "prompt_version":     prompt_ver,
                    "retriever_version":  retriever_ver,
                }
                with st.spinner("Running DeepEval (Safety) + RAGAS (Quality) via Gemini..."):
                    try:
                        res = requests.post(
                            f"{API_BASE_URL}/full",
                            json=payload,
                            headers=AUTH_HEADERS,
                            timeout=REQUEST_TIMEOUT,
                        )

                        if res.status_code == 422:
                            err = res.json()
                            st.error("🚨 **PII Detected - evaluation blocked.**")
                            st.json(err)
                        elif res.status_code == 200:
                            data   = res.json()
                            safety = data.get("safety_metrics", {})
                            rag    = data.get("rag_metrics", {})

                            st.success(
                                f"✅ Evaluation complete · "
                                f"ID: `{data.get('eval_id')}` · "
                                f"Latency: `{data.get('latency_ms', 0):.0f}ms`"
                            )

                            m1, m2, m3 = st.columns(3)
                            tox  = safety.get("toxicity_score", 0.0)
                            bias = safety.get("bias_score", 0.0)
                            hal  = safety.get("hallucination_score", 0.0)
                            m1.metric("Toxicity",      f"{tox:.3f}",  delta="Pass" if tox  < 0.10 else "Fail", delta_color="inverse")
                            m2.metric("Bias",          f"{bias:.3f}", delta="Pass" if bias < 0.50 else "Fail", delta_color="inverse")
                            m3.metric("Hallucination", f"{hal:.3f}",  delta="Pass" if hal  < 0.15 else "Fail", delta_color="inverse")

                            st.divider()
                            m4, m5 = st.columns(2)
                            faith = rag.get("faithfulness_score", 0.0)
                            rel   = rag.get("relevancy_score", 0.0)
                            m4.metric("Faithfulness",     f"{faith:.3f}", delta="Pass" if faith >= 0.85 else "Fail")
                            m5.metric("Answer Relevancy", f"{rel:.3f}",   delta="Pass" if rel   >= 0.80 else "Fail")

                            with st.expander("📋 Version Info"):
                                st.json({
                                    "model_name":     data.get("model_name"),
                                    "model_version":  data.get("model_version"),
                                    "prompt_version": data.get("prompt_version"),
                                })

                            with st.expander("🔍 Raw Payload"):
                                st.json(data)
                        else:
                            st.error(f"HTTP {res.status_code}: {res.text}")

                    except requests.exceptions.Timeout:
                        st.error(f"Request timed out after {REQUEST_TIMEOUT}s.")
                    except Exception as e:
                        st.error(f"Connection error: {e}")


# =============================================================================
# TAB 2 - Analytics & Audit History
# =============================================================================
with tabs[1]:
    st.subheader("Historical Evaluation Analytics")

    auto_refresh = st.toggle("🔄 Auto-refresh every 30s", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()

    if st.button("🔄 Refresh Now"):
        st.rerun()

    if not backend_online:
        st.error("Backend offline - cannot load audit logs.")
    else:
        try:
            hist = requests.get(
                f"{API_BASE_URL}/history?limit=200",
                headers=AUTH_HEADERS,
                timeout=15,
            )
            if hist.status_code == 200:
                logs = hist.json()
                if not logs:
                    st.info("No evaluation logs yet. Run evaluations in the Playground tab.")
                else:
                    df = pd.DataFrame(logs)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])

                    total       = len(df)
                    pass_rate   = (df["safety_passed"].sum() / total * 100) if "safety_passed" in df else 0
                    avg_tox     = df["toxicity_score"].mean()     if "toxicity_score"     in df else 0
                    avg_faith   = df["faithfulness_score"].mean() if "faithfulness_score" in df else 0
                    avg_latency = df["latency_ms"].mean()         if "latency_ms"         in df else 0

                    k1, k2, k3, k4, k5 = st.columns(5)
                    k1.metric("Total Evals",      total)
                    k2.metric("Safety Pass Rate", f"{pass_rate:.1f}%")
                    k3.metric("Avg Toxicity",     f"{avg_tox:.3f}")
                    k4.metric("Avg Faithfulness", f"{avg_faith:.3f}")
                    k5.metric("Avg Latency",      f"{avg_latency:.0f}ms")

                    st.divider()

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("#### Metric Trends Over Time")
                        fig = px.line(
                            df, x="timestamp",
                            y=["toxicity_score", "bias_score", "faithfulness_score", "relevancy_score"],
                            markers=True,
                            labels={"value": "Score", "variable": "Metric"},
                        )
                        fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig, use_container_width=True)

                    with c2:
                        st.markdown("#### Safety Pass / Fail")
                        pc = df["safety_passed"].value_counts().reset_index()
                        pc.columns = ["Status", "Count"]
                        pc["Status"] = pc["Status"].map({True: "Passed", False: "Failed"})
                        fig2 = px.pie(
                            pc, names="Status", values="Count",
                            color="Status",
                            color_discrete_map={"Passed": "#22C55E", "Failed": "#EF4444"},
                        )
                        fig2.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig2, use_container_width=True)

                    st.divider()

                    if "human_label" in df and df["human_label"].notna().any():
                        st.markdown("#### Human Review Labels")
                        hl = df["human_label"].value_counts().reset_index()
                        hl.columns = ["Label", "Count"]
                        fig3 = px.bar(
                            hl, x="Label", y="Count", color="Label",
                            color_discrete_sequence=px.colors.qualitative.Set2,
                        )
                        fig3.update_layout(height=280, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
                        st.plotly_chart(fig3, use_container_width=True)

                    st.markdown("#### Audit Records")
                    display_cols = [c for c in [
                        "id", "timestamp", "query", "model_name", "model_version",
                        "prompt_version", "toxicity_score", "bias_score",
                        "faithfulness_score", "safety_passed", "latency_ms",
                        "human_label", "reviewer_id",
                    ] if c in df.columns]
                    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Export as CSV", csv, "eval_audit_log.csv", "text/csv")
            else:
                st.error(f"HTTP {hist.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")


# =============================================================================
# TAB 3 - Human Feedback
# =============================================================================
with tabs[2]:
    st.subheader("Human Review Workflow")
    st.write("Submit reviewer judgment on a specific evaluation to track automated vs human agreement.")

    col_a, col_b = st.columns(2)

    with col_a:
        eval_id          = st.number_input("Evaluation ID", min_value=1, step=1, value=1)
        human_label      = st.selectbox("Human Label", ["Correct", "Incorrect", "Unsafe", "Irrelevant", "Partial"])
        reviewer_id      = st.text_input("Reviewer ID", value="reviewer_01")
        reviewer_comment = st.text_area("Reviewer Comment (optional)", height=100)

        submit_feedback = st.button("✅ Submit Feedback", type="primary", use_container_width=True)

    with col_b:
        st.info("""
**Label Definitions**

| Label | Meaning |
|---|---|
| Correct | Response is accurate and grounded |
| Incorrect | Response contains factual errors |
| Unsafe | Response is harmful or toxic |
| Irrelevant | Response does not answer the question |
| Partial | Response is partially correct |
        """)

    if submit_feedback:
        if not backend_online:
            st.error("Backend offline.")
        else:
            try:
                res = requests.post(
                    f"{API_BASE_URL}/feedback/{eval_id}",
                    json={
                        "human_label":      human_label,
                        "reviewer_id":      reviewer_id,
                        "reviewer_comment": reviewer_comment,
                    },
                    headers=AUTH_HEADERS,
                    timeout=15,
                )
                if res.status_code == 200:
                    result = res.json()
                    st.success("Feedback recorded successfully!")
                    st.json({
                        "Eval ID":           result.get("eval_id"),
                        "Human Label":       result.get("human_label"),
                        "Automated Passed":  result.get("automated_passed"),
                        "Agreement":         "✅ Yes" if result.get("agreement") else "❌ No",
                        "Auto Faithfulness": result.get("automated_faithfulness"),
                    })
                elif res.status_code == 404:
                    st.error(f"Evaluation ID {eval_id} not found.")
                else:
                    st.error(f"HTTP {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Error: {e}")


# =============================================================================
# TAB 4 - Model Comparison
# =============================================================================
with tabs[3]:
    st.subheader("Model / Version Comparison")
    st.write("Compare average scores, latency, and pass rates across two models or versions.")

    ca, cb = st.columns(2)
    with ca:
        model_a = st.text_input("Model A", value="gemini-2.0-flash-lite")
    with cb:
        model_b = st.text_input("Model B", value="gemini-2.5-flash")

    if st.button("⚖️ Compare Models", type="primary"):
        if not backend_online:
            st.error("Backend offline.")
        else:
            try:
                res = requests.get(
                    f"{API_BASE_URL}/compare",
                    params={"model_a": model_a, "model_b": model_b},
                    headers=AUTH_HEADERS,
                    timeout=30,
                )
                if res.status_code == 200:
                    data = res.json()
                    a    = data.get("model_a", {})
                    b    = data.get("model_b", {})

                    if "error" in a:
                        st.warning(f"Model A: {a['error']}")
                    if "error" in b:
                        st.warning(f"Model B: {b['error']}")

                    if "count" in a and "count" in b:
                        metrics = [
                            "avg_toxicity", "avg_bias", "avg_hallucination",
                            "avg_faithfulness", "avg_relevancy",
                            "avg_latency_ms", "safety_pass_rate",
                        ]
                        labels = [
                            "Toxicity", "Bias", "Hallucination",
                            "Faithfulness", "Relevancy",
                            "Latency (ms)", "Safety Pass Rate",
                        ]
                        compare_df = pd.DataFrame({
                            "Metric": labels,
                            model_a:  [a.get(m, 0) for m in metrics],
                            model_b:  [b.get(m, 0) for m in metrics],
                        })

                        st.dataframe(compare_df.set_index("Metric"), use_container_width=True)

                        fig = px.bar(
                            compare_df.melt(id_vars="Metric", var_name="Model", value_name="Score"),
                            x="Metric", y="Score", color="Model", barmode="group",
                            color_discrete_sequence=["#3B82F6", "#F59E0B"],
                        )
                        fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"HTTP {res.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")


# =============================================================================
# TAB 5 - Drift Detection
# =============================================================================
with tabs[4]:
    st.subheader("Drift Detection")
    st.write("Compare recent evaluation scores against a historical baseline to detect quality degradation.")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        baseline_days = st.number_input("Baseline Window (days)", min_value=1, value=7)
    with col_d2:
        recent_days = st.number_input("Recent Window (days)", min_value=1, value=1)

    if st.button("🔍 Analyse Drift", type="primary"):
        if not backend_online:
            st.error("Backend offline.")
        else:
            try:
                res = requests.get(
                    f"{API_BASE_URL}/drift",
                    params={"baseline_days": baseline_days, "recent_days": recent_days},
                    headers=AUTH_HEADERS,
                    timeout=30,
                )
                if res.status_code == 200:
                    data = res.json()

                    if not data["results"]:
                        st.info("Not enough data for drift analysis. Run more evaluations across multiple days.")
                    else:
                        d1, d2, d3 = st.columns(3)
                        d1.metric("Metrics Analysed", data["total_metrics"])
                        d2.metric("Drifted Metrics",  data["drifted_count"])
                        d3.metric("Drift Detected",   "⚠️ Yes" if data["drift_detected"] else "✅ No")

                        st.divider()

                        for r in data["results"]:
                            col_m, col_b, col_r, col_d, col_s = st.columns(5)
                            col_m.write(f"**{r['metric']}**")
                            col_b.metric("Baseline", f"{r['baseline_avg']:.3f}")
                            col_r.metric("Recent",   f"{r['recent_avg']:.3f}")
                            col_d.metric("Drift %",  f"{r['drift_pct']:+.1f}%")
                            if r["drifted"]:
                                col_s.error("⚠️ Drifted")
                            else:
                                col_s.success("✅ Stable")

                        st.divider()

                        drift_df = pd.DataFrame(data["results"])
                        fig = px.bar(
                            drift_df,
                            x="metric",
                            y="drift_pct",
                            color="direction",
                            color_discrete_map={
                                "degraded": "#EF4444",
                                "improved": "#22C55E",
                                "stable":   "#94A3B8",
                            },
                            title="Drift % per Metric",
                        )
                        fig.add_hline(y=5,  line_dash="dash", line_color="orange", annotation_text="+5% threshold")
                        fig.add_hline(y=-5, line_dash="dash", line_color="orange", annotation_text="-5% threshold")
                        fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error(f"HTTP {res.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")


# =============================================================================
# TAB 6 - Latency Tracking
# =============================================================================
with tabs[5]:
    st.subheader("p50 / p95 / p99 Latency Tracking")
    st.write("Percentile latency breakdown per evaluation engine and per model.")

    if st.button("📊 Load Latency Stats", type="primary"):
        if not backend_online:
            st.error("Backend offline.")
        else:
            try:
                res = requests.get(
                    f"{API_BASE_URL}/latency",
                    params={"limit": 500},
                    headers=AUTH_HEADERS,
                    timeout=30,
                )
                if res.status_code == 200:
                    data    = res.json()
                    overall = data.get("overall", {})

                    if data.get("p95_alert"):
                        st.error(data.get("alert_msg"))
                    else:
                        st.success(data.get("alert_msg"))

                    st.divider()

                    st.markdown("#### Overall Latency Percentiles")
                    l1, l2, l3, l4, l5 = st.columns(5)
                    l1.metric("Total Evals", overall.get("count", 0))
                    l2.metric("Avg (ms)",    f"{overall.get('avg_ms', 0):.0f}")
                    l3.metric("p50 (ms)",    f"{overall.get('p50_ms', 0):.0f}")
                    l4.metric("p95 (ms)",    f"{overall.get('p95_ms', 0):.0f}",
                              delta="⚠️ High" if overall.get("p95_ms", 0) > 5000 else "✅ OK",
                              delta_color="inverse")
                    l5.metric("p99 (ms)",    f"{overall.get('p99_ms', 0):.0f}")

                    st.divider()

                    per_model = data.get("per_model", {})
                    if per_model:
                        st.markdown("#### Per Model Latency Breakdown")

                        rows = []
                        for model, stats in per_model.items():
                            rows.append({
                                "Model":    model,
                                "Count":    stats["count"],
                                "Min (ms)": stats["min_ms"],
                                "Avg (ms)": stats["avg_ms"],
                                "p50 (ms)": stats["p50_ms"],
                                "p95 (ms)": stats["p95_ms"],
                                "p99 (ms)": stats["p99_ms"],
                                "Max (ms)": stats["max_ms"],
                            })

                        lat_df = pd.DataFrame(rows)
                        st.dataframe(lat_df.set_index("Model"), use_container_width=True)

                        st.divider()

                        st.markdown("#### p50 vs p95 vs p99 per Model")
                        chart_data = []
                        for model, stats in per_model.items():
                            for pct in ["p50_ms", "p95_ms", "p99_ms"]:
                                chart_data.append({
                                    "Model":      model,
                                    "Percentile": pct.replace("_ms", "").upper(),
                                    "Latency ms": stats[pct],
                                })

                        chart_df = pd.DataFrame(chart_data)
                        fig = px.bar(
                            chart_df,
                            x="Model",
                            y="Latency ms",
                            color="Percentile",
                            barmode="group",
                            color_discrete_map={
                                "P50": "#22C55E",
                                "P95": "#F59E0B",
                                "P99": "#EF4444",
                            },
                        )
                        fig.add_hline(
                            y=5000,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="5000ms p95 threshold",
                        )
                        fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.info("No per-model data yet. Run evaluations with model name set.")

                else:
                    st.error(f"HTTP {res.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")


# =============================================================================
# TAB 7 - System Health
# =============================================================================
with tabs[6]:
    st.subheader("System Health & Infrastructure")

    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown("### FastAPI Backend")
        if backend_online:
            st.success("🟢 Online (HTTP 200)")
        else:
            st.error("🔴 Offline")
        st.code(f"URL: {API_BASE_URL}")

    with h2:
        st.markdown("### Evaluator LLM")
        st.info("🟢 Gemini 2.0 Flash Lite")
        st.code("Model: gemini-2.0-flash-lite")

    with h3:
        st.markdown("### Embedding Model")
        st.info("🟢 Gemini Embeddings")
        st.code("Model: gemini-embedding-001")

    st.divider()
    st.markdown("### Connection Parameters")
    st.json({
        "API_BASE_URL":         API_BASE_URL,
        "HEALTH_CHECK_URL":     HEALTH_CHECK_URL,
        "HTTP_TIMEOUT_SECONDS": REQUEST_TIMEOUT,
        "AUTH_STATUS":          "Authenticated" if AUTH_TOKEN else "Anonymous",
        "DATABASE_DRIVER":      "AsyncSQLAlchemy 2.0 (asyncpg)",
        "STREAMLIT_HOST":       "0.0.0.0:8501",
        "FEATURES": [
            "JWT Authentication",
            "PII detection pre-check",
            "Model & prompt versioning",
            "Human feedback workflow",
            "Slack alerting",
            "Model comparison endpoint",
            "Golden dataset CI/CD gate",
            "Drift detection (PSI + KS)",
            "p50/p95/p99 latency tracking",
            "Dashboard auto-refresh",
            "CSV export",
            "Production trace replay",
            "Rate limiting + mock mode",
        ],
    })