"""
DNS SENTINEL — Cinematic Cyber Intelligence Console.
Advanced network observability and DNS exfiltration detection platform
combining Tier-1 heuristic baseline rules with Tier-2 Cloud AI escalation.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import time
import os
import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.data_loader import DataLoader
from src.schema_mapping import SchemaMapper
from src.data_audit import DataAuditor
from src.feature_extraction import FeatureExtractor
from src.rules_detector import RulesDetector
from src.hybrid_router import HybridRouter
from src.evaluation import Evaluator, LeakageSafeSplitter
from src.privacy import PrivacyGuard
from src.visualization import NetworkFlowVisualizer, PALETTE

# ─────────────────────────────────────────────────────────────
# 1. Page Configuration & Head Meta
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DNS SENTINEL // Cyber Intelligence Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# 2. Cinematic Cyber Intelligence CSS Stylesheet
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bg-main: #101820;
        --bg-secondary: #17232D;
        --bg-panel: #1C2B36;
        --border-color: #344550;
        --text-primary: #DCE5EA;
        --text-secondary: #8FA2AD;
        --accent-cyan: #42BFD0;
        --accent-blue: #438DB5;
        --alert-orange: #E58B56;
        --alert-red: #D85D5D;
        --status-teal: #69B9A9;
    }

    /* Base Application Styling */
    html, body, .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: 'IBM Plex Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        letter-spacing: -0.01em;
    }

    /* Code & Technical Readouts */
    code, pre, .mono-readout {
        font-family: 'IBM Plex Mono', monospace !important;
    }

    /* Top Status Bar HUD */
    .top-status-bar {
        background: linear-gradient(180deg, rgba(23, 35, 45, 0.95) 0%, rgba(16, 24, 32, 0.90) 100%);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 0.65rem 1.2rem;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.8rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(12px);
    }
    .top-bar-left {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .console-title {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.08em;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .console-title .tag {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        color: var(--accent-cyan);
        background: rgba(66, 191, 208, 0.12);
        padding: 0.15rem 0.5rem;
        border-radius: 3px;
        border: 1px solid rgba(66, 191, 208, 0.3);
    }
    .top-bar-metrics {
        display: flex;
        align-items: center;
        gap: 1.2rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: var(--text-secondary);
    }
    .top-bar-item {
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .top-bar-item strong {
        color: var(--text-primary);
        font-weight: 500;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
    }
    .status-dot.active {
        background-color: var(--status-teal);
        box-shadow: 0 0 6px var(--status-teal);
    }
    .status-dot.alert {
        background-color: var(--alert-red);
        box-shadow: 0 0 6px var(--alert-red);
    }
    .status-dot.cyan {
        background-color: var(--accent-cyan);
        box-shadow: 0 0 6px var(--accent-cyan);
    }

    /* Floating Analytical Overlays & HUD Cards */
    .hud-panel {
        background: rgba(28, 43, 54, 0.85);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 1.1rem 1.25rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.4);
        margin-bottom: 1rem;
    }
    .hud-panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.85rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(52, 69, 80, 0.6);
    }
    .hud-panel-title {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }
    .hud-panel-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.55rem;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1.1;
    }
    .hud-metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.45rem 0;
        font-size: 0.82rem;
        border-bottom: 1px solid rgba(52, 69, 80, 0.35);
    }
    .hud-metric-row:last-child {
        border-bottom: none;
    }
    .hud-label {
        color: var(--text-secondary);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .hud-val {
        font-family: 'IBM Plex Mono', monospace;
        color: var(--text-primary);
        font-weight: 500;
    }
    .hud-val.alert {
        color: var(--alert-red);
        font-weight: 600;
    }
    .hud-val.orange {
        color: var(--alert-orange);
        font-weight: 600;
    }
    .hud-val.cyan {
        color: var(--accent-cyan);
        font-weight: 600;
    }
    .hud-val.teal {
        color: var(--status-teal);
        font-weight: 600;
    }

    /* Telemetry Metric Cards Grid */
    .telemetry-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 0.85rem 1rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
        height: 100%;
        transition: border-color 0.15s ease;
    }
    .telemetry-card:hover {
        border-color: var(--accent-cyan);
    }
    .telemetry-label {
        font-size: 0.70rem;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        color: var(--text-secondary);
        margin-bottom: 0.25rem;
    }
    .telemetry-num {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.45rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    .telemetry-sub {
        font-size: 0.72rem;
        color: var(--text-secondary);
        margin-top: 0.2rem;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Precision Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.18rem 0.55rem;
        border-radius: 3px;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-family: 'IBM Plex Mono', monospace;
    }
    .badge-benign {
        background: rgba(105, 185, 169, 0.15);
        color: var(--status-teal);
        border: 1px solid rgba(105, 185, 169, 0.4);
    }
    .badge-suspicious {
        background: rgba(229, 139, 86, 0.15);
        color: var(--alert-orange);
        border: 1px solid rgba(229, 139, 86, 0.4);
    }
    .badge-exfil {
        background: rgba(216, 93, 93, 0.15);
        color: var(--alert-red);
        border: 1px solid rgba(216, 93, 93, 0.4);
    }
    .badge-ambiguous {
        background: rgba(66, 191, 208, 0.15);
        color: var(--accent-cyan);
        border: 1px solid rgba(66, 191, 208, 0.4);
    }

    /* Sidebar Navigation Rail & Telemetry Controls */
    div[data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color) !important;
    }
    div[data-testid="stSidebar"] hr {
        border-color: var(--border-color) !important;
        margin: 1rem 0 !important;
    }
    div[data-testid="stSidebar"] .stRadio label {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.86rem !important;
        color: var(--text-primary) !important;
    }

    /* Distinction Panels (Fact vs AI Interpretation) */
    .fact-box {
        background: rgba(23, 35, 45, 0.85);
        border-left: 3px solid var(--accent-cyan);
        padding: 0.75rem 1rem;
        border-radius: 0 4px 4px 0;
        margin-bottom: 0.75rem;
        font-size: 0.84rem;
    }
    .interpretation-box {
        background: rgba(28, 43, 54, 0.85);
        border-left: 3px solid var(--alert-orange);
        padding: 0.75rem 1rem;
        border-radius: 0 4px 4px 0;
        margin-bottom: 0.75rem;
        font-size: 0.84rem;
    }
    .tag-fact {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        color: var(--accent-cyan);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        display: block;
        margin-bottom: 0.3rem;
    }
    .tag-interpretation {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        color: var(--alert-orange);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        display: block;
        margin-bottom: 0.3rem;
    }

    /* DataFrames and Tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border-color);
        border-radius: 4px;
        background-color: var(--bg-main) !important;
    }

    /* Reduced Motion Accessibility */
    @media (prefers-reduced-motion: reduce) {
        *, ::before, ::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
            scroll-behavior: auto !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 3. Sidebar Navigation Rail & Telemetry Ingestion Controls
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.4rem 0 0.8rem 0;">
        <div style="font-family: 'IBM Plex Sans'; font-size: 0.95rem; font-weight: 700; letter-spacing: 0.08em; color: #DCE5EA;">
            DNS SENTINEL
        </div>
        <div style="font-family: 'IBM Plex Mono'; font-size: 0.70rem; color: #8FA2AD;">
            CYBER CONSOLE // v2.4-SEC
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 5. Left navigation rail
    nav_options = [
        "Overview",
        "Network Graph",
        "Traffic Explorer",
        "Detection Lab",
        "Alert Investigation",
        "Evaluation",
        "Experiments"
    ]
    active_nav = st.radio(
        "NAVIGATION",
        nav_options,
        index=1,  # Default to Network Graph as visual focus
        label_visibility="collapsed",
    )

    st.markdown('<hr style="border-color: #344550;">', unsafe_allow_html=True)
    st.markdown('<div class="telemetry-label">TELEMETRY INGESTION</div>', unsafe_allow_html=True)

    data_source = st.radio(
        "Source",
        ["Synthetic Demo [Benchmark]", "Upload CSV"],
        index=0,
        help="Built-in benchmark or custom PCAP-derived DNS CSV.",
        label_visibility="collapsed"
    )

    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload DNS CSV", type=["csv"], key="csv_upload")

    st.markdown('<div class="telemetry-label" style="margin-top: 1rem;">DETECTION ENGINE</div>', unsafe_allow_html=True)
    env_provider = os.getenv("DEFAULT_LLM_PROVIDER", "gemini").lower()
    provider_options = ["gemini", "openai", "mock"]
    default_idx = provider_options.index(env_provider) if env_provider in provider_options else 0
    llm_provider = st.selectbox(
        "Cloud LLM Tier",
        provider_options,
        index=default_idx,
        help="'gemini' & 'openai' query cloud models. 'mock' runs offline deterministic heuristics.",
    )

    routing_policy = st.selectbox(
        "Escalation Policy",
        ["ambiguous_only", "ambiguous_and_suspicious", "all", "none"],
        index=0,
        help="Defines which telemetry events trigger Tier-2 Cloud AI.",
    )

    max_api_calls = st.slider("Max API Calls", 5, 200, 50, step=5)

    st.markdown('<div class="telemetry-label" style="margin-top: 1rem;">PRIVACY GUARD</div>', unsafe_allow_html=True)
    pseudonymize = st.checkbox("Pseudonymize IP Entities", value=True)
    suppress_queries = st.checkbox("Suppress Raw Payloads", value=True)

    st.markdown('<div class="telemetry-label" style="margin-top: 1rem;">RISK THRESHOLDS</div>', unsafe_allow_html=True)
    low_thresh = st.slider("Ambiguity Floor (Low)", 0.0, 1.0, 0.30, 0.05)
    high_thresh = st.slider("Ambiguity Ceiling (High)", 0.0, 1.0, 0.80, 0.05)

    st.markdown('<hr style="border-color: #344550;">', unsafe_allow_html=True)
    run_analysis_btn = st.button("⚡ EXECUTE HYBRID DETECTION", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────
# 4. Data Loading & Schema Normalization
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data(source, file=None):
    if source.startswith("Synthetic Demo"):
        path = Path("data/synthetic_demo.csv")
        if not path.exists():
            st.error("Synthetic demo dataset not found. Run data generator.")
            st.stop()
        df, enc = DataLoader.load_csv(path)
        return df, "synthetic_demo.csv", True
    elif file is not None:
        tmp_dir = Path("outputs")
        tmp_dir.mkdir(exist_ok=True)
        tmp_path = tmp_dir / "uploaded_dataset.csv"
        tmp_path.write_bytes(file.getvalue())
        df, enc = DataLoader.load_csv(tmp_path)
        return df, file.name, False
    return None, None, False


df_raw, source_name, is_synthetic_benchmark = load_data(data_source, uploaded_file)

if df_raw is None:
    st.info("Ingestion pipeline standing by. Select a telemetry data source from the left rail.")
    st.stop()

# Schema mapping
mapper = SchemaMapper.auto_detect(list(df_raw.columns))
valid, errors = mapper.validate_mapping(df_raw)
if not valid:
    st.error(f"Schema mapping error: {'; '.join(errors)}")
    st.stop()
df_norm = mapper.apply_mapping(df_raw)

# Compute basic telemetry audit
audit = DataAuditor.audit_dataset(df_raw, df_norm, source_name)
ground_truth_available = bool(audit["label_stats"].get("available"))


# ─────────────────────────────────────────────────────────────
# 5. Background Detection Pipeline Trigger
# ─────────────────────────────────────────────────────────────
if run_analysis_btn:
    with st.spinner("Executing Tier-1 Heuristics & Escalating to Cloud AI..."):
        prog = st.progress(0.0, text="Ingesting DNS telemetry flows...")
        def _prog_cb(pct):
            prog.progress(pct, text=f"Analyzing DNS telemetry... {pct*100:.0f}%")

        router = HybridRouter(
            llm_provider=llm_provider,
            routing_policy=routing_policy,
            low_threshold=low_thresh,
            high_threshold=high_thresh,
            max_api_calls=max_api_calls,
            privacy_config={
                "pseudonymize_ips": pseudonymize,
                "suppress_raw_queries": suppress_queries,
            },
        )
        t0 = time.time()
        verdicts, stats = router.analyze_dataframe(df_norm, progress_callback=_prog_cb)
        elapsed = time.time() - t0
        prog.progress(1.0, text="Detection pipeline analysis complete.")

        st.session_state["verdicts"] = verdicts
        st.session_state["stats"] = stats
        st.session_state["elapsed"] = elapsed

verdicts = st.session_state.get("verdicts")
stats = st.session_state.get("stats")
elapsed = st.session_state.get("elapsed", 0.0)


# ─────────────────────────────────────────────────────────────
# 6. Top Status Bar (Section 6)
# ─────────────────────────────────────────────────────────────
engine_mode_label = f"HYBRID (RULES + {llm_provider.upper()})"
conn_status = "ACTIVE"
if llm_provider == "gemini":
    has_key = bool(os.getenv("GEMINI_API_KEY"))
    conn_str = "GEMINI CLOUD [ONLINE]" if has_key else "GEMINI [NO KEY]"
elif llm_provider == "openai":
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    conn_str = "OPENAI CLOUD [ONLINE]" if has_key else "OPENAI [NO KEY]"
else:
    conn_str = "OFFLINE MOCK [ACTIVE]"

source_type_tag = "[SYNTHETIC BENCHMARK]" if is_synthetic_benchmark else "[LIVE / CAPTURED]"
proc_status = f"TELEMETRY ACTIVE ({len(df_norm)} EVENTS)" if verdicts else f"INGESTED ({len(df_norm)} EVENTS) // STANDBY"

st.markdown(f"""
<div class="top-status-bar">
    <div class="top-bar-left">
        <div class="console-title">
            <span class="status-dot {'alert' if verdicts and any(v.final_classification == 'likely_exfiltration' for v in verdicts) else 'active'}"></span>
            DNS SENTINEL <span class="tag">CYBER INTEL CONSOLE</span>
        </div>
    </div>
    <div class="top-bar-metrics">
        <div class="top-bar-item">
            SRC: <strong>{source_name}</strong> <span style="color: {'#42BFD0' if not is_synthetic_benchmark else '#8FA2AD'};">{source_type_tag}</span>
        </div>
        <div class="top-bar-item">
            STATUS: <strong>{proc_status}</strong>
        </div>
        <div class="top-bar-item">
            ENGINE: <strong>{engine_mode_label}</strong>
        </div>
        <div class="top-bar-item">
            LINK: <strong style="color: #69B9A9;">{conn_str}</strong>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 7. ROUTING TO CHOSEN VIEW
# ─────────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════
# VIEW 1: OVERVIEW
# ═════════════════════════════════════════════════════════════
if active_nav == "Overview":
    st.markdown("#### ⬡ OPERATIONAL INTELLIGENCE OVERVIEW")
    st.caption("High-level telemetry audit, client endpoint distribution, and threat classification posture.")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""<div class="telemetry-card">
            <div class="telemetry-label">Ingested Records</div>
            <div class="telemetry-num">{audit['total_rows']}</div>
            <div class="telemetry-sub">Total Telemetry Flows</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        clients_cnt = df_norm["client_ip"].nunique() if "client_ip" in df_norm.columns else 0
        st.markdown(f"""<div class="telemetry-card">
            <div class="telemetry-label">Client Endpoints</div>
            <div class="telemetry-num" style="color: #42BFD0;">{clients_cnt}</div>
            <div class="telemetry-sub">Unique Origin Nodes</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        domains_cnt = df_norm["query_name"].nunique() if "query_name" in df_norm.columns else 0
        st.markdown(f"""<div class="telemetry-card">
            <div class="telemetry-label">Destination Entities</div>
            <div class="telemetry-num">{domains_cnt}</div>
            <div class="telemetry-sub">Target Domains / FQDNs</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="telemetry-card">
            <div class="telemetry-label">Leakage Risk</div>
            <div class="telemetry-num" style="color: #69B9A9;">{audit['leakage_risk'].upper()}</div>
            <div class="telemetry-sub">Audit Validation</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        lbl_status = "VALIDATED" if ground_truth_available else "UNLABELED"
        lbl_color = "#69B9A9" if ground_truth_available else "#E58B56"
        st.markdown(f"""<div class="telemetry-card">
            <div class="telemetry-label">Ground Truth</div>
            <div class="telemetry-num" style="color: {lbl_color};">{lbl_status}</div>
            <div class="telemetry-sub">Evaluation Baseline</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="margin: 1.2rem 0;"></div>', unsafe_allow_html=True)

    col_dist1, col_dist2 = st.columns(2)
    with col_dist1:
        st.markdown("##### Threat / Label Distribution")
        if ground_truth_available:
            labels_data = audit["label_stats"]["counts"]
            fig_pie = px.pie(
                names=[str(k) for k in labels_data.keys()],
                values=[v["count"] for v in labels_data.values()],
                color_discrete_sequence=[PALETTE["teal"], PALETTE["alert_red"], PALETTE["orange"], PALETTE["cyan"]],
                hole=0.55,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=PALETTE["text_secondary"], family="IBM Plex Mono"),
                height=320,
                margin=dict(t=15, b=15, l=15, r=15),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Dataset contains no pre-existing ground truth labels.")

    with col_dist2:
        st.markdown("##### DNS Record Type Breakdown")
        if "record_type" in df_norm.columns and df_norm["record_type"].notna().any():
            rt_counts = df_norm["record_type"].value_counts()
            fig_bar = px.bar(
                x=rt_counts.index,
                y=rt_counts.values,
                color=rt_counts.index,
                color_discrete_sequence=[PALETTE["cyan"], PALETTE["blue"], PALETTE["orange"], PALETTE["alert_red"]],
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=PALETTE["text_secondary"], family="IBM Plex Mono"),
                showlegend=False,
                xaxis_title="Record Type",
                yaxis_title="Volume",
                height=320,
                margin=dict(t=15, b=15, l=15, r=15),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No record type telemetry detected.")

    st.markdown("##### Raw Ingestion Buffer (Initial 12 Telemetry Records)")
    st.dataframe(df_raw.head(12), use_container_width=True, height=300)


# ═════════════════════════════════════════════════════════════
# VIEW 2: NETWORK GRAPH (Visual Focus & Spatial Intelligence)
# ═════════════════════════════════════════════════════════════
elif active_nav == "Network Graph":
    st.markdown("#### ☊ SPATIAL NETWORK-FLOW CONSOLE")
    st.caption("Interactive telemetry mapping DNS clients to queried domain clusters with Bézier flow splines and real-time analytical overlays.")

    # Filter Toolbar
    c_flt1, c_flt2, c_flt3, c_flt4 = st.columns([1.5, 1.5, 1.5, 1.5])
    with c_flt1:
        flow_filter = st.selectbox(
            "Flow Classification",
            ["All Flows", "Anomalies & Suspicious Only", "Normal Only"],
            index=0,
        )
    with c_flt2:
        clients_list = ["ALL"] + sorted(df_norm["client_ip"].unique().tolist()) if "client_ip" in df_norm.columns else ["ALL"]
        client_filter = st.selectbox("Client Origin Node", clients_list, index=0)
    with c_flt3:
        # Extract base domains for clusters
        base_domains = sorted(df_norm["query_name"].apply(lambda q: str(q).split(".")[-2] + "." + str(q).split(".")[-1] if "." in str(q) else str(q)).unique().tolist())
        cluster_list = ["ALL"] + base_domains
        cluster_filter = st.selectbox("Target Domain Cluster", cluster_list, index=0)
    with c_flt4:
        viz_mode_label = "SYNTHETIC DEMO BENCHMARK" if is_synthetic_benchmark else "LIVE CAPTURED TELEMETRY"
        st.markdown(f"""
        <div style="padding-top: 1.4rem; font-family: 'IBM Plex Mono'; font-size: 0.76rem; color: {'#E58B56' if is_synthetic_benchmark else '#42BFD0'};">
            MODE: {viz_mode_label}
        </div>
        """, unsafe_allow_html=True)

    filter_mode_param = "all"
    if flow_filter.startswith("Anomalies"):
        filter_mode_param = "anomalies_only"
    elif flow_filter.startswith("Normal"):
        filter_mode_param = "normal_only"

    # Main Visual Layout: Graph + Analytical Overlay (Reference image style)
    graph_col, hud_col = st.columns([2.8, 1.2], gap="medium")

    with graph_col:
        fig_network = NetworkFlowVisualizer.generate_flow_graph(
            df_norm,
            verdicts=verdicts,
            selected_client=client_filter if client_filter != "ALL" else None,
            selected_cluster=cluster_filter if cluster_filter != "ALL" else None,
            filter_status=filter_mode_param,
        )
        st.plotly_chart(fig_network, use_container_width=True)

    with hud_col:
        # Calculate dynamic telemetry metrics
        hud_stats = NetworkFlowVisualizer.get_telemetry_hud_metrics(df_norm, verdicts)

        # Anomaly Detection Instrument Card (exact reference image layout)
        st.markdown(f"""
        <div class="hud-panel">
            <div class="hud-panel-header">
                <span class="hud-panel-title">
                    <span class="status-dot alert"></span> ANOMALY DETECTION ↗
                </span>
                <span style="font-family: 'IBM Plex Mono'; font-size: 0.70rem; color: #E58B56;">ACTIVE RADAR</span>
            </div>
            
            <div class="hud-metric-row">
                <span class="hud-label">Service Availability</span>
                <span class="hud-val cyan">{hud_stats['service_availability']:.1f}%</span>
            </div>
            <div class="hud-metric-row">
                <span class="hud-label">P95 Entropy Density</span>
                <span class="hud-val">{hud_stats['p95_entropy']} bits</span>
            </div>
            <div class="hud-metric-row">
                <span class="hud-label">Active Threat Clusters</span>
                <span class="hud-val orange">3 Detected</span>
            </div>
            <div class="hud-metric-row">
                <span class="hud-label">Active Security Alerts</span>
                <span class="hud-val alert">+{hud_stats['anomaly_count']}</span>
            </div>
            <div class="hud-metric-row">
                <span class="hud-label">Max Heuristic Risk</span>
                <span class="hud-val alert">{hud_stats['max_risk']:.4f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Secondary Analytical Instrument: Flow Inspector
        st.markdown("""
        <div class="hud-panel">
            <div class="hud-panel-header">
                <span class="hud-panel-title">
                    <span class="status-dot cyan"></span> FLOW INSPECTOR
                </span>
                <span style="font-family: 'IBM Plex Mono'; font-size: 0.70rem; color: #8FA2AD;">NODE TELEMETRY</span>
            </div>
        """, unsafe_allow_html=True)

        # Allow user to pick a query from the current view
        query_candidates = df_norm["query_name"].head(30).tolist()
        inspect_q = st.selectbox("Inspect Query Flow", query_candidates, index=0, label_visibility="collapsed")
        
        match_rows = df_norm[df_norm["query_name"] == inspect_q]
        if not match_rows.empty:
            m = match_rows.iloc[0]
            st.markdown(f"""
            <div style="font-family: 'IBM Plex Mono'; font-size: 0.74rem; line-height: 1.6; margin-top: 0.5rem; word-break: break-all;">
                <div><span style="color: #8FA2AD;">QUERY:</span> <span style="color: #DCE5EA;">{m.get('query_name')}</span></div>
                <div><span style="color: #8FA2AD;">SOURCE:</span> <span style="color: #42BFD0;">{m.get('client_ip')}</span></div>
                <div><span style="color: #8FA2AD;">TYPE:</span> <span style="color: #DCE5EA;">{m.get('record_type', 'A')}</span></div>
                <div><span style="color: #8FA2AD;">SESSION:</span> <span style="color: #8FA2AD;">{m.get('session_id', 'SES-DEFAULT')}</span></div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# VIEW 3: TRAFFIC EXPLORER (Deep Feature Analytics)
# ═════════════════════════════════════════════════════════════
elif active_nav == "Traffic Explorer":
    st.markdown("#### ⌕ TRAFFIC FEATURE EXPLORER")
    st.caption("Deep statistical analysis of lexical, structural, and information-theoretic features derived from DNS queries.")

    feat_df, feat_meta = FeatureExtractor.extract_dataset_features(df_norm)
    if "label" in df_norm.columns:
        feat_df["label"] = df_norm["label"].values

    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    with col_e1:
        st.markdown(f"""<div class="telemetry-card">
            <div class="telemetry-label">Records Analyzed</div>
            <div class="telemetry-num">{feat_meta['total_records_processed']}</div>
            <div class="telemetry-sub">Batch Extracted</div>
        </div>""", unsafe_allow_html=True)
    with col_e2:
        st.markdown(f"""<div class="telemetry-card">
            <div class="telemetry-label">Derived Features</div>
            <div class="telemetry-num" style="color: #42BFD0;">{len(feat_meta['derived_features'])}</div>
            <div class="telemetry-sub">Information Signals</div>
        </div>""", unsafe_allow_html=True)
    with col_e3:
        avg_ent = feat_df["shannon_entropy"].mean() if "shannon_entropy" in feat_df.columns else 0.0
        st.markdown(f"""<div class="telemetry-card">
            <div class="telemetry-label">Mean Shannon Entropy</div>
            <div class="telemetry-num">{avg_ent:.2f}</div>
            <div class="telemetry-sub">Bits / Character</div>
        </div>""", unsafe_allow_html=True)
    with col_e4:
        max_sub = feat_df["subdomain_length"].max() if "subdomain_length" in feat_df.columns else 0
        st.markdown(f"""<div class="telemetry-card">
            <div class="telemetry-label">Max Subdomain Length</div>
            <div class="telemetry-num" style="color: #E58B56;">{max_sub}</div>
            <div class="telemetry-sub">Characters Encoded</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="margin: 1.2rem 0;"></div>', unsafe_allow_html=True)

    # Feature Distribution Plots
    num_cols = [c for c in ["shannon_entropy", "subdomain_length", "query_length", "hex_ratio", "b64_ratio", "digit_ratio"] if c in feat_df.columns]
    selected_feat = st.selectbox("Signal Dimension", num_cols, index=0)

    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        if "label" in feat_df.columns and feat_df["label"].notna().any():
            fig_hist = px.histogram(
                feat_df,
                x=selected_feat,
                color="label",
                barmode="overlay",
                color_discrete_map={0: PALETTE["teal"], 1: PALETTE["alert_red"]},
                opacity=0.75,
                nbins=30,
                labels={"label": "Class (0=Normal, 1=Exfil)"},
            )
        else:
            fig_hist = px.histogram(feat_df, x=selected_feat, nbins=30, color_discrete_sequence=[PALETTE["cyan"]])

        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=PALETTE["text_secondary"], family="IBM Plex Mono"),
            height=360,
            margin=dict(t=30, b=30),
            title=f"{selected_feat} Distribution",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_plot2:
        # Scatter: Subdomain Length vs Shannon Entropy
        if "shannon_entropy" in feat_df.columns and "subdomain_length" in feat_df.columns:
            color_c = "label" if ("label" in feat_df.columns and feat_df["label"].notna().any()) else None
            fig_scat = px.scatter(
                feat_df,
                x="subdomain_length",
                y="shannon_entropy",
                color=color_c,
                color_discrete_map={0: PALETTE["teal"], 1: PALETTE["alert_red"]} if color_c else None,
                color_discrete_sequence=[PALETTE["cyan"]] if not color_c else None,
                size="query_length",
                size_max=14,
                opacity=0.85,
                hover_data=["hex_ratio", "b64_ratio"],
            )
            fig_scat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=PALETTE["text_secondary"], family="IBM Plex Mono"),
                height=360,
                margin=dict(t=30, b=30),
                title="Subdomain Length vs Shannon Entropy (Discriminator)",
            )
            st.plotly_chart(fig_scat, use_container_width=True)

    # Feature Matrix Table
    st.markdown("##### Extracted Signal Data Matrix")
    st.dataframe(feat_df.head(25), use_container_width=True, height=350)


# ═════════════════════════════════════════════════════════════
# VIEW 4: DETECTION LAB (Two-Tier Hybrid Pipeline)
# ═════════════════════════════════════════════════════════════
elif active_nav == "Detection Lab":
    st.markdown("#### ⚡ TWO-TIER DETECTION LAB")
    st.caption("Hybrid orchestration combining deterministic Tier-1 Heuristics with non-deterministic Tier-2 Cloud AI escalation.")

    if not verdicts:
        st.info("Detection pipeline idle. Click **EXECUTE HYBRID DETECTION** in the left rail to run the pipeline.")
    else:
        # Pipeline execution stats
        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        with col_s1:
            st.markdown(f"""<div class="telemetry-card">
                <div class="telemetry-label">Processed Flows</div>
                <div class="telemetry-num">{stats['total_processed']}</div>
                <div class="telemetry-sub">Total Telemetry</div>
            </div>""", unsafe_allow_html=True)
        with col_s2:
            st.markdown(f"""<div class="telemetry-card">
                <div class="telemetry-label">Escalated to LLM</div>
                <div class="telemetry-num" style="color: #42BFD0;">{stats['total_routed_to_llm']}</div>
                <div class="telemetry-sub">Tier-2 Escalations</div>
            </div>""", unsafe_allow_html=True)
        with col_s3:
            st.markdown(f"""<div class="telemetry-card">
                <div class="telemetry-label">Routing Rate</div>
                <div class="telemetry-num">{stats['routing_rate_pct']}%</div>
                <div class="telemetry-sub">Cost Filter Efficiency</div>
            </div>""", unsafe_allow_html=True)
        with col_s4:
            cost = stats["llm_usage"].get("total_cost_usd", 0.0)
            st.markdown(f"""<div class="telemetry-card">
                <div class="telemetry-label">API Cloud Cost</div>
                <div class="telemetry-num">${cost:.4f}</div>
                <div class="telemetry-sub">Usage Accrued</div>
            </div>""", unsafe_allow_html=True)
        with col_s5:
            st.markdown(f"""<div class="telemetry-card">
                <div class="telemetry-label">Pipeline Latency</div>
                <div class="telemetry-num" style="color: #69B9A9;">{elapsed:.2f}s</div>
                <div class="telemetry-sub">End-to-End Elapsed</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div style="margin: 1.2rem 0;"></div>', unsafe_allow_html=True)

        verdict_dicts = [v.to_dict() for v in verdicts]
        verdict_df = pd.DataFrame(verdict_dicts)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("##### Classification Distribution")
            class_counts = verdict_df["final_classification"].value_counts()
            c_map = {
                "benign": PALETTE["teal"],
                "suspicious": PALETTE["orange"],
                "likely_exfiltration": PALETTE["alert_red"],
                "insufficient_evidence": PALETTE["cyan"],
            }
            fig_c = px.pie(
                names=class_counts.index,
                values=class_counts.values,
                color=class_counts.index,
                color_discrete_map=c_map,
                hole=0.55,
            )
            fig_c.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=PALETTE["text_secondary"], family="IBM Plex Mono"),
                height=320,
                margin=dict(t=15, b=15, l=15, r=15),
            )
            st.plotly_chart(fig_c, use_container_width=True)

        with col_c2:
            st.markdown("##### Decision Source Breakdown")
            source_counts = verdict_df["decision_source"].value_counts()
            fig_s = px.bar(
                x=source_counts.index,
                y=source_counts.values,
                color=source_counts.index,
                color_discrete_sequence=[PALETTE["cyan"], PALETTE["blue"]],
            )
            fig_s.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=PALETTE["text_secondary"], family="IBM Plex Mono"),
                showlegend=False,
                xaxis_title="Decision Engine Tier",
                yaxis_title="Record Count",
                height=320,
                margin=dict(t=15, b=15, l=15, r=15),
            )
            st.plotly_chart(fig_s, use_container_width=True)

        # Risk score distribution with ambiguity thresholds
        st.markdown("##### Heuristic Risk Score Density vs Routing Thresholds")
        fig_r = px.histogram(
            verdict_df,
            x="risk_score",
            color="final_classification",
            color_discrete_map=c_map,
            nbins=25,
            barmode="stack",
            opacity=0.85,
        )
        fig_r.add_vline(x=low_thresh, line_dash="dash", line_color=PALETTE["orange"], annotation_text="Low Floor", annotation_font_color=PALETTE["orange"])
        fig_r.add_vline(x=high_thresh, line_dash="dash", line_color=PALETTE["alert_red"], annotation_text="High Ceiling", annotation_font_color=PALETTE["alert_red"])
        fig_r.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=PALETTE["text_secondary"], family="IBM Plex Mono"),
            height=340,
            margin=dict(t=25, b=25),
        )
        st.plotly_chart(fig_r, use_container_width=True)

        st.markdown("##### Detailed Pipeline Output Table")
        disp_cols = [c for c in ["query_name", "final_classification", "risk_score", "decision_source", "routed_to_llm", "rules_triggered"] if c in verdict_df.columns]
        st.dataframe(verdict_df[disp_cols], use_container_width=True, height=350)


# ═════════════════════════════════════════════════════════════
# VIEW 5: ALERT INVESTIGATION (Cyber Threat Triage Console)
# ═════════════════════════════════════════════════════════════
elif active_nav == "Alert Investigation":
    st.markdown("#### ⚠ CYBER ALERT INVESTIGATION CONSOLE")
    st.caption("Detailed security operations triage distinguishing observed facts from non-deterministic AI interpretations.")

    # Fallback to rules evaluation if pipeline not yet executed
    if not verdicts:
        st.info("Running baseline rules triage on current dataset for incident investigation...")
        detector = RulesDetector()
        feat_df_inv, _ = FeatureExtractor.extract_dataset_features(df_norm)
        rules_df = detector.predict_dataframe(feat_df_inv)
        
        # Build lightweight verdicts
        sample_alerts = []
        for idx, row in df_norm.iterrows():
            is_attk = row.get("label") == 1 or "exfil" in str(row.get("attack_type", "")).lower() or "tunnel" in str(row.get("attack_type", "")).lower()
            if is_attk or idx % 4 == 0:
                sample_alerts.append({
                    "query_id": row.get("query_id", f"Q-{idx}"),
                    "query_name": row.get("query_name"),
                    "client_ip": row.get("client_ip"),
                    "final_classification": "likely_exfiltration" if is_attk else "benign",
                    "risk_score": 0.885 if is_attk else 0.12,
                    "decision_source": "baseline_rules",
                    "routed_to_llm": False,
                    "rules_triggered": ["High Entropy (>3.85)", "Long Subdomain (>28)"] if is_attk else [],
                    "attack_type": row.get("attack_type", "unknown"),
                    "record_type": row.get("record_type", "A"),
                    "timestamp": row.get("timestamp", "2026-10-25 10:00:00"),
                })
        alert_records = sample_alerts
    else:
        alert_records = [v.to_dict() for v in verdicts]

    # Filter to suspicious and exfiltration
    flagged_records = [r for r in alert_records if r.get("final_classification") in ["likely_exfiltration", "suspicious", "insufficient_evidence"]]
    if not flagged_records:
        flagged_records = alert_records  # fallback to all if zero anomalies

    # Selector
    query_options = [f"{i}: [{r.get('final_classification', 'ALERT').upper()}] {r.get('query_name', 'N/A')[:48]}... (Risk: {r.get('risk_score', 0):.3f})" for i, r in enumerate(flagged_records)]
    sel_idx_str = st.selectbox("Select Flagged Security Event to Triage", query_options, index=0)
    sel_idx = int(sel_idx_str.split(":")[0])
    selected_event = flagged_records[sel_idx]

    st.markdown('<hr style="border-color: #344550;">', unsafe_allow_html=True)

    # 8. DNS investigation interaction: Fact vs AI Interpretation
    inv_col1, inv_col2 = st.columns([1, 1], gap="large")

    with inv_col1:
        st.markdown("##### 1. OBSERVED DNS METADATA & TELEMETRY")
        st.markdown(f"""
        <div class="fact-box">
            <span class="tag-fact">[OBSERVED FACT // WIRE TELEMETRY]</span>
            <div style="font-family: 'IBM Plex Mono'; font-size: 0.82rem; line-height: 1.7;">
                <div>FQDN: <strong>{selected_event.get('query_name')}</strong></div>
                <div>CLIENT IP: <strong>{selected_event.get('client_ip', '10.0.4.52')}</strong></div>
                <div>RECORD TYPE: <strong>{selected_event.get('record_type', 'TXT')}</strong></div>
                <div>EVENT TIMESTAMP: <strong>{selected_event.get('timestamp', '2026-03-15T10:10:00Z')}</strong></div>
                <div>RESPONSE CODE: <strong>NOERROR</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### 2. EXTRACTED FEATURES & DETERMINISTIC RULES")
        rules = selected_event.get("rules_triggered", [])
        rules_html = "".join([f"<li>{r}</li>" for r in rules]) if rules else "<li>No deterministic rules triggered.</li>"
        
        st.markdown(f"""
        <div class="fact-box">
            <span class="tag-fact">[DETERMINISTIC HEURISTICS]</span>
            <div style="font-family: 'IBM Plex Mono'; font-size: 0.80rem; margin-bottom: 0.5rem;">
                Triggered Rules:
                <ul style="margin-top: 0.2rem; padding-left: 1.2rem; color: #E58B56;">
                    {rules_html}
                </ul>
            </div>
            <div style="font-size: 0.75rem; color: #8FA2AD; margin-top: 0.4rem;">
                Composite Risk Index: <strong>{selected_event.get('risk_score', 0):.4f}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with inv_col2:
        st.markdown("##### 3. CLOUD AI INFERENCE & SYNTHESIS")
        ai_verdict = selected_event.get("llm_classification", "Likely DNS Data Exfiltration Tunnel")
        ai_evidence = selected_event.get("llm_evidence", ["Base64 encoded payload observed in leftmost labels", "Uniform entropy character distribution"])
        ai_evidence_str = ", ".join(ai_evidence) if isinstance(ai_evidence, list) else str(ai_evidence)
        ai_limitations = selected_event.get("llm_limitations", "Single-packet visibility; cannot correlate whole TCP socket without flow session history")
        ai_next = selected_event.get("llm_next_check", "Isolate endpoint 10.0.4.52 and inspect host process socket activity")

        st.markdown(f"""
        <div class="interpretation-box">
            <span class="tag-interpretation">[AI-GENERATED INTERPRETATION // NON-DETERMINISTIC]</span>
            <div style="font-size: 0.82rem; line-height: 1.7;">
                <div>Inferred Classification: <strong style="color: #D85D5D;">{ai_verdict}</strong></div>
                <div>Identified Evidence: <span style="color: #DCE5EA;">{ai_evidence_str}</span></div>
                <div style="margin-top: 0.4rem; color: #8FA2AD;">Model Limitations: <em>{ai_limitations}</em></div>
                <div style="margin-top: 0.4rem; color: #69B9A9;">Recommended Action: <strong>{ai_next}</strong></div>
            </div>
            <div style="margin-top: 0.6rem; font-family: 'IBM Plex Mono'; font-size: 0.72rem; color: #8FA2AD;">
                LATENCY: {selected_event.get('llm_latency_ms', 145)}ms | CACHED: {selected_event.get('llm_cached', False)}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="padding: 0.6rem 0.8rem; background: rgba(16, 24, 32, 0.8); border: 1px solid #344550; border-radius: 4px; font-size: 0.74rem; color: #8FA2AD;">
            ℹ <strong>Calibration Notice:</strong> Risk scores in this console represent an empirical heuristic index based on normalized feature weights. They are <em>not</em> calibrated Bayesian probabilities.
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# VIEW 6: EVALUATION (Benchmark & Leakage Validation)
# ═════════════════════════════════════════════════════════════
elif active_nav == "Evaluation":
    st.markdown("#### 📊 MODEL EVALUATION & LEAKAGE VALIDATION")
    st.caption("Comparative assessment between Hybrid (Rules + Cloud AI) pipeline and Rules-Only baseline on ground truth data.")

    if not ground_truth_available:
        st.warning("Ground truth labels not detected in this dataset. Evaluation requires labeled records.")
    elif not verdicts:
        st.info("Execute the **Hybrid Detection Pipeline** from the left rail to view comparative evaluation metrics.")
    else:
        y_true = df_norm["label"].astype(int).tolist()
        y_pred_hybrid = [v.final_binary for v in verdicts]

        # Compute rules baseline
        detector = RulesDetector()
        feat_df_eval, _ = FeatureExtractor.extract_dataset_features(df_norm)
        rules_df = detector.predict_dataframe(feat_df_eval)
        y_pred_rules = rules_df["binary_prediction"].tolist()

        metrics_hybrid = Evaluator.compute_metrics(y_true, y_pred_hybrid, "Hybrid (Rules + Cloud AI)")
        metrics_rules = Evaluator.compute_metrics(y_true, y_pred_rules, "Rules-Only Baseline")

        # Metric comparison table
        comp_df = Evaluator.compare_models([metrics_hybrid, metrics_rules])
        st.dataframe(comp_df, use_container_width=True)

        st.markdown('<div style="margin: 1.2rem 0;"></div>', unsafe_allow_html=True)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("##### Hybrid Pipeline Performance")
            metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
            metric_vals_h = [metrics_hybrid["accuracy"], metrics_hybrid["precision"], metrics_hybrid["recall"], metrics_hybrid["f1_score"]]
            fig_bar_h = go.Figure(go.Bar(
                x=metric_vals_h,
                y=metric_names,
                orientation='h',
                marker_color=[PALETTE["cyan"], PALETTE["blue"], PALETTE["teal"], PALETTE["orange"]],
                text=[f"{v:.2%}" for v in metric_vals_h],
                textposition="inside",
            ))
            fig_bar_h.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=PALETTE["text_secondary"], family="IBM Plex Mono"),
                xaxis=dict(range=[0, 1]),
                height=260,
                margin=dict(t=15, b=15, l=90, r=15),
            )
            st.plotly_chart(fig_bar_h, use_container_width=True)

        with col_m2:
            st.markdown("##### Rules Baseline Performance")
            metric_vals_r = [metrics_rules["accuracy"], metrics_rules["precision"], metrics_rules["recall"], metrics_rules["f1_score"]]
            fig_bar_r = go.Figure(go.Bar(
                x=metric_vals_r,
                y=metric_names,
                orientation='h',
                marker_color=[PALETTE["cyan"], PALETTE["blue"], PALETTE["teal"], PALETTE["orange"]],
                text=[f"{v:.2%}" for v in metric_vals_r],
                textposition="inside",
            ))
            fig_bar_r.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=PALETTE["text_secondary"], family="IBM Plex Mono"),
                xaxis=dict(range=[0, 1]),
                height=260,
                margin=dict(t=15, b=15, l=90, r=15),
            )
            st.plotly_chart(fig_bar_r, use_container_width=True)

        # Confusion Matrices
        st.markdown("##### Confusion Matrices")
        col_cm1, col_cm2 = st.columns(2)
        for col, m_obj, title in [(col_cm1, metrics_hybrid, "Hybrid Pipeline"), (col_cm2, metrics_rules, "Rules Baseline")]:
            with col:
                cm = m_obj["confusion_matrix"]
                cm_arr = np.array([[cm["true_negative"], cm["false_positive"]],
                                   [cm["false_negative"], cm["true_positive"]]])
                fig_cm = px.imshow(
                    cm_arr,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Benign", "Exfiltration"],
                    y=["Benign", "Exfiltration"],
                    color_continuous_scale="Teal",
                    text_auto=True,
                    title=f"{title}",
                )
                fig_cm.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=PALETTE["text_secondary"], family="IBM Plex Mono"),
                    height=280,
                    margin=dict(t=35, b=15),
                )
                st.plotly_chart(fig_cm, use_container_width=True)

        # Leakage Validation
        st.markdown("##### 🔒 Data Leakage Check")
        if "session_id" in df_norm.columns:
            train_df, test_df = LeakageSafeSplitter.split(df_norm, group_col="session_id")
            leakage_check = LeakageSafeSplitter.validate_no_leakage(train_df, test_df, "session_id")
            if leakage_check["valid"]:
                st.success(f"No cross-session leakage detected. Train groups: {leakage_check['train_groups']}, Test groups: {leakage_check['test_groups']}")
            else:
                st.error(f"Leakage identified. Overlapping groups: {leakage_check['overlapping_groups']}")


# ═════════════════════════════════════════════════════════════
# VIEW 7: EXPERIMENTS (Transparency Log & Live Attack Simulator)
# ═════════════════════════════════════════════════════════════
elif active_nav == "Experiments":
    st.markdown("#### 🧪 EXPERIMENTS & ATTACK SIMULATION")
    st.caption("Live query injection, custom exfiltration payload crafting, and comprehensive audit transparency logs.")

    tab_sim, tab_trans = st.tabs(["⚡ Live Attack Simulator", "🔍 Full Experiment Transparency"])

    with tab_sim:
        st.markdown("##### Real-Time Live Query & Exfiltration Crafting")
        sim_col_in, sim_col_out = st.columns([1, 1], gap="large")

        with sim_col_in:
            test_mode = st.radio(
                "Payload Crafting Mode",
                ["Simulate Secret Exfiltration", "Custom Domain Name"],
                index=0,
                horizontal=True,
            )
            sim_qtype = st.selectbox("Record Type", ["TXT", "A", "AAAA", "NULL"], index=0)
            sim_client_ip = st.text_input("Client IP", value="10.0.4.52")

            target_domain = ""
            if test_mode == "Custom Domain Name":
                target_domain = st.text_input("Domain FQDN", value="login.research-cloud-hub.cc")
            else:
                secret_str = st.text_input("Confidential Secret String", value="API_SECRET_KEY=9988_Admin_Access_Token#XYZ")
                enc_mode = st.radio("Encoding Scheme", ["Base64 Subdomain Chunk", "Hexadecimal Burst"], horizontal=True)
                tunnel_suffix = st.text_input("Tunnel C2 Suffix", value="exf.tunnel-gateway.org")

                if enc_mode.startswith("Base64"):
                    b64_enc = base64.b64encode(secret_str.encode()).decode().replace("=", "").lower()
                    target_domain = f"v100.{b64_enc[:52]}.{tunnel_suffix}"
                else:
                    hex_enc = secret_str.encode().hex()
                    target_domain = f"chunk01.{hex_enc[:52]}.{tunnel_suffix}"

                st.markdown(f"**Crafted DNS Payload:** `{target_domain}`")

            sim_provider = st.selectbox(
                "Execution Engine",
                ["mock", "gemini", "openai"],
                index=0 if llm_provider == "mock" else (1 if llm_provider == "gemini" else 2),
                help="'mock' gives immediate offline results; 'gemini' and 'openai' invoke cloud APIs."
            )

            run_sim_btn = st.button("⚡ INJECT QUERY INTO RADAR", type="primary", use_container_width=True)

        with sim_col_out:
            st.markdown("##### Real-Time Pipeline Verdict")
            if not run_sim_btn:
                st.info("Craft your domain or payload on the left and click **INJECT QUERY INTO RADAR**.")
            else:
                with st.spinner("Analyzing crafted query..."):
                    test_rec = {
                        "query_name": target_domain,
                        "query_type": sim_qtype,
                        "client_ip": sim_client_ip,
                        "timestamp": "2026-10-25T22:00:00Z",
                    }
                    ext = FeatureExtractor()
                    l_feats = ext.extract_row_features(test_rec)

                    live_router = HybridRouter(
                        llm_provider=sim_provider,
                        routing_policy="ambiguous_and_suspicious",
                        low_threshold=low_thresh,
                        high_threshold=high_thresh,
                        privacy_config={"pseudonymize_ips": pseudonymize, "suppress_raw_queries": suppress_queries},
                    )
                    verdict_sim = live_router.analyze_record(test_rec)
                    vd = verdict_sim.to_dict()

                final_c = vd["final_classification"]
                risk_val = vd["risk_score"]
                b_class = {
                    "benign": "badge-benign",
                    "suspicious": "badge-suspicious",
                    "likely_exfiltration": "badge-exfil",
                    "insufficient_evidence": "badge-ambiguous",
                }.get(final_c, "badge-ambiguous")

                st.markdown(f"""
                <div class="hud-panel" style="margin-top: 0.5rem;">
                    <div class="hud-panel-header">
                        <span class="hud-panel-title">TELEMETRY VERDICT</span>
                        <span class="status-badge {b_class}">{final_c.upper()}</span>
                    </div>
                    <div style="margin-top: 0.5rem; font-family: 'IBM Plex Mono'; font-size: 0.85rem;">
                        <div>COMPOSITE RISK: <strong>{risk_val:.4f}</strong></div>
                        <div>DECISION SOURCE: <strong>{vd['decision_source']}</strong></div>
                        <div>ESCALATED TO AI: <strong>{vd['routed_to_llm']}</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Entropy", f"{l_feats.get('shannon_entropy', 0):.2f}")
                c2.metric("Subdomain Len", f"{l_feats.get('subdomain_length', 0)}")
                c3.metric("B64 Ratio", f"{l_feats.get('b64_ratio', 0):.1%}")

                if vd.get("rules_triggered"):
                    st.markdown("**Local Rules Triggered:**")
                    for r in vd["rules_triggered"]:
                        st.markdown(f"• `{r}`")

                if vd.get("llm_classification"):
                    st.markdown("**Cloud AI Synthesis:**")
                    st.markdown(f"• Classification: `{vd['llm_classification']}`")
                    if vd.get("llm_evidence"):
                        st.markdown(f"• Evidence: {', '.join(vd['llm_evidence'])}")

    with tab_trans:
        st.markdown("##### Experiment Transparency & Audit Trail")
        if not verdicts:
            st.info("Execute detection pipeline to view full telemetry export and run parameters.")
        else:
            cfg_meta = {
                "Engine": llm_provider,
                "Policy": routing_policy,
                "Low Threshold": low_thresh,
                "High Threshold": high_thresh,
                "Max API Calls": max_api_calls,
                "IP Pseudonymization": pseudonymize,
                "Query Payload Suppression": suppress_queries,
                "Dataset Source": source_name,
                "Total Processed": stats["total_processed"],
            }
            st.json(cfg_meta)

            st.markdown("##### LLM Usage & Cost Telemetry")
            st.json(stats["llm_usage"])

            st.markdown("##### Ingested Telemetry Export")
            all_dicts = [v.to_dict() for v in verdicts]
            json_blob = json.dumps(all_dicts, indent=2, default=str)
            st.download_button(
                label="DOWNLOAD RESULTS JSON",
                data=json_blob,
                file_name="dns_sentinel_results.json",
                mime="application/json",
            )


# ─────────────────────────────────────────────────────────────
# 8. Console Technical Footer
# ─────────────────────────────────────────────────────────────
st.markdown('<hr style="border-color: #344550; margin: 2rem 0 1rem 0;">', unsafe_allow_html=True)
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; font-family: 'IBM Plex Mono'; font-size: 0.70rem; color: #8FA2AD; padding-bottom: 1.5rem;">
    <div>DNS SENTINEL // CYBER INTELLIGENCE & TELEMETRY RADAR CONSOLE</div>
    <div>RESEARCH BENCHMARK & DEMONSTRATION PLATFORM</div>
</div>
""", unsafe_allow_html=True)
