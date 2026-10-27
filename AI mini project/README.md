# 🛡️ DNS Exfiltration Detection with Cloud AI

A hybrid detection system that combines **local heuristic rules** with **cloud-based Large Language Models** (Google Gemini / OpenAI) to identify DNS tunneling and data exfiltration attacks. Features an interactive Streamlit dashboard for real-time analysis, evaluation, and full experiment transparency.

## Architecture

```
DNS Records → Schema Mapping → Feature Extraction → Rules-Based Detector
                                                          │
                                              ┌───────────┼───────────┐
                                              │           │           │
                                          Clear      Ambiguous     Clear
                                         Benign    (score 0.3-0.8)  Exfil
                                              │           │           │
                                              │     Privacy Guard     │
                                              │           │           │
                                              │     Cloud LLM API     │
                                              │           │           │
                                              └───────────┼───────────┘
                                                          │
                                                   Merged Verdict
                                                          │
                                               Evaluation & Dashboard
```

## Features

- **Hybrid Two-Tier Detection**: Local rules handle clear-cut cases; cloud AI analyzes ambiguous traffic
- **Multi-Provider LLM Support**: Google Gemini, OpenAI, and deterministic mock for offline testing
- **Privacy-Preserving**: IP pseudonymization, raw query suppression, internal hostname stripping before any cloud API call
- **Explainable Decisions**: Every verdict includes triggered rules, LLM evidence, and decision source
- **Leakage-Safe Evaluation**: Group-based train/test splitting on session IDs
- **Cost Tracking**: Per-call token usage and estimated USD cost
- **Interactive Dashboard**: 5-tab Streamlit UI with Plotly visualizations

## Project Structure

```
├── app.py                          # Streamlit dashboard
├── configs/
│   └── default.yaml                # All thresholds, weights, LLM settings
├── prompts/
│   ├── dns_detection_system.txt    # System prompt for LLM
│   └── prompt_version.txt          # Prompt versioning
├── src/
│   ├── schema_mapping.py           # Auto-detect & normalize CSV schemas
│   ├── data_loader.py              # Robust CSV loading with encoding fallback
│   ├── data_audit.py               # Dataset health & leakage risk audit
│   ├── feature_extraction.py       # Entropy, character stats, structural features
│   ├── rules_detector.py           # Heuristic baseline with explainable rules
│   ├── llm_client.py               # Mock/Gemini/OpenAI with caching & budgets
│   ├── hybrid_router.py            # Two-tier routing engine
│   ├── privacy.py                  # IP hashing, query suppression, hostname stripping
│   └── evaluation.py               # Metrics, comparison, leakage-safe splitting
├── tests/                          # 38 unit tests (pytest)
├── data/
│   ├── generate_synthetic.py       # Reproducible synthetic dataset generator
│   ├── synthetic_demo.csv          # 58-record demo fixture
│   └── README.md                   # Dataset documentation
├── outputs/                        # Exported results
├── requirements.txt
├── .env.example                    # API key template
└── .gitignore
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)
```bash
cp .env.example .env
# Edit .env with your Gemini/OpenAI API keys
# Skip this step to use the mock provider (no keys needed)
```

### 3. Generate Synthetic Demo Data
```bash
python data/generate_synthetic.py
```

### 4. Run Tests
```bash
python -m pytest tests/ -v
```

### 5. Launch Dashboard
```bash
streamlit run app.py
```

## Detection Rules

The heuristic baseline evaluates DNS queries on 5 weighted dimensions:

| Rule | Weight | Threshold | Rationale |
|------|--------|-----------|-----------|
| Shannon Entropy | 0.35 | ≥ 3.85 | Encoded payloads exhibit high information density |
| Subdomain Length | 0.25 | ≥ 28 chars | Tunneling maximizes per-label data throughput |
| Character Distribution | 0.20 | Hex ≥ 0.75 / Low vowel | Binary encoding lacks natural phonetics |
| Label Count | 0.10 | ≥ 4 labels | Chunked payloads use multiple subdomain labels |
| Query Length | 0.10 | ≥ 48 chars | Overall query size anomaly |

Records scoring **< 0.30** → auto-classified benign (no LLM call)  
Records scoring **> 0.80** → auto-classified exfiltration (no LLM call)  
Records scoring **0.30 – 0.80** → routed to cloud LLM for expert analysis

## LLM Providers

| Provider | Requires API Key | Cost | Best For |
|----------|-----------------|------|----------|
| `mock` | No | Free | Offline testing, demos, CI |
| `gemini` | Yes (`GEMINI_API_KEY`) | ~$0.0003/query | Production, low cost |
| `openai` | Yes (`OPENAI_API_KEY`) | ~$0.0006/query | Production, alternative |

## License

This project is for academic/research purposes. Not a production security tool.
