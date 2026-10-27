"""
Dataset auditing module for DNS exfiltration datasets.
Analyzes column types, missing values, class distributions, duplicates,
leakage risks, and feature feasibility before modeling.
"""
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

class DataAuditor:
    """Audits raw and normalized DNS datasets for research integrity."""

    @staticmethod
    def audit_dataset(
        raw_df: pd.DataFrame, 
        norm_df: Optional[pd.DataFrame] = None,
        source_name: str = "dataset.csv"
    ) -> Dict[str, Any]:
        """
        Runs comprehensive data health, schema, and leakage risk inspection.
        """
        total_rows = len(raw_df)
        total_cols = len(raw_df.columns)
        
        # Missing values
        missing_counts = raw_df.isnull().sum().to_dict()
        missing_pcts = {k: round((v / total_rows) * 100, 2) for k, v in missing_counts.items()}
        
        # Duplicates
        duplicate_rows = int(raw_df.duplicated().sum())
        
        # Data types
        col_types = {k: str(v) for k, v in raw_df.dtypes.items()}
        
        # Check label distribution if available
        label_stats = None
        target_series = None
        if norm_df is not None and "label" in norm_df.columns and norm_df["label"].notna().any():
            target_series = norm_df["label"].dropna()
        elif "label" in raw_df.columns:
            target_series = raw_df["label"].dropna()
            
        if target_series is not None and len(target_series) > 0:
            counts = target_series.value_counts().to_dict()
            total_labeled = len(target_series)
            distribution = {str(k): {"count": int(v), "percentage": round((v / total_labeled) * 100, 2)}
                            for k, v in counts.items()}
            imbalance_ratio = round(max(counts.values()) / max(min(counts.values()), 1), 2) if len(counts) > 1 else 1.0
            label_stats = {
                "available": True,
                "counts": distribution,
                "total_labeled": total_labeled,
                "imbalance_ratio": imbalance_ratio
            }
        else:
            label_stats = {
                "available": False,
                "note": "No ground truth labels detected. Dataset can only be used for inference or rules labeling."
            }

        # Temporal availability
        has_timestamp = False
        time_span = None
        if norm_df is not None and "timestamp" in norm_df.columns and norm_df["timestamp"].notna().any():
            has_timestamp = True
            time_series = norm_df["timestamp"].dropna()
            try:
                parsed = pd.to_datetime(time_series, errors="coerce").dropna()
                if not parsed.empty:
                    time_span = {
                        "min": str(parsed.min()),
                        "max": str(parsed.max()),
                        "duration_seconds": round((parsed.max() - parsed.min()).total_seconds(), 2)
                    }
            except Exception:
                time_span = {"note": "Timestamp present but non-standard format."}

        # Grouping & leakage risks
        has_session = bool(norm_df is not None and "session_id" in norm_df.columns and norm_df["session_id"].notna().any())
        has_ip = bool(norm_df is not None and "client_ip" in norm_df.columns and norm_df["client_ip"].notna().any())
        
        leakage_risk = "HIGH"
        recommended_split = "Random splitting only (WARNING: Potential cross-session feature leakage)"
        if has_session:
            leakage_risk = "LOW"
            recommended_split = "Group-based split on session_id (Leakage-safe across attacks)"
        elif has_ip:
            leakage_risk = "MODERATE"
            recommended_split = "Group-based split on client_ip"

        # Feature availability audit
        feature_feasibility = {
            "name_structure": {
                "feasible": bool(norm_df is not None and "query_name" in norm_df.columns),
                "features": ["query_length", "subdomain_length", "label_count"],
                "note": "Derived from canonical query_name"
            },
            "character_statistics": {
                "feasible": bool(norm_df is not None and "query_name" in norm_df.columns),
                "features": ["shannon_entropy", "vowel_consonant_ratio", "digit_ratio", "hex_ratio"],
                "note": "Derived from query_name character frequency distributions"
            },
            "temporal_dynamics": {
                "feasible": has_timestamp,
                "features": ["inter_query_interval_mean", "inter_query_interval_std", "query_frequency_per_window"],
                "note": "Requires valid timestamp column" if not has_timestamp else "Supported by timestamp"
            },
            "per_host_diversity": {
                "feasible": has_ip,
                "features": ["unique_subdomains_per_host", "unique_query_ratio"],
                "note": "Requires client_ip identifier" if not has_ip else "Supported by client_ip"
            }
        }

        return {
            "source_name": source_name,
            "total_rows": total_rows,
            "total_columns": total_cols,
            "missing_values": missing_counts,
            "missing_percentages": missing_pcts,
            "duplicate_rows": duplicate_rows,
            "column_types": col_types,
            "label_stats": label_stats,
            "has_timestamp": has_timestamp,
            "time_span": time_span,
            "has_session": has_session,
            "has_client_ip": has_ip,
            "leakage_risk": leakage_risk,
            "recommended_split": recommended_split,
            "feature_feasibility": feature_feasibility
        }

    @staticmethod
    def generate_markdown_report(audit: Dict[str, Any]) -> str:
        """Formats the audit dictionary into an academic-grade markdown report."""
        lines = [
            f"# Dataset Audit Report: {audit['source_name']}",
            "",
            "## 1. Volume & Completeness",
            f"- **Total Records**: {audit['total_rows']}",
            f"- **Total Columns**: {audit['total_columns']}",
            f"- **Duplicate Rows**: {audit['duplicate_rows']}",
            "",
            "## 2. Missing Value Summary",
            "| Column | Missing Count | Missing % |",
            "| :--- | :--- | :--- |"
        ]
        for col, cnt in audit["missing_values"].items():
            pct = audit["missing_percentages"].get(col, 0)
            lines.append(f"| `{col}` | {cnt} | {pct}% |")
            
        lines.append("")
        lines.append("## 3. Label Balance & Ground Truth")
        l_stat = audit["label_stats"]
        if l_stat.get("available"):
            lines.append(f"- **Total Labeled**: {l_stat['total_labeled']}")
            lines.append(f"- **Imbalance Ratio**: {l_stat['imbalance_ratio']}:1")
            for lbl, data in l_stat["counts"].items():
                lines.append(f"  - Label `{lbl}`: {data['count']} ({data['percentage']}%)")
        else:
            lines.append(f"- *{l_stat.get('note')}*")
            
        lines.append("")
        lines.append("## 4. Leakage Risk & Splitting Strategy")
        lines.append(f"- **Leakage Risk Level**: **{audit['leakage_risk']}**")
        lines.append(f"- **Recommended Split**: {audit['recommended_split']}")
        
        lines.append("")
        lines.append("## 5. Feature Extraction Feasibility")
        for group, feat in audit["feature_feasibility"].items():
            status = "✅ Supported" if feat["feasible"] else "❌ Unsupported (Fields Missing)"
            lines.append(f"- **{group}**: {status}")
            lines.append(f"  - *Features*: {', '.join(feat['features'])}")
            lines.append(f"  - *Constraint*: {feat['note']}")

        return "\n".join(lines)
