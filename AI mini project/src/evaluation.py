"""
Evaluation module for DNS exfiltration detection.
Computes classification metrics (accuracy, precision, recall, F1, confusion matrix),
generates comparative reports between rules-only vs hybrid pipeline,
and supports leakage-safe group-based train/test splitting.
"""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import GroupShuffleSplit


class Evaluator:
    """Computes and formats classification metrics for DNS detection experiments."""

    @staticmethod
    def compute_metrics(
        y_true: List[int], y_pred: List[int], model_name: str = "model"
    ) -> Dict[str, Any]:
        """Computes standard binary classification metrics."""
        y_t = np.array(y_true)
        y_p = np.array(y_pred)

        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        metrics = {
            "model_name": model_name,
            "accuracy": round(accuracy_score(y_t, y_p), 4),
            "precision": round(precision_score(y_t, y_p, zero_division=0), 4),
            "recall": round(recall_score(y_t, y_p, zero_division=0), 4),
            "f1_score": round(f1_score(y_t, y_p, zero_division=0), 4),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
            "total_samples": len(y_t),
            "positive_samples": int(y_t.sum()),
            "negative_samples": int((y_t == 0).sum()),
        }

        # Derived rates
        metrics["false_positive_rate"] = round(fp / max(fp + tn, 1), 4)
        metrics["false_negative_rate"] = round(fn / max(fn + tp, 1), 4)

        return metrics

    @staticmethod
    def compare_models(
        metrics_list: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Creates a comparison dataframe across multiple model evaluations."""
        rows = []
        for m in metrics_list:
            rows.append({
                "Model": m["model_name"],
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1 Score": m["f1_score"],
                "FP Rate": m["false_positive_rate"],
                "FN Rate": m["false_negative_rate"],
                "TP": m["confusion_matrix"]["true_positive"],
                "FP": m["confusion_matrix"]["false_positive"],
                "TN": m["confusion_matrix"]["true_negative"],
                "FN": m["confusion_matrix"]["false_negative"],
            })
        return pd.DataFrame(rows)

    @staticmethod
    def generate_report_markdown(
        metrics: Dict[str, Any], extra_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generates a formatted markdown evaluation report."""
        cm = metrics["confusion_matrix"]
        lines = [
            f"# Evaluation Report: {metrics['model_name']}",
            "",
            "## Classification Metrics",
            f"| Metric | Value |",
            f"| :--- | :--- |",
            f"| **Accuracy** | {metrics['accuracy']:.4f} |",
            f"| **Precision** | {metrics['precision']:.4f} |",
            f"| **Recall** | {metrics['recall']:.4f} |",
            f"| **F1 Score** | {metrics['f1_score']:.4f} |",
            f"| False Positive Rate | {metrics['false_positive_rate']:.4f} |",
            f"| False Negative Rate | {metrics['false_negative_rate']:.4f} |",
            "",
            "## Confusion Matrix",
            "| | Predicted Benign | Predicted Exfiltration |",
            "| :--- | :---: | :---: |",
            f"| **Actual Benign** | {cm['true_negative']} (TN) | {cm['false_positive']} (FP) |",
            f"| **Actual Exfiltration** | {cm['false_negative']} (FN) | {cm['true_positive']} (TP) |",
            "",
            "## Sample Distribution",
            f"- Total samples: {metrics['total_samples']}",
            f"- Positive (exfiltration): {metrics['positive_samples']}",
            f"- Negative (benign): {metrics['negative_samples']}",
        ]

        if extra_context:
            lines.append("")
            lines.append("## Pipeline Statistics")
            for k, v in extra_context.items():
                if isinstance(v, dict):
                    lines.append(f"### {k}")
                    for kk, vv in v.items():
                        lines.append(f"- {kk}: {vv}")
                else:
                    lines.append(f"- {k}: {v}")

        return "\n".join(lines)


class LeakageSafeSplitter:
    """
    Group-based train/test splitting to prevent data leakage.
    Records from the same session/group never appear in both train and test sets.
    """

    @staticmethod
    def split(
        df: pd.DataFrame,
        group_col: str = "session_id",
        test_size: float = 0.30,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits dataframe by groups to prevent information leakage.
        Falls back to random splitting if group column is missing.
        """
        if group_col not in df.columns or df[group_col].isna().all():
            # Fallback: random split with warning
            from sklearn.model_selection import train_test_split

            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state, stratify=df.get("label")
            )
            return train_df, test_df

        gss = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        groups = df[group_col]
        train_idx, test_idx = next(gss.split(df, groups=groups))

        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        return train_df, test_df

    @staticmethod
    def validate_no_leakage(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        group_col: str = "session_id",
    ) -> Dict[str, Any]:
        """Validates that no groups leak between train and test sets."""
        if group_col not in train_df.columns:
            return {"valid": False, "error": f"Column '{group_col}' not found."}

        train_groups = set(train_df[group_col].dropna().unique())
        test_groups = set(test_df[group_col].dropna().unique())
        overlap = train_groups & test_groups

        return {
            "valid": len(overlap) == 0,
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
            "overlapping_groups": list(overlap),
            "train_size": len(train_df),
            "test_size": len(test_df),
        }
