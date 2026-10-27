"""
Unit tests for evaluation module.
"""
import pytest
import pandas as pd
from src.evaluation import Evaluator, LeakageSafeSplitter


def test_compute_metrics_perfect():
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1]
    m = Evaluator.compute_metrics(y_true, y_pred, model_name="perfect_model")
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1_score"] == 1.0
    assert m["confusion_matrix"]["true_positive"] == 3
    assert m["confusion_matrix"]["false_positive"] == 0


def test_compute_metrics_imperfect():
    y_true = [0, 0, 1, 1, 1, 0]
    y_pred = [0, 1, 1, 0, 1, 0]
    m = Evaluator.compute_metrics(y_true, y_pred, model_name="imperfect")
    assert 0 < m["accuracy"] < 1.0
    assert m["confusion_matrix"]["false_positive"] == 1
    assert m["confusion_matrix"]["false_negative"] == 1


def test_compare_models():
    m1 = Evaluator.compute_metrics([0, 1, 1], [0, 1, 1], "model_a")
    m2 = Evaluator.compute_metrics([0, 1, 1], [1, 1, 0], "model_b")
    comp = Evaluator.compare_models([m1, m2])
    assert len(comp) == 2
    assert list(comp["Model"]) == ["model_a", "model_b"]


def test_generate_report_markdown():
    m = Evaluator.compute_metrics([0, 0, 1, 1], [0, 1, 1, 1], "test_model")
    report = Evaluator.generate_report_markdown(m)
    assert "# Evaluation Report" in report
    assert "test_model" in report
    assert "Confusion Matrix" in report


def test_leakage_safe_split():
    df = pd.DataFrame({
        "query_name": [f"q{i}.com" for i in range(20)],
        "session_id": ["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5,
        "label": [0] * 10 + [1] * 10,
    })
    train, test = LeakageSafeSplitter.split(df, group_col="session_id", test_size=0.3)
    assert len(train) + len(test) == 20
    
    validation = LeakageSafeSplitter.validate_no_leakage(train, test, "session_id")
    assert validation["valid"] is True
    assert len(validation["overlapping_groups"]) == 0


def test_leakage_safe_split_fallback():
    df = pd.DataFrame({
        "query_name": [f"q{i}.com" for i in range(20)],
        "label": [0] * 10 + [1] * 10,
    })
    # No session_id column -> fallback to random split
    train, test = LeakageSafeSplitter.split(df, group_col="session_id", test_size=0.3)
    assert len(train) + len(test) == 20
