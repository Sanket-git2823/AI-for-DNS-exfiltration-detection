"""
Unit tests for rules-based baseline detector.
"""
import pytest
from src.rules_detector import RulesDetector
from src.feature_extraction import FeatureExtractor

def test_benign_evaluation():
    detector = RulesDetector()
    feat = FeatureExtractor.extract_row_features({"query_name": "en.wikipedia.org"})
    result = detector.evaluate_record(feat)
    
    assert result["classification"] == "benign"
    assert result["binary_prediction"] == 0
    assert result["risk_score"] < 0.40
    assert len(result["triggered_rules"]) == 0

def test_exfiltration_evaluation():
    detector = RulesDetector()
    # High entropy + long subdomain + hex ratio
    feat = FeatureExtractor.extract_row_features({
        "query_name": "7b2275736572223a2261646d696e227d89abcdef0123.tunnel-c2.evil.com"
    })
    result = detector.evaluate_record(feat)
    
    assert result["classification"] in {"suspicious", "likely_exfiltration"}
    assert result["binary_prediction"] == 1
    assert result["risk_score"] >= 0.40
    assert len(result["triggered_rules"]) > 0

def test_configurable_thresholds():
    # Make thresholds extremely strict
    detector = RulesDetector(config={
        "entropy_threshold": 1.0,
        "subdomain_length_threshold": 2,
        "score_threshold_suspicious": 0.1
    })
    feat = FeatureExtractor.extract_row_features({"query_name": "mail.google.com"})
    result = detector.evaluate_record(feat)
    assert result["binary_prediction"] == 1
