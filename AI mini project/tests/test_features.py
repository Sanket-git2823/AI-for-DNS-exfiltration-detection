"""
Unit tests for feature extraction module.
"""
import pytest
from src.feature_extraction import FeatureExtractor

def test_shannon_entropy():
    # Empty string has 0 entropy
    assert FeatureExtractor.calculate_shannon_entropy("") == 0.0
    
    # Uniform repetitive character has 0 entropy
    assert FeatureExtractor.calculate_shannon_entropy("aaaaaaaaaa") == 0.0
    
    # Equal distribution of 2 characters has entropy = 1.0
    assert FeatureExtractor.calculate_shannon_entropy("abababab") == 1.0
    
    # High entropy string
    high_ent = FeatureExtractor.calculate_shannon_entropy("7b2275736572223a2261646d696e227d")
    assert high_ent > 3.0

def test_split_domain():
    # Simple 2-label domain
    parts = FeatureExtractor.split_domain("google.com")
    assert parts["root_domain"] == "google.com"
    assert parts["subdomain"] == ""

    # Multi-label domain
    parts2 = FeatureExtractor.split_domain("sub.data.example.com")
    assert parts2["root_domain"] == "example.com"
    assert parts2["subdomain"] == "sub.data"

    # Trailing dot handling
    parts3 = FeatureExtractor.split_domain("api.github.com.")
    assert parts3["root_domain"] == "github.com"
    assert parts3["subdomain"] == "api"

def test_extract_row_features():
    # Normal web query
    normal_row = {"query_name": "www.google.com"}
    feat = FeatureExtractor.extract_row_features(normal_row)
    assert feat["query_length"] == 14
    assert feat["subdomain_length"] == 3
    assert feat["label_count"] == 3
    assert feat["has_subdomain"] is True
    assert feat["shannon_entropy"] < 3.0

    # Exfiltration query
    exf_row = {"query_name": "aGVsbG8td29ybGQtdGhpcy1pcy1leGZpbHRyYXRpb24=.tunnel.evilcorp.org"}
    feat_exf = FeatureExtractor.extract_row_features(exf_row)
    assert feat_exf["subdomain_length"] > 30
    assert feat_exf["shannon_entropy"] > 3.5
    assert feat_exf["b64_ratio"] > 0.8
