"""
Unit tests for privacy guard module.
"""
import pytest
from src.privacy import PrivacyGuard


def test_ip_pseudonymization():
    guard = PrivacyGuard(pseudonymize_ips=True)
    pseudo = guard.hash_ip("192.168.1.100")
    assert pseudo.startswith("HOST-")
    assert len(pseudo) == 21  # HOST- + 16 hex chars
    # Same IP should produce same hash (consistency)
    assert guard.hash_ip("192.168.1.100") == pseudo
    # Different IP should produce different hash
    assert guard.hash_ip("10.0.0.1") != pseudo


def test_internal_hostname_detection():
    guard = PrivacyGuard()
    assert guard.is_internal_hostname("server.internal.corp.com") is True
    assert guard.is_internal_hostname("mypc.local") is True
    assert guard.is_internal_hostname("host.lan") is True
    assert guard.is_internal_hostname("www.google.com") is False
    assert guard.is_internal_hostname("api.github.com") is False


def test_sanitize_features_suppresses_query():
    guard = PrivacyGuard(suppress_raw_queries=True, pseudonymize_ips=True)
    features = {
        "query_name": "secret.tunnel.evil.com",
        "client_ip": "10.0.4.52",
        "shannon_entropy": 4.2,
        "subdomain_length": 30,
    }
    safe = guard.sanitize_features_for_llm(features, raw_query="secret.tunnel.evil.com")
    assert "query_name" not in safe
    assert safe["client_ip"].startswith("HOST-")
    assert safe["shannon_entropy"] == 4.2


def test_sanitize_features_redacts_internal():
    guard = PrivacyGuard(strip_internal_hostnames=True, suppress_raw_queries=False)
    features = {
        "root_domain": "db.internal.corp.com",
        "shannon_entropy": 2.0,
    }
    safe = guard.sanitize_features_for_llm(features)
    assert safe["root_domain"] == "[INTERNAL_REDACTED]"


def test_sanitize_features_preserves_external():
    guard = PrivacyGuard(strip_internal_hostnames=True)
    features = {"root_domain": "google.com", "shannon_entropy": 1.5}
    safe = guard.sanitize_features_for_llm(features)
    assert safe["root_domain"] == "google.com"
