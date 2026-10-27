"""
Unit tests for LLM client module (mock provider only - no API keys needed).
"""
import pytest
from src.llm_client import MockLLMProvider, LLMClient, ResponseCache, _extract_json_from_text


def test_extract_json_plain():
    text = '{"classification": "benign", "evidence": []}'
    result = _extract_json_from_text(text)
    assert result is not None
    assert result["classification"] == "benign"


def test_extract_json_with_fences():
    text = '```json\n{"classification": "suspicious"}\n```'
    result = _extract_json_from_text(text)
    assert result is not None
    assert result["classification"] == "suspicious"


def test_extract_json_with_surrounding_prose():
    text = 'Here is my analysis:\n{"classification": "benign", "evidence": ["low entropy"]}\nEnd.'
    result = _extract_json_from_text(text)
    assert result is not None
    assert result["classification"] == "benign"


def test_mock_provider_benign():
    provider = MockLLMProvider()
    features = {"shannon_entropy": 1.5, "subdomain_length": 3, "hex_ratio": 0.1, "risk_score": 0.05}
    resp = provider.analyze(features, "system prompt")
    assert resp.success
    assert resp.parsed_json["classification"] in {"benign", "insufficient_evidence"}
    assert resp.provider == "mock"


def test_mock_provider_exfiltration():
    provider = MockLLMProvider()
    features = {"shannon_entropy": 4.5, "subdomain_length": 40, "hex_ratio": 0.9, "risk_score": 0.85}
    resp = provider.analyze(features, "system prompt")
    assert resp.success
    assert resp.parsed_json["classification"] == "likely_exfiltration"


def test_llm_client_mock_mode():
    client = LLMClient(provider="mock", max_calls=10)
    features = {"shannon_entropy": 2.0, "subdomain_length": 5, "hex_ratio": 0.2, "risk_score": 0.1}
    resp = client.analyze(features)
    assert resp.success
    stats = client.get_usage_stats()
    assert stats["total_calls"] == 1
    assert stats["total_cost_usd"] == 0.0


def test_llm_client_budget_exhaustion():
    client = LLMClient(provider="mock", max_calls=2)
    features = {"shannon_entropy": 2.0, "subdomain_length": 5, "hex_ratio": 0.2, "risk_score": 0.1}
    client.analyze(features)
    client.analyze({"shannon_entropy": 3.0, "subdomain_length": 10, "hex_ratio": 0.3, "risk_score": 0.2})
    # Third call should be blocked (first 2 might be cached, but unique features ensure 2 real calls)
    resp3 = client.analyze({"shannon_entropy": 4.0, "subdomain_length": 20, "hex_ratio": 0.5, "risk_score": 0.5})
    assert resp3.success is False
    assert "budget" in resp3.error.lower()


def test_response_cache():
    cache = ResponseCache(enabled=True, max_size=5)
    from src.llm_client import LLMResponse
    resp = LLMResponse(
        raw_text='{"test": true}', parsed_json={"test": True},
        provider="mock", model="test", latency_ms=1.0
    )
    features = {"a": 1, "b": 2}
    cache.put(features, "mock", resp)
    cached = cache.get(features, "mock")
    assert cached is not None
    assert cached.cached is True

    # Different features should miss
    assert cache.get({"a": 99}, "mock") is None
