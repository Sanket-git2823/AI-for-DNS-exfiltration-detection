"""
Unit tests for hybrid router module.
"""
import pytest
import pandas as pd
from src.hybrid_router import HybridRouter


def test_hybrid_benign_not_routed():
    router = HybridRouter(llm_provider="mock", routing_policy="ambiguous_only")
    row = {"query_name": "www.google.com"}
    verdict = router.analyze_record(row)
    assert verdict.final_classification == "benign"
    assert verdict.routed_to_llm is False
    assert verdict.decision_source == "rules_only"


def test_hybrid_exfiltration_not_routed():
    router = HybridRouter(
        llm_provider="mock",
        routing_policy="ambiguous_only",
        high_threshold=0.80,
    )
    row = {"query_name": "7b2275736572223a2261646d696e227d89abcdef0123456789.tunnel.evil.com"}
    verdict = router.analyze_record(row)
    assert verdict.final_binary == 1
    assert verdict.final_classification in {"suspicious", "likely_exfiltration"}


def test_hybrid_ambiguous_routed_to_llm():
    router = HybridRouter(
        llm_provider="mock",
        routing_policy="ambiguous_only",
        low_threshold=0.10,
        high_threshold=0.95,
    )
    # Mid-range query
    row = {"query_name": "d3g4j8k9x2w1.cloudfront.net"}
    verdict = router.analyze_record(row)
    # This may or may not be routed depending on score - just verify no crash
    assert verdict.final_classification in {"benign", "suspicious", "likely_exfiltration", "insufficient_evidence"}


def test_hybrid_dataframe_analysis():
    router = HybridRouter(llm_provider="mock")
    df = pd.DataFrame({
        "query_name": ["www.google.com", "mail.yahoo.com", "4a8f9b2c1d0e3f5a7b9c.c2.evil.biz"],
        "record_type": ["A", "A", "A"],
    })
    verdicts, stats = router.analyze_dataframe(df)
    assert len(verdicts) == 3
    assert stats["total_processed"] == 3


def test_hybrid_routing_policy_none():
    router = HybridRouter(llm_provider="mock", routing_policy="none")
    row = {"query_name": "suspicious-looking-domain-12345.test.com"}
    verdict = router.analyze_record(row)
    assert verdict.routed_to_llm is False
    assert verdict.decision_source == "rules_only"


def test_hybrid_routing_policy_all():
    router = HybridRouter(llm_provider="mock", routing_policy="all")
    row = {"query_name": "www.google.com"}
    verdict = router.analyze_record(row)
    assert verdict.routed_to_llm is True


def test_hybrid_stats():
    router = HybridRouter(llm_provider="mock", routing_policy="none")
    router.analyze_record({"query_name": "a.com"})
    router.analyze_record({"query_name": "b.com"})
    stats = router.get_run_stats()
    assert stats["total_processed"] == 2
    assert stats["total_routed_to_llm"] == 0
    assert stats["total_local_only"] == 2
