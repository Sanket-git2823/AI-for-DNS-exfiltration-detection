"""
Hybrid routing engine that combines local rules-based detection with cloud LLM analysis.
Routes only ambiguous cases to the LLM, enforces API budgets, and merges verdicts
with full transparency on which system produced each decision.
"""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import time

from src.feature_extraction import FeatureExtractor
from src.rules_detector import RulesDetector
from src.llm_client import LLMClient, LLMResponse
from src.privacy import PrivacyGuard


class HybridVerdict:
    """Container for a single record's hybrid analysis result."""

    def __init__(
        self,
        query_name: str,
        rules_result: Dict[str, Any],
        llm_response: Optional[LLMResponse],
        final_classification: str,
        final_binary: int,
        decision_source: str,
        routed_to_llm: bool,
        risk_score: float,
    ):
        self.query_name = query_name
        self.rules_result = rules_result
        self.llm_response = llm_response
        self.final_classification = final_classification
        self.final_binary = final_binary
        self.decision_source = decision_source
        self.routed_to_llm = routed_to_llm
        self.risk_score = risk_score

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "query_name": self.query_name,
            "final_classification": self.final_classification,
            "final_binary_prediction": self.final_binary,
            "risk_score": self.risk_score,
            "decision_source": self.decision_source,
            "routed_to_llm": self.routed_to_llm,
            "rules_classification": self.rules_result.get("classification"),
            "rules_triggered": self.rules_result.get("triggered_rules", []),
        }
        if self.llm_response and self.llm_response.success:
            llm_json = self.llm_response.parsed_json or {}
            d["llm_classification"] = llm_json.get("classification")
            d["llm_evidence"] = llm_json.get("evidence", [])
            d["llm_limitations"] = llm_json.get("limitations", [])
            d["llm_next_check"] = llm_json.get("recommended_next_check")
            d["llm_latency_ms"] = self.llm_response.latency_ms
            d["llm_cached"] = self.llm_response.cached
        elif self.llm_response and self.llm_response.error:
            d["llm_error"] = self.llm_response.error
        return d


# Classification priority for LLM override merging
_CLASSIFICATION_RANK = {
    "likely_exfiltration": 4,
    "suspicious": 3,
    "insufficient_evidence": 2,
    "benign": 1,
}


class HybridRouter:
    """
    Orchestrates the two-tier detection pipeline:
    1. All records pass through the local rules-based detector.
    2. Records in the ambiguous score range are routed to the cloud LLM for expert analysis.
    3. Final verdict merges both signals with configurable override policy.
    """

    def __init__(
        self,
        rules_config: Optional[Dict[str, Any]] = None,
        llm_provider: str = "mock",
        routing_policy: str = "ambiguous_only",
        low_threshold: float = 0.30,
        high_threshold: float = 0.80,
        max_api_calls: int = 50,
        privacy_config: Optional[Dict[str, bool]] = None,
        cache_enabled: bool = True,
        system_prompt_path: str = "prompts/dns_detection_system.txt",
    ):
        self.rules = RulesDetector(config=rules_config)
        self.llm = LLMClient(
            provider=llm_provider,
            cache_enabled=cache_enabled,
            max_calls=max_api_calls,
            system_prompt_path=system_prompt_path,
        )
        self.privacy = PrivacyGuard(**(privacy_config or {}))
        self.routing_policy = routing_policy
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.extractor = FeatureExtractor()

        # Tracking
        self._total_processed = 0
        self._total_routed = 0
        self._total_local_only = 0

    def _should_route_to_llm(self, rules_result: Dict[str, Any]) -> bool:
        """Determines if a record should be escalated to the LLM based on routing policy."""
        score = rules_result.get("risk_score", 0.0)

        if self.routing_policy == "ambiguous_only":
            return self.low_threshold <= score < self.high_threshold
        elif self.routing_policy == "ambiguous_and_suspicious":
            return score >= self.low_threshold
        elif self.routing_policy == "all":
            return True
        elif self.routing_policy == "none":
            return False
        else:
            return self.low_threshold <= score < self.high_threshold

    def _merge_verdicts(
        self, rules_result: Dict[str, Any], llm_response: Optional[LLMResponse]
    ) -> Tuple[str, int]:
        """
        Merges rules-based and LLM classifications.
        Policy: LLM verdict overrides rules for ambiguous cases if LLM returns a valid result.
        For clear-cut cases, rules verdict stands.
        """
        rules_class = rules_result.get("classification", "benign")

        if llm_response and llm_response.success and llm_response.parsed_json:
            llm_class = llm_response.parsed_json.get("classification", "")
            if llm_class in _CLASSIFICATION_RANK:
                # LLM overrides for ambiguous cases
                final = llm_class
            else:
                final = rules_class
        else:
            final = rules_class

        binary = 1 if final in {"suspicious", "likely_exfiltration"} else 0
        return final, binary

    def analyze_record(
        self, row: Dict[str, Any], include_query_name: bool = True
    ) -> HybridVerdict:
        """Analyzes a single DNS record through the full hybrid pipeline."""
        query_name = str(row.get("query_name", ""))

        # Step 1: Extract features
        features = self.extractor.extract_row_features(row)

        # Step 2: Rules-based evaluation
        rules_result = self.rules.evaluate_record(features)

        # Step 3: Determine LLM routing
        llm_response = None
        routed = self._should_route_to_llm(rules_result)

        if routed:
            # Step 3a: Privacy-sanitize features before sending
            safe_features = self.privacy.sanitize_features_for_llm(
                {**features, **rules_result}, raw_query=query_name
            )
            safe_features["risk_score"] = rules_result["risk_score"]
            safe_features["rules_classification"] = rules_result["classification"]
            safe_features["triggered_rules"] = rules_result["triggered_rules"]

            llm_response = self.llm.analyze(safe_features)
            self._total_routed += 1
        else:
            self._total_local_only += 1

        self._total_processed += 1

        # Step 4: Merge verdicts
        final_class, final_binary = self._merge_verdicts(rules_result, llm_response)
        source = "hybrid_llm" if routed and llm_response and llm_response.success else "rules_only"

        return HybridVerdict(
            query_name=query_name,
            rules_result=rules_result,
            llm_response=llm_response,
            final_classification=final_class,
            final_binary=final_binary,
            decision_source=source,
            routed_to_llm=routed,
            risk_score=rules_result["risk_score"],
        )

    def analyze_dataframe(
        self, df: pd.DataFrame, progress_callback=None
    ) -> Tuple[List[HybridVerdict], Dict[str, Any]]:
        """
        Analyzes an entire dataframe through the hybrid pipeline.
        Returns list of verdicts and aggregate statistics.
        """
        verdicts = []
        total = len(df)

        for i, (idx, row) in enumerate(df.iterrows()):
            verdict = self.analyze_record(row.to_dict())
            verdicts.append(verdict)

            if progress_callback and (i + 1) % max(1, total // 20) == 0:
                progress_callback((i + 1) / total)

        stats = self.get_run_stats()
        return verdicts, stats

    def get_run_stats(self) -> Dict[str, Any]:
        """Returns aggregate pipeline statistics."""
        llm_stats = self.llm.get_usage_stats()
        return {
            "total_processed": self._total_processed,
            "total_routed_to_llm": self._total_routed,
            "total_local_only": self._total_local_only,
            "routing_rate_pct": round(
                (self._total_routed / max(self._total_processed, 1)) * 100, 2
            ),
            "routing_policy": self.routing_policy,
            "llm_usage": llm_stats,
        }
