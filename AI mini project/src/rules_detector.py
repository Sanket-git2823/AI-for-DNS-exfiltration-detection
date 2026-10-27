"""
Transparent, heuristic rules-based baseline detector for DNS exfiltration.
Features configurable thresholds, explainable rule triggers, and risk scoring.
"""
from typing import Dict, Any, List, Optional
import pandas as pd

DEFAULT_RULES_CONFIG = {
    "entropy_threshold": 3.85,
    "subdomain_length_threshold": 28,
    "query_length_threshold": 48,
    "label_count_threshold": 4,
    "hex_ratio_threshold": 0.75,
    "base64_ratio_threshold": 0.85,
    "vowel_consonant_ratio_min": 0.10,
    "score_threshold_suspicious": 0.40,
    "score_threshold_exfiltration": 0.75,
    "weights": {
        "entropy": 0.35,
        "subdomain_length": 0.25,
        "character_distribution": 0.20,
        "label_count": 0.10,
        "query_length": 0.10
    }
}

class RulesDetector:
    """Heuristic detector implementing transparent baseline rules for DNS tunneling detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**DEFAULT_RULES_CONFIG, **(config or {})}

    def evaluate_record(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a single feature dictionary against baseline heuristic rules.
        Returns prediction, composite risk score, and list of triggered rules.
        """
        entropy = float(features.get("shannon_entropy", 0.0))
        subdomain_len = int(features.get("subdomain_length", 0))
        query_len = int(features.get("query_length", 0))
        label_count = int(features.get("label_count", 0))
        hex_ratio = float(features.get("hex_ratio", 0.0))
        vc_ratio = float(features.get("vowel_consonant_ratio", 1.0))
        
        triggered_rules = []
        rule_scores = {}
        
        # Rule 1: High Shannon Entropy
        # Justification: Tunneling protocols pack compressed/encrypted payloads into domain labels,
        # yielding elevated information density relative to natural language.
        if entropy >= self.config["entropy_threshold"]:
            triggered_rules.append(f"High Entropy ({entropy} >= {self.config['entropy_threshold']})")
            rule_scores["entropy"] = min(1.0, (entropy - self.config["entropy_threshold"] + 0.5) / 1.5)
        else:
            rule_scores["entropy"] = 0.0

        # Rule 2: Excessive Subdomain Length
        # Justification: Covert channels maximize throughput by packing maximum data (up to 63 chars per label).
        if subdomain_len >= self.config["subdomain_length_threshold"]:
            triggered_rules.append(f"Long Subdomain ({subdomain_len} chars >= {self.config['subdomain_length_threshold']})")
            rule_scores["subdomain_length"] = min(1.0, subdomain_len / 50.0)
        else:
            rule_scores["subdomain_length"] = 0.0

        # Rule 3: Abnormal Character Distribution (Hex/Base64/Low Vowel)
        # Justification: Encoded binaries/keys exhibit near-absence of phonetic vowel structure or pure hex alphabet.
        is_high_hex = hex_ratio >= self.config["hex_ratio_threshold"]
        is_low_vowel = (subdomain_len > 15) and (vc_ratio < self.config["vowel_consonant_ratio_min"])
        if is_high_hex or is_low_vowel:
            reason = "High Hex Ratio" if is_high_hex else "Low Vowel-to-Consonant Ratio"
            triggered_rules.append(f"Abnormal Character Distribution ({reason})")
            rule_scores["character_distribution"] = 1.0
        else:
            rule_scores["character_distribution"] = 0.0

        # Rule 4: Excessive Label Count (Deep Nesting)
        # Justification: Chunked payloads frequently concatenate multiple subdomain labels.
        if label_count >= self.config["label_count_threshold"]:
            triggered_rules.append(f"Deep Label Hierarchy ({label_count} labels >= {self.config['label_count_threshold']})")
            rule_scores["label_count"] = min(1.0, (label_count - self.config["label_count_threshold"] + 1) / 3.0)
        else:
            rule_scores["label_count"] = 0.0

        # Rule 5: Long Overall Query Length
        if query_len >= self.config["query_length_threshold"]:
            triggered_rules.append(f"Long Query Length ({query_len} >= {self.config['query_length_threshold']})")
            rule_scores["query_length"] = min(1.0, query_len / 80.0)
        else:
            rule_scores["query_length"] = 0.0

        # Compute weighted composite score [0.0 - 1.0]
        weights = self.config.get("weights", DEFAULT_RULES_CONFIG["weights"])
        composite_score = sum(rule_scores[k] * weights.get(k, 0.2) for k in rule_scores)
        composite_score = round(min(1.0, max(0.0, composite_score)), 4)

        # Classification mapping
        if composite_score >= self.config["score_threshold_exfiltration"]:
            classification = "likely_exfiltration"
            binary_prediction = 1
        elif composite_score >= self.config["score_threshold_suspicious"]:
            classification = "suspicious"
            binary_prediction = 1
        else:
            classification = "benign"
            binary_prediction = 0

        return {
            "classification": classification,
            "binary_prediction": binary_prediction,
            "risk_score": composite_score,
            "triggered_rules": triggered_rules,
            "rule_scores": rule_scores,
            "is_ambiguous": (self.config["score_threshold_suspicious"] <= composite_score < self.config["score_threshold_exfiltration"])
        }

    def predict_dataframe(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Evaluates all records in a features dataframe."""
        results = []
        for idx, row in features_df.iterrows():
            eval_res = self.evaluate_record(row.to_dict())
            eval_res["_row_id"] = idx
            results.append(eval_res)
            
        res_df = pd.DataFrame(results).set_index("_row_id")
        return res_df
