"""
Defensive DNS feature extraction pipeline.
Calculates structural, lexical, entropy, and temporal metrics.
Only extracts features supported by actual data columns without fabricating missing fields.
"""
import math
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

HEX_CHARS = set("0123456789abcdefABCDEF")
BASE64_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/=")
VOWELS = set("aeiouAEIOU")

class FeatureExtractor:
    """Extracts verifiable DNS features from normalized records."""

    @staticmethod
    def calculate_shannon_entropy(text: str) -> float:
        """Calculates Shannon character entropy in bits per character."""
        if not text:
            return 0.0
        length = len(text)
        counts = Counter(text)
        entropy = -sum((cnt / length) * math.log2(cnt / length) for cnt in counts.values())
        return round(entropy, 4)

    @classmethod
    def split_domain(cls, fqdn: str) -> Dict[str, str]:
        """
        Splits FQDN into root domain and subdomain.
        Handles simple 2-level TLD heuristics safely without external network calls.
        """
        clean = fqdn.strip().lower().rstrip(".")
        if not clean:
            return {"fqdn": "", "root_domain": "", "subdomain": ""}
            
        labels = clean.split(".")
        if len(labels) <= 2:
            return {"fqdn": clean, "root_domain": clean, "subdomain": ""}
            
        # Common second-level TLDs heuristic (e.g., .co.uk, .com.au)
        common_sltlds = {"co.uk", "gov.uk", "ac.uk", "com.au", "net.au", "co.jp", "com.br"}
        potential_sltld = f"{labels[-2]}.{labels[-1]}"
        
        if potential_sltld in common_sltlds and len(labels) > 3:
            root_domain = f"{labels[-3]}.{potential_sltld}"
            subdomain = ".".join(labels[:-3])
        else:
            root_domain = f"{labels[-2]}.{labels[-1]}"
            subdomain = ".".join(labels[:-2])
            
        return {"fqdn": clean, "root_domain": root_domain, "subdomain": subdomain}

    @classmethod
    def extract_row_features(cls, row: pd.Series | Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts structural, lexical, and character-entropy features from a single DNS record.
        """
        query_name = str(row.get("query_name") or "").strip().lower().rstrip(".")
        dom_parts = cls.split_domain(query_name)
        subdomain = dom_parts["subdomain"]
        
        labels = query_name.split(".") if query_name else []
        label_count = len(labels)
        query_len = len(query_name)
        subdomain_len = len(subdomain)
        
        # Target string for exfiltration analysis: prioritize subdomain if present, else entire query
        target_str = subdomain if subdomain else query_name
        target_len = len(target_str)
        
        # Character distributions
        if target_len > 0:
            hex_count = sum(1 for c in target_str if c in HEX_CHARS)
            b64_count = sum(1 for c in target_str if c in BASE64_CHARS)
            vowel_count = sum(1 for c in target_str if c in VOWELS)
            consonant_count = sum(1 for c in target_str if c.isalpha() and c not in VOWELS)
            digit_count = sum(1 for c in target_str if c.isdigit())
            
            hex_ratio = round(hex_count / target_len, 4)
            b64_ratio = round(b64_count / target_len, 4)
            digit_ratio = round(digit_count / target_len, 4)
            vowel_ratio = round(vowel_count / target_len, 4)
            vowel_consonant_ratio = round(vowel_count / max(consonant_count, 1), 4)
        else:
            hex_ratio = 0.0
            b64_ratio = 0.0
            digit_ratio = 0.0
            vowel_ratio = 0.0
            vowel_consonant_ratio = 0.0
            
        entropy = cls.calculate_shannon_entropy(target_str)

        features = {
            "query_length": query_len,
            "subdomain_length": subdomain_len,
            "label_count": label_count,
            "shannon_entropy": entropy,
            "hex_ratio": hex_ratio,
            "b64_ratio": b64_ratio,
            "digit_ratio": digit_ratio,
            "vowel_ratio": vowel_ratio,
            "vowel_consonant_ratio": vowel_consonant_ratio,
            "has_subdomain": bool(subdomain_len > 0),
            "root_domain": dom_parts["root_domain"]
        }
        
        # Include optional metadata if present in row
        if "record_type" in row and pd.notna(row["record_type"]):
            features["record_type"] = str(row["record_type"]).upper()
        if "response_code" in row and pd.notna(row["response_code"]):
            features["response_code"] = str(row["response_code"]).upper()
            
        return features

    @classmethod
    def extract_dataset_features(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extracts row-level and optional window-level features across an entire dataframe.
        Returns: (features_df, extraction_metadata)
        """
        records = []
        for idx, row in df.iterrows():
            feat = cls.extract_row_features(row)
            feat["_row_id"] = idx
            records.append(feat)
            
        feat_df = pd.DataFrame(records).set_index("_row_id")
        
        derived_features = list(feat_df.columns)
        missing_features = []
        
        # Check temporal windowing capability
        has_temporal = "timestamp" in df.columns and df["timestamp"].notna().any()
        if not has_temporal:
            missing_features.extend(["inter_query_interval", "window_query_rate"])
        else:
            # Calculate simple inter-query intervals if timestamp can be parsed
            try:
                ts = pd.to_datetime(df["timestamp"], errors="coerce")
                if ts.notna().sum() > 1:
                    time_diffs = ts.diff().dt.total_seconds().fillna(0.0)
                    feat_df["inter_query_interval_sec"] = time_diffs.round(2)
                    derived_features.append("inter_query_interval_sec")
            except Exception:
                missing_features.append("inter_query_interval_sec")
                
        # Check host diversity capability
        has_host = "client_ip" in df.columns and df["client_ip"].notna().any()
        if has_host:
            # Count subdomains per host
            host_subdomain_counts = df.groupby("client_ip")["query_name"].nunique().to_dict()
            feat_df["host_unique_queries"] = df["client_ip"].map(host_subdomain_counts).fillna(1)
            derived_features.append("host_unique_queries")
        else:
            missing_features.append("host_unique_queries")

        metadata = {
            "total_records_processed": len(feat_df),
            "derived_features": derived_features,
            "missing_features": missing_features,
            "temporal_features_enabled": has_temporal,
            "host_diversity_enabled": has_host
        }
        
        return feat_df, metadata
