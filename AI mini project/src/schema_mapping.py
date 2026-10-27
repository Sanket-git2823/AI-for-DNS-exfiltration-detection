"""
Explicit schema mapping and normalization for DNS exfiltration datasets.
Never assumes universal column names; provides heuristic auto-detection with explicit overrides.
"""
from typing import Dict, Optional, Tuple
import pandas as pd
import re

# Canonical field definitions
CANONICAL_FIELDS = {
    "query_name": "Full domain name / DNS query string (e.g. 'sub.example.com')",
    "timestamp": "Timestamp of the query event (ISO, UNIX epoch, or formatted string)",
    "client_ip": "Client or source IP address (e.g. '192.168.1.100')",
    "record_type": "DNS RR type (e.g. 'A', 'TXT', 'AAAA', 'NULL')",
    "response_code": "DNS response code / RCODE (e.g. 'NOERROR', 'NXDOMAIN', '0')",
    "label": "Ground truth label (0/1, 'benign'/'malicious', etc.)",
    "session_id": "Session or stream identifier for group-based leakage prevention",
    "attack_type": "Specific attack or traffic sub-category identifier"
}

REQUIRED_CANONICAL_FIELDS = ["query_name"]

CANDIDATE_PATTERNS = {
    "query_name": [r"^query.*name", r"^qname", r"^domain", r"^query", r"^fqdn", r"^dns_query", r"^name"],
    "timestamp": [r"^time", r"^date", r"^timestamp", r"^ts", r"^datetime"],
    "client_ip": [r"^src.*ip", r"^client.*ip", r"^ip.*src", r"^source.*ip", r"^host", r"^ip"],
    "record_type": [r"^record.*type", r"^qtype", r"^rr.*type", r"^type", r"^dns_type"],
    "response_code": [r"^response.*code", r"^rcode", r"^status", r"^reply.*code"],
    "label": [r"^label", r"^class", r"^is.*attack", r"^is.*exfil", r"^target", r"^malicious"],
    "session_id": [r"^session.*id", r"^stream.*id", r"^flow.*id", r"^group", r"^connection.*id"],
    "attack_type": [r"^attack.*type", r"^scenario", r"^category", r"^sub_label", r"^attack"]
}

class SchemaMapper:
    """Manages mapping from arbitrary dataset columns to canonical project fields."""
    
    def __init__(self, mapping: Optional[Dict[str, Optional[str]]] = None):
        """
        mapping: Dictionary mapping canonical_field -> dataset_column_name.
        """
        self.mapping: Dict[str, Optional[str]] = mapping or {}
        
    @classmethod
    def auto_detect(cls, columns: list[str]) -> "SchemaMapper":
        """Suggests a mapping using regex heuristics on available column names."""
        mapping: Dict[str, Optional[str]] = {}
        used_cols = set()
        
        for canonical, patterns in CANDIDATE_PATTERNS.items():
            matched_col = None
            for pattern in patterns:
                for col in columns:
                    if col in used_cols:
                        continue
                    clean_col = col.strip().lower()
                    if re.search(pattern, clean_col):
                        matched_col = col
                        break
                if matched_col:
                    break
            if matched_col:
                mapping[canonical] = matched_col
                used_cols.add(matched_col)
            else:
                mapping[canonical] = None
                
        return cls(mapping=mapping)

    def validate_mapping(self, df: pd.DataFrame) -> Tuple[bool, list[str]]:
        """Validates that mapped columns actually exist in the dataframe."""
        errors = []
        for req in REQUIRED_CANONICAL_FIELDS:
            target_col = self.mapping.get(req)
            if not target_col:
                errors.append(f"Required canonical field '{req}' is not mapped to any column.")
            elif target_col not in df.columns:
                errors.append(f"Mapped column '{target_col}' for '{req}' does not exist in dataset.")
                
        for canonical, target_col in self.mapping.items():
            if target_col and target_col not in df.columns:
                errors.append(f"Mapped column '{target_col}' for '{canonical}' is missing in dataset.")
                
        return (len(errors) == 0, errors)

    def apply_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Produces a normalized dataframe containing canonical columns.
        Preserves unmapped columns with a prefix 'raw_'.
        """
        valid, errors = self.validate_mapping(df)
        if not valid:
            raise ValueError(f"Schema mapping validation failed: {'; '.join(errors)}")

        norm_df = pd.DataFrame(index=df.index)
        
        for canonical, source_col in self.mapping.items():
            if source_col and source_col in df.columns:
                norm_df[canonical] = df[source_col]
            else:
                norm_df[canonical] = None

        # Clean query names: strip whitespace and trailing root dot
        if "query_name" in norm_df.columns and norm_df["query_name"] is not None:
            norm_df["query_name"] = (
                norm_df["query_name"]
                .astype(str)
                .str.strip()
                .str.lower()
                .str.rstrip(".")
            )

        # Standardize labels to binary 0 / 1 if label column exists
        if norm_df["label"] is not None and not norm_df["label"].isna().all():
            norm_df["label"] = self._normalize_labels(norm_df["label"])

        return norm_df

    @staticmethod
    def _normalize_labels(series: pd.Series) -> pd.Series:
        """Standardizes diverse label formats (0/1, 'benign'/'malicious', etc.) to 0/1."""
        def map_val(v):
            if pd.isna(v):
                return None
            s = str(v).strip().lower()
            if s in {"1", "1.0", "true", "malicious", "attack", "exfiltration", "tunneling", "bad"}:
                return 1
            if s in {"0", "0.0", "false", "benign", "normal", "legitimate", "good"}:
                return 0
            return None
        return series.apply(map_val)
