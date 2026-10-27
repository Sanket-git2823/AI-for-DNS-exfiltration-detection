"""
Privacy-preserving data transformations for DNS exfiltration detection.
Implements IP pseudonymization, raw query suppression, and internal hostname stripping
so that sensitive network identifiers are never sent to external cloud AI services.
"""
import hashlib
import re
from typing import Dict, Any, Optional
import pandas as pd


# Heuristic patterns for internal/private hostnames
INTERNAL_PATTERNS = [
    r"\.internal\.",
    r"\.local$",
    r"\.corp\.",
    r"\.lan$",
    r"\.priv\.",
    r"\.intranet\.",
    r"^10\.\d+\.\d+\.\d+$",
    r"^172\.(1[6-9]|2\d|3[01])\.\d+\.\d+$",
    r"^192\.168\.\d+\.\d+$",
]

INTERNAL_RE = re.compile("|".join(INTERNAL_PATTERNS), re.IGNORECASE)


class PrivacyGuard:
    """Applies privacy transformations before data leaves the local system."""

    def __init__(
        self,
        pseudonymize_ips: bool = True,
        suppress_raw_queries: bool = True,
        strip_internal_hostnames: bool = True,
        salt: str = "dns-exfil-research-2026",
    ):
        self.pseudonymize_ips = pseudonymize_ips
        self.suppress_raw_queries = suppress_raw_queries
        self.strip_internal_hostnames = strip_internal_hostnames
        self._salt = salt
        self._ip_cache: Dict[str, str] = {}

    def hash_ip(self, ip: str) -> str:
        """One-way keyed hash of an IP address for consistent pseudonymization."""
        if not ip or pd.isna(ip):
            return "UNKNOWN_HOST"
        ip_str = str(ip).strip()
        if ip_str in self._ip_cache:
            return self._ip_cache[ip_str]
        hashed = hashlib.sha256(f"{self._salt}:{ip_str}".encode()).hexdigest()[:16]
        pseudo = f"HOST-{hashed}"
        self._ip_cache[ip_str] = pseudo
        return pseudo

    def is_internal_hostname(self, hostname: str) -> bool:
        """Checks if a hostname matches internal/private network patterns."""
        return bool(INTERNAL_RE.search(str(hostname)))

    def sanitize_features_for_llm(self, features: Dict[str, Any], raw_query: str = "") -> Dict[str, Any]:
        """
        Produces a sanitized feature dictionary safe for transmission to cloud LLM.
        - Replaces client_ip with pseudonym
        - Strips raw query if configured (sends only derived metrics)
        - Redacts internal hostnames from root_domain
        """
        safe = dict(features)

        if self.pseudonymize_ips and "client_ip" in safe:
            safe["client_ip"] = self.hash_ip(safe["client_ip"])

        if self.strip_internal_hostnames and "root_domain" in safe:
            if self.is_internal_hostname(str(safe.get("root_domain", ""))):
                safe["root_domain"] = "[INTERNAL_REDACTED]"

        if self.suppress_raw_queries:
            safe.pop("raw_query", None)
            safe.pop("query_name", None)
            # Keep only structural/statistical features
        else:
            if raw_query:
                safe["query_name"] = raw_query

        return safe

    def sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies bulk privacy transformations to a full dataframe."""
        out = df.copy()

        if self.pseudonymize_ips and "client_ip" in out.columns:
            out["client_ip"] = out["client_ip"].apply(self.hash_ip)

        if self.strip_internal_hostnames and "root_domain" in out.columns:
            mask = out["root_domain"].apply(lambda x: self.is_internal_hostname(str(x)))
            out.loc[mask, "root_domain"] = "[INTERNAL_REDACTED]"

        if self.suppress_raw_queries and "query_name" in out.columns:
            out = out.drop(columns=["query_name"], errors="ignore")

        return out
