# DNS Datasets Documentation

## Synthetic Demonstration Dataset (`synthetic_demo.csv`)

> **RESEARCH INTEGRITY NOTICE**:
> `synthetic_demo.csv` is a **synthetic defense demonstration fixture** created exclusively for offline testing, code validation, and interactive UI verification.
> It **MUST NOT** be presented or reported as a real-world benchmark in scientific publications or evaluations.

### Fields in `synthetic_demo.csv`:
- `query_id`: Unique record identifier (e.g., `SYNTH-001`).
- `timestamp`: ISO-8601 UTC timestamp.
- `client_ip`: Simulated client IP address (used for session grouping and privacy testing).
- `query_name`: Fully Qualified Domain Name (FQDN) queried.
- `record_type`: DNS Resource Record Type (`A`, `AAAA`, `TXT`, `CNAME`).
- `response_code`: DNS RCODE (`NOERROR`, `NXDOMAIN`).
- `session_id`: Attack/benign session identifier (enables group-based, leakage-safe train/test splits).
- `attack_type`: Ground truth scenario (`benign_web`, `benign_cdn`, `benign_spf`, `dns_tunnel_iodine`, `dns_exfil_base64`, `dns_c2_beacon`).
- `label`: Binary classification target (`0` = benign, `1` = exfiltration/tunneling).
- `is_synthetic`: Constant boolean `True`.

### Public Benchmark Datasets (For Rigorous Research)
For empirical research evaluations, refer to the following peer-reviewed public datasets:
1. **CIC-Bell-DNS-EXF-2021**
   - Source: Canadian Institute for Cybersecurity, University of New Brunswick & Bell Canada.
   - URL: https://www.unb.ca/cic/datasets/dns-exf-2021.html
   - Characteristics: Captures real DNS tunneling tools (Iodine, DNSteal, DNS2TCP, etc.) with labeled PCAPs and flow statistics.
2. **DNSTunnel2026**
   - Source: Zenodo Repository (https://zenodo.org/records/20137065).
   - Contains high-volume benign and tunneling traffic under diverse network conditions.
