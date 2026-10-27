"""
Generate reproducible synthetic DNS records fixture for offline testing and demo mode.
Clearly marked as synthetic; not a benchmark.
"""
import csv
import random
from datetime import datetime, timedelta, timezone

def generate_synthetic_data(output_path="data/synthetic_demo.csv", seed=42):
    random.seed(seed)
    base_time = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
    
    records = []
    
    # 1. Benign Web & Cloud traffic (Session SES-BEN-01, SES-BEN-02)
    benign_domains = [
        ("www.google.com", "A", "NOERROR"),
        ("mail.google.com", "A", "NOERROR"),
        ("api.github.com", "A", "NOERROR"),
        ("github.com", "AAAA", "NOERROR"),
        ("raw.githubusercontent.com", "A", "NOERROR"),
        ("login.microsoftonline.com", "A", "NOERROR"),
        ("outlook.office365.com", "A", "NOERROR"),
        ("en.wikipedia.org", "A", "NOERROR"),
        ("upload.wikimedia.org", "A", "NOERROR"),
        ("stackoverflow.com", "A", "NOERROR"),
        ("cdn.sstatic.net", "A", "NOERROR"),
        ("slack.com", "A", "NOERROR"),
        ("app.slack.com", "A", "NOERROR"),
        ("edge.microsoft.com", "A", "NOERROR"),
        ("weather.com", "A", "NOERROR"),
        ("news.ycombinator.com", "A", "NOERROR"),
        ("docs.python.org", "A", "NOERROR"),
        ("pypi.org", "A", "NOERROR"),
        ("registry.npmjs.org", "A", "NOERROR"),
        ("fonts.googleapis.com", "A", "NOERROR"),
        ("fonts.gstatic.com", "A", "NOERROR"),
        ("code.jquery.com", "A", "NOERROR"),
        ("cdnjs.cloudflare.com", "A", "NOERROR"),
        ("portal.azure.com", "A", "NOERROR"),
        ("aws.amazon.com", "A", "NOERROR")
    ]
    
    for i, (dom, rtype, rcode) in enumerate(benign_domains):
        dt = base_time + timedelta(seconds=i * random.randint(3, 18))
        session_id = "SES-BEN-01" if i < 15 else "SES-BEN-02"
        records.append({
            "query_id": f"SYNTH-{len(records)+1:03d}",
            "timestamp": dt.isoformat(),
            "client_ip": "192.168.1.105" if session_id == "SES-BEN-01" else "192.168.1.112",
            "query_name": dom,
            "record_type": rtype,
            "response_code": rcode,
            "session_id": session_id,
            "attack_type": "benign_web",
            "label": 0,
            "is_synthetic": True
        })
        
    # 2. Benign Challenge/CDN/DKIM cases (Session SES-BEN-03)
    benign_challenges = [
        ("d3g4j8k9x2w1.cloudfront.net", "A", "NOERROR", "benign_cdn"),
        ("a1879.g.akamai.net", "A", "NOERROR", "benign_cdn"),
        ("k1._domainkey.company-secure.com", "TXT", "NOERROR", "benign_dkim"),
        ("v=spf1.include._spf.google.com", "TXT", "NOERROR", "benign_spf"),
        ("telemetry-v4-prod-us-east-1.internal-cloud.net", "A", "NOERROR", "benign_telemetry"),
        ("s3-accelerate.dualstack.amazonaws.com", "A", "NOERROR", "benign_cdn"),
        ("assets-global.website-files.com", "A", "NOERROR", "benign_cdn"),
        ("eu-central-1.elasticbeanstalk.com", "A", "NOERROR", "benign_cloud"),
        ("ocsp.digicert.com", "A", "NOERROR", "benign_cert"),
        ("crl3.digicert.com", "A", "NOERROR", "benign_cert")
    ]
    for i, (dom, rtype, rcode, atype) in enumerate(benign_challenges):
        dt = base_time + timedelta(seconds=350 + i * random.randint(4, 15))
        records.append({
            "query_id": f"SYNTH-{len(records)+1:03d}",
            "timestamp": dt.isoformat(),
            "client_ip": "192.168.1.115",
            "query_name": dom,
            "record_type": rtype,
            "response_code": rcode,
            "session_id": "SES-BEN-03",
            "attack_type": atype,
            "label": 0,
            "is_synthetic": True
        })
        
    # 3. Malicious DNS Tunneling (Iodine-style, Session SES-EXF-01)
    iodine_subdomains = [
        "v100.c7a8b9e0f1.data.tunnel-gateway.org",
        "v100.d2e3f4a5b6.data.tunnel-gateway.org",
        "v100.89abcdef0123.data.tunnel-gateway.org",
        "v100.456789abcdef.data.tunnel-gateway.org",
        "v100.feadcba98765.data.tunnel-gateway.org",
        "v100.1234567890ab.data.tunnel-gateway.org",
        "v100.cdef01234567.data.tunnel-gateway.org",
        "v100.89abcdef4321.data.tunnel-gateway.org"
    ]
    for i, dom in enumerate(iodine_subdomains):
        dt = base_time + timedelta(seconds=600 + i * 2)  # High frequency
        records.append({
            "query_id": f"SYNTH-{len(records)+1:03d}",
            "timestamp": dt.isoformat(),
            "client_ip": "10.0.4.52",
            "query_name": dom,
            "record_type": "NULL" if i % 2 == 0 else "TXT",
            "response_code": "NOERROR",
            "session_id": "SES-EXF-01",
            "attack_type": "dns_tunnel_iodine",
            "label": 1,
            "is_synthetic": True
        })
        
    # 4. Malicious Base64 Data Exfiltration (Session SES-EXF-02)
    b64_exfil = [
        "eyJhZG1pbiI6ICJ0cnVlIiwgInRva2VuIjogInhzOTg3In0=.exf.research-cloud-hub.cc",
        "dXNlcm5hbWU9cm9vdDtwYXNzd29yZD1TZWNyZXQxMjMh.exf.research-cloud-hub.cc",
        "Y29uZmlnLWRhdGEtcGF5bG9hZC1jaHVuay0wMQ==.exf.research-cloud-hub.cc",
        "Y29uZmlnLWRhdGEtcGF5bG9hZC1jaHVuay0wMg==.exf.research-cloud-hub.cc",
        "Y29uZmlnLWRhdGEtcGF5bG9hZC1jaHVuay0wMw==.exf.research-cloud-hub.cc",
        "a2V5PWY4OTM4OTIwMzljYTkyODNiYzkwMTI4ZGMz.exf.research-cloud-hub.cc",
        "aHR0cHM6Ly92YXVsdC5pbnRlcm5hbC5jb21wYW55LmV4.exf.research-cloud-hub.cc",
        "bXlwYXNzdzByZHRleHQxMjM0NTY3ODkwOTg3NjU0.exf.research-cloud-hub.cc"
    ]
    for i, dom in enumerate(b64_exfil):
        dt = base_time + timedelta(seconds=700 + i * random.randint(1, 3))
        records.append({
            "query_id": f"SYNTH-{len(records)+1:03d}",
            "timestamp": dt.isoformat(),
            "client_ip": "10.0.4.88",
            "query_name": dom,
            "record_type": "TXT",
            "response_code": "NOERROR",
            "session_id": "SES-EXF-02",
            "attack_type": "dns_exfil_base64",
            "label": 1,
            "is_synthetic": True
        })

    # 5. Malicious Hex C2 Beaconing (Session SES-EXF-03)
    hex_beacons = [
        "4a8f9b2c1d0e3f5a7b9c.c2.covert-ops-node.biz",
        "5b9c0d1e2f3a4b6c8d0e.c2.covert-ops-node.biz",
        "6c0d1e2f3a4b5c7d9e1f.c2.covert-ops-node.biz",
        "7d1e2f3a4b5c6d8e0f2a.c2.covert-ops-node.biz",
        "8e2f3a4b5c6d7e9f1a3b.c2.covert-ops-node.biz",
        "9f3a4b5c6d7e8f0a2b4c.c2.covert-ops-node.biz",
        "a04b5c6d7e8f9a1b3c5d.c2.covert-ops-node.biz"
    ]
    for i, dom in enumerate(hex_beacons):
        dt = base_time + timedelta(seconds=850 + i * 10)  # Regular heartbeat interval
        records.append({
            "query_id": f"SYNTH-{len(records)+1:03d}",
            "timestamp": dt.isoformat(),
            "client_ip": "10.0.4.99",
            "query_name": dom,
            "record_type": "A",
            "response_code": "NXDOMAIN" if i % 3 == 0 else "NOERROR",
            "session_id": "SES-EXF-03",
            "attack_type": "dns_c2_beacon",
            "label": 1,
            "is_synthetic": True
        })

    keys = list(records[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)

    print(f"Generated {len(records)} synthetic records in {output_path}")

if __name__ == "__main__":
    generate_synthetic_data()
