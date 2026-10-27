"""
Live DNS Exfiltration Tester CLI.
Allows analysts to test individual live queries or simulate real-time exfiltration attacks
against the hybrid rules + Cloud AI detection pipeline.
"""
import argparse
import base64
import json
import os
import sys
import base64
import json
import os
from dotenv import load_dotenv

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

from src.hybrid_router import HybridRouter
from src.feature_extraction import FeatureExtractor
from src.rules_detector import RulesDetector


def test_query(domain: str, qtype: str = "A", client_ip: str = "192.168.1.100", provider: str = "gemini"):
    print("=" * 70)
    print("🛡️  LIVE DNS EXFILTRATION DETECTION TEST")
    print("=" * 70)
    print(f"Domain Tested : {domain}")
    print(f"Record Type   : {qtype}")
    print(f"Client IP     : {client_ip}")
    print(f"LLM Provider  : {provider}")
    print("-" * 70)

    # 1. Feature Extraction
    extractor = FeatureExtractor()
    record = {
        "query_name": domain,
        "query_type": qtype,
        "client_ip": client_ip,
        "timestamp": "2026-10-25 20:00:00",
    }
    features = extractor.extract_row_features(record)
    print("\n🔬 1. EXTRACTED LEXICAL & STATISTICAL FEATURES:")
    print(f"   • Shannon Entropy        : {features.get('shannon_entropy', 0):.4f}")
    print(f"   • Query Length           : {features.get('query_length', 0)}")
    print(f"   • Subdomain Length       : {features.get('subdomain_length', 0)}")
    print(f"   • Base64 Char Ratio      : {features.get('b64_ratio', 0):.2%}")
    print(f"   • Hex Character Ratio    : {features.get('hex_ratio', 0):.2%}")
    print(f"   • Vowel/Consonant Ratio  : {features.get('vowel_consonant_ratio', 0):.4f}")

    # 2. Local Heuristic Rules
    detector = RulesDetector()
    rules_res = detector.evaluate_record(features)
    print("\n📋 2. LOCAL HEURISTIC RULES EVALUATION:")
    print(f"   • Base Risk Score        : {rules_res.get('risk_score', 0):.4f}")
    print(f"   • Rules Classification   : {rules_res.get('classification')}")
    triggered = rules_res.get("triggered_rules", [])
    if triggered:
        print("   • Triggered Rules:")
        for r in triggered:
            print(f"     ⚠️  {r}")
    else:
        print("   • Triggered Rules        : None (Clean traffic)")

    # 3. Hybrid Routing & Cloud AI
    print("\n🤖 3. HYBRID ROUTER & CLOUD AI REASONING:")
    router = HybridRouter(llm_provider=provider, routing_policy="ambiguous_and_suspicious")
    verdict = router.analyze_record(record)

    print(f"   • Decision Source        : {verdict.decision_source}")
    print(f"   • Escalated to LLM?      : {'YES' if verdict.routed_to_llm else 'NO (Resolved locally)'}")
    print(f"   • Final Risk Score       : {verdict.risk_score:.4f}")

    if verdict.llm_response and verdict.llm_response.success:
        llm_data = verdict.llm_response.parsed_json or {}
        print(f"   • Cloud AI Verdict       : {llm_data.get('classification')}")
        print(f"   • Cloud AI Evidence      : {', '.join(llm_data.get('evidence', []))}")
        print(f"   • Recommended Action     : {llm_data.get('recommended_next_check')}")
        print(f"   • API Latency            : {verdict.llm_response.latency_ms} ms")
    elif verdict.llm_response and verdict.llm_response.error:
        print(f"   • Cloud AI Status        : {verdict.llm_response.error}")

    # Final Verdict Summary
    print("\n" + "=" * 70)
    final_class = verdict.final_classification.upper()
    if final_class == "LIKELY_EXFILTRATION":
        print(f"🚨 FINAL VERDICT: [{final_class}] — HIGH CONFIDENCE ATTACK CAUGHT!")
    elif final_class == "SUSPICIOUS":
        print(f"⚠️  FINAL VERDICT: [{final_class}] — ANOMALY FLAGGED FOR REVIEW")
    else:
        print(f"✅ FINAL VERDICT: [{final_class}] — NORMAL LEGITIMATE TRAFFIC")
    print("=" * 70 + "\n")


def simulate_exfiltration(secret_text: str, encoding: str = "base64", tunnel_domain: str = "exfil-c2.net", provider: str = "gemini"):
    if encoding == "base64":
        encoded = base64.b64encode(secret_text.encode()).decode().replace("=", "").lower()
    else:
        encoded = secret_text.encode().hex()
    
    # DNS label limit is 63 chars
    chunk = encoded[:50]
    simulated_domain = f"chunk01.{chunk}.{tunnel_domain}"
    print(f"\n[Exfiltration Simulation]")
    print(f"Original Secret : '{secret_text}'")
    print(f"Encoded Payload : '{chunk}'")
    print(f"Crafted Domain  : '{simulated_domain}'")
    test_query(domain=simulated_domain, qtype="TXT" if encoding == "base64" else "A", provider=provider)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live DNS Exfiltration Tester")
    parser.add_argument("--domain", type=str, help="Specific domain name to test live")
    parser.add_argument("--secret", type=str, help="Confidential string to encode and simulate exfiltrating")
    parser.add_argument("--encoding", type=str, default="base64", choices=["base64", "hex"], help="Encoding for secret")
    parser.add_argument("--provider", type=str, default="gemini", choices=["gemini", "openai", "mock"])
    
    args = parser.parse_args()

    if args.secret:
        simulate_exfiltration(args.secret, encoding=args.encoding, provider=args.provider)
    elif args.domain:
        test_query(domain=args.domain, provider=args.provider)
    else:
        # Run demo comparison
        print("No domain or secret specified. Running live side-by-side comparison:\n")
        print(">>> TEST 1: Normal Legitimate Domain (api.github.com)")
        test_query("api.github.com", provider="mock")
        print("\n>>> TEST 2: Active Data Exfiltration Domain (Secret Token Stolen)")
        simulate_exfiltration("CONFIDENTIAL_API_TOKEN_xyz987654321_COMPANY_SECRET", encoding="base64")
