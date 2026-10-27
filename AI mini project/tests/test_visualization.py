"""
Unit tests for NetworkFlowVisualizer module.
"""
import pandas as pd
import pytest
from src.visualization import NetworkFlowVisualizer, _compute_cubic_bezier, _extract_base_domain


def test_bezier_curve_computation():
    p0 = (0.0, 0.0)
    p1 = (1.0, 2.0)
    p2 = (2.0, 3.0)
    p3 = (3.0, 4.0)
    x, y = _compute_cubic_bezier(p0, p1, p2, p3, n_points=10)
    assert len(x) == 10
    assert len(y) == 10
    assert x[0] == pytest.approx(0.0)
    assert x[-1] == pytest.approx(3.0)
    assert y[0] == pytest.approx(0.0)
    assert y[-1] == pytest.approx(4.0)


def test_base_domain_extraction():
    assert _extract_base_domain("www.google.com") == "google.com"
    assert _extract_base_domain("v100.data.tunnel-gateway.org") == "tunnel-gateway.org"
    assert _extract_base_domain("test.covert-ops-node.biz") == "covert-ops-node.biz"
    assert _extract_base_domain("localhost") == "localhost"
    assert _extract_base_domain("") == "unknown"


def test_generate_flow_graph():
    df = pd.DataFrame([
        {
            "query_id": "Q1",
            "client_ip": "192.168.1.100",
            "query_name": "api.github.com",
            "record_type": "A",
            "label": 0,
        },
        {
            "query_id": "Q2",
            "client_ip": "10.0.4.50",
            "query_name": "data.tunnel-gateway.org",
            "record_type": "TXT",
            "label": 1,
            "attack_type": "dns_tunnel_iodine",
        }
    ])
    fig = NetworkFlowVisualizer.generate_flow_graph(df)
    assert fig is not None
    assert len(fig.data) >= 2


def test_telemetry_hud_metrics():
    df = pd.DataFrame([
        {"client_ip": "192.168.1.1", "query_name": "example.com", "label": 0},
        {"client_ip": "10.0.0.1", "query_name": "bad.tunnel.org", "label": 1},
    ])
    metrics = NetworkFlowVisualizer.get_telemetry_hud_metrics(df)
    assert metrics["total_queries"] == 2
    assert metrics["unique_clients"] == 2
    assert metrics["anomaly_count"] == 1
    assert metrics["service_availability"] == pytest.approx(50.0)
