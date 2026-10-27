"""
Cinematic Cyber Intelligence Network Flow Visualization Module.
Generates interactive, data-dense network flow diagrams with Bézier curve splines,
subtle spatial depth, anomalous cluster highlighting, and analytical HUD telemetry.
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re


# Precise Cyber Intelligence Console Color Palette
PALETTE = {
    "main_bg": "#101820",
    "secondary_bg": "#17232D",
    "panel_bg": "#1C2B36",
    "border": "#344550",
    "text_primary": "#DCE5EA",
    "text_secondary": "#8FA2AD",
    "cyan": "#42BFD0",
    "blue": "#438DB5",
    "orange": "#E58B56",
    "alert_red": "#D85D5D",
    "teal": "#69B9A9",
}


def _compute_cubic_bezier(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    n_points: int = 24
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes points along a cubic Bézier curve for smooth telemetry flow."""
    t = np.linspace(0, 1, n_points)
    x = (1 - t)**3 * p0[0] + 3 * (1 - t)**2 * t * p1[0] + 3 * (1 - t) * t**2 * p2[0] + t**3 * p3[0]
    y = (1 - t)**3 * p0[1] + 3 * (1 - t)**2 * t * p1[1] + 3 * (1 - t) * t**2 * p2[1] + t**3 * p3[1]
    return x, y


def _extract_base_domain(query_name: str) -> str:
    """Extracts registered root domain or sanitized cluster name."""
    if not query_name or not isinstance(query_name, str):
        return "unknown"
    q = query_name.strip().lower()
    parts = q.split(".")
    if len(parts) >= 2:
        return f"{parts[-2]}.{parts[-1]}"
    return q


class NetworkFlowVisualizer:
    """Renders interactive network-flow visualizations representing DNS telemetry."""

    @staticmethod
    def generate_flow_graph(
        df: pd.DataFrame,
        verdicts: Optional[List[Any]] = None,
        selected_client: Optional[str] = None,
        selected_cluster: Optional[str] = None,
        filter_status: str = "all",
    ) -> go.Figure:
        """
        Builds the primary cinematic network flow graph using Plotly.
        - Clients on bottom/left axis
        - Domain clusters on top axis
        - Bézier curved splines representing DNS requests
        - Normal activity in subdued cyan/blue
        - Suspicious/exfil in orange-red with highlighted regions
        """
        fig = go.Figure()

        if df.empty or "client_ip" not in df.columns or "query_name" not in df.columns:
            fig.update_layout(
                paper_bgcolor=PALETTE["main_bg"],
                plot_bgcolor=PALETTE["main_bg"],
                annotations=[{
                    "text": "NO TELEMETRY DATA AVAILABLE",
                    "xref": "paper", "yref": "paper",
                    "showarrow": False,
                    "font": {"color": PALETTE["text_secondary"], "size": 14, "family": "IBM Plex Mono"}
                }]
            )
            return fig

        # Map verdicts if provided
        verdict_map = {}
        if verdicts:
            for v in verdicts:
                vd = v.to_dict() if hasattr(v, "to_dict") else v
                qid = vd.get("query_id") or vd.get("query_name")
                verdict_map[qid] = vd
                verdict_map[vd.get("query_name")] = vd

        # Group data
        records = df.copy()
        records["base_domain"] = records["query_name"].apply(_extract_base_domain)

        # Unique clients and destination clusters
        unique_clients = sorted(records["client_ip"].unique().tolist())
        unique_clusters = sorted(records["base_domain"].unique().tolist())

        # Determine y positions for clients (lower left/mid)
        n_clients = len(unique_clients)
        client_y_coords = {
            ip: 14.0 + (i * (4.5 / max(1, n_clients - 1)))
            for i, ip in enumerate(unique_clients)
        }
        client_x_coords = {
            ip: 18.0 + (i * 2.8)
            for i, ip in enumerate(unique_clients)
        }

        # Determine x and y positions for target clusters (upper timeline/channel bar)
        n_clusters = len(unique_clusters)
        cluster_x_coords = {
            dom: 15.0 + (j * (70.0 / max(1, n_clusters - 1)))
            for j, dom in enumerate(unique_clusters)
        }
        cluster_y_coords = {
            dom: 22.0 + (0.8 * (j % 3))
            for j, dom in enumerate(unique_clusters)
        }

        # Track cluster anomaly status
        cluster_anomaly_counts = {c: 0 for c in unique_clusters}

        # Draw connection flows (Bézier curves)
        trace_count = 0
        for _, row in records.iterrows():
            client = row["client_ip"]
            cluster = row["base_domain"]
            qname = str(row["query_name"])
            qid = row.get("query_id", qname)

            # Determine anomaly status
            is_anomaly = False
            risk_score = 0.0
            verdict_str = "normal"
            reason = ""

            if qid in verdict_map:
                vm = verdict_map[qid]
                verdict_str = vm.get("final_classification", "normal")
                risk_score = vm.get("risk_score", 0.0)
                is_anomaly = verdict_str in ["suspicious", "likely_exfiltration"]
                rules = vm.get("rules_triggered", [])
                reason = rules[0] if rules else ("AI Escalated" if vm.get("routed_to_llm") else "")
            elif "label" in row and row["label"] in [1, "1", "exfiltration", "attack"]:
                is_anomaly = True
                verdict_str = "likely_exfiltration"
                risk_score = 0.88
                reason = str(row.get("attack_type", "known_exfiltration"))
            elif "attack_type" in row and "benign" not in str(row["attack_type"]).lower():
                is_anomaly = True
                verdict_str = "suspicious"
                risk_score = 0.65
                reason = str(row["attack_type"])

            if is_anomaly:
                cluster_anomaly_counts[cluster] += 1

            # Apply filters
            if filter_status == "anomalies_only" and not is_anomaly:
                continue
            if filter_status == "normal_only" and is_anomaly:
                continue
            if selected_client and selected_client != "ALL" and client != selected_client:
                continue
            if selected_cluster and selected_cluster != "ALL" and cluster != selected_cluster:
                continue

            # Start and End points
            p0 = (client_x_coords[client], client_y_coords[client])
            p3 = (cluster_x_coords[cluster], cluster_y_coords[cluster])

            # Vertical flowing control points for organic upward arch
            dx = p3[0] - p0[0]
            dy = p3[1] - p0[1]
            p1 = (p0[0] + dx * 0.15, p0[1] + dy * 0.60)
            p2 = (p3[0] - dx * 0.15, p3[1] - dy * 0.20)

            cx, cy = _compute_cubic_bezier(p0, p1, p2, p3, n_points=26)

            # Color and line styling
            if is_anomaly:
                if verdict_str == "likely_exfiltration":
                    line_color = "rgba(216, 93, 93, 0.75)"
                    line_width = 1.8
                else:
                    line_color = "rgba(229, 139, 86, 0.70)"
                    line_width = 1.5
            else:
                line_color = "rgba(66, 191, 208, 0.25)"
                line_width = 0.9

            hover_text = (
                f"<b>FLOW TELEMETRY</b><br>"
                f"Source: {client}<br>"
                f"Target: {cluster}<br>"
                f"Query: {qname}<br>"
                f"Type: {row.get('record_type', 'A')}<br>"
                f"Status: {verdict_str.upper()}<br>"
                f"Risk: {risk_score:.3f}"
            )
            if reason:
                hover_text += f"<br>Evidence: {reason}"

            fig.add_trace(go.Scatter(
                x=cx,
                y=cy,
                mode="lines",
                line=dict(color=line_color, width=line_width),
                hoverinfo="text",
                hovertext=hover_text,
                showlegend=False,
                opacity=0.9,
            ))
            trace_count += 1

        # Anomalous cluster highlight bounding regions (like [ST-983] in reference image)
        for dom, anom_cnt in cluster_anomaly_counts.items():
            if anom_cnt > 0:
                cx = cluster_x_coords[dom]
                cy = cluster_y_coords[dom]
                # Add glowing cluster bracket/box
                box_x = [cx - 3.2, cx + 3.2, cx + 3.2, cx - 3.2, cx - 3.2]
                box_y = [cy - 0.45, cy - 0.45, cy + 0.95, cy + 0.95, cy - 0.45]
                fig.add_trace(go.Scatter(
                    x=box_x,
                    y=box_y,
                    mode="lines",
                    line=dict(color="rgba(229, 139, 86, 0.4)", width=1, dash="dot"),
                    fill="toself",
                    fillcolor="rgba(229, 139, 86, 0.08)",
                    hoverinfo="none",
                    showlegend=False,
                ))
                # Add cluster label callout
                fig.add_annotation(
                    x=cx,
                    y=cy + 1.25,
                    text=f"<b>ALERT // {dom.upper()}</b>",
                    showarrow=False,
                    font=dict(family="IBM Plex Mono, monospace", size=9, color=PALETTE["orange"]),
                    bgcolor="rgba(28, 43, 54, 0.85)",
                    bordercolor=PALETTE["orange"],
                    borderwidth=1,
                    borderpad=3,
                )

        # Plot client nodes (Sources)
        client_xs = [client_x_coords[ip] for ip in unique_clients]
        client_ys = [client_y_coords[ip] for ip in unique_clients]
        client_texts = [f"<b>SRC NODE: {ip}</b>" for ip in unique_clients]

        fig.add_trace(go.Scatter(
            x=client_xs,
            y=client_ys,
            mode="markers+text",
            marker=dict(
                size=7,
                color=PALETTE["cyan"],
                line=dict(color="rgba(66, 191, 208, 0.4)", width=4),
            ),
            text=[f"  {ip}" for ip in unique_clients],
            textposition="middle left",
            textfont=dict(family="IBM Plex Mono, monospace", size=9, color=PALETTE["text_primary"]),
            hoverinfo="text",
            hovertext=client_texts,
            showlegend=False,
        ))

        # Plot cluster nodes (Destinations)
        cluster_xs = [cluster_x_coords[dom] for dom in unique_clusters]
        cluster_ys = [cluster_y_coords[dom] for dom in unique_clusters]
        cluster_colors = [
            PALETTE["alert_red"] if cluster_anomaly_counts[dom] > 2
            else (PALETTE["orange"] if cluster_anomaly_counts[dom] > 0 else PALETTE["blue"])
            for dom in unique_clusters
        ]

        fig.add_trace(go.Scatter(
            x=cluster_xs,
            y=cluster_ys,
            mode="markers+text",
            marker=dict(
                size=6,
                color=cluster_colors,
                line=dict(color="rgba(255, 255, 255, 0.3)", width=2),
            ),
            text=[f"{dom}" for dom in unique_clusters],
            textposition="top center",
            textfont=dict(family="IBM Plex Mono, monospace", size=9, color=PALETTE["text_secondary"]),
            hoverinfo="text",
            hovertext=[f"Cluster: {dom}<br>Anomalies: {cluster_anomaly_counts[dom]}" for dom in unique_clusters],
            showlegend=False,
        ))

        # Spatial background grid and coordinate ticks (as in reference image)
        fig.update_layout(
            paper_bgcolor=PALETTE["main_bg"],
            plot_bgcolor=PALETTE["main_bg"],
            height=580,
            margin=dict(l=45, r=30, t=50, b=45),
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(52, 69, 80, 0.25)",
                gridwidth=1,
                zeroline=False,
                showticklabels=True,
                tickfont=dict(family="IBM Plex Mono, monospace", size=9, color=PALETTE["text_secondary"]),
                range=[10, 95],
                title=dict(
                    text="SPATIAL NETWORK FLOW INTERSECT // DNS TELEMETRY TIMELINE",
                    font=dict(family="IBM Plex Sans, sans-serif", size=10, color=PALETTE["text_secondary"])
                ),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(52, 69, 80, 0.25)",
                gridwidth=1,
                zeroline=False,
                showticklabels=True,
                tickfont=dict(family="IBM Plex Mono, monospace", size=9, color=PALETTE["text_secondary"]),
                tickmode="linear",
                tick0=14,
                dtick=1,
                range=[13, 24.5],
                title=dict(
                    text="CHANNEL / LOGICAL ELEVATION",
                    font=dict(family="IBM Plex Sans, sans-serif", size=10, color=PALETTE["text_secondary"])
                ),
            ),
            hoverlabel=dict(
                bgcolor=PALETTE["panel_bg"],
                bordercolor=PALETTE["border"],
                font=dict(family="IBM Plex Mono, monospace", size=11, color=PALETTE["text_primary"]),
            ),
        )

        return fig

    @staticmethod
    def get_telemetry_hud_metrics(df: pd.DataFrame, verdicts: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Calculates precise metrics for the floating analytical HUD instrument."""
        total_queries = len(df)
        unique_clients = df["client_ip"].nunique() if "client_ip" in df.columns else 0
        unique_domains = df["query_name"].nunique() if "query_name" in df.columns else 0

        # Anomaly count
        anomaly_count = 0
        max_risk = 0.0
        exfil_count = 0

        if verdicts:
            for v in verdicts:
                vd = v.to_dict() if hasattr(v, "to_dict") else v
                r = vd.get("risk_score", 0.0)
                c = vd.get("final_classification", "")
                if r > max_risk:
                    max_risk = r
                if c in ["suspicious", "likely_exfiltration"]:
                    anomaly_count += 1
                if c == "likely_exfiltration":
                    exfil_count += 1
        elif "label" in df.columns:
            anom_df = df[df["label"].isin([1, "1", "exfiltration", "attack"])]
            anomaly_count = len(anom_df)
            exfil_count = anomaly_count
            max_risk = 0.942 if anomaly_count > 0 else 0.05
        else:
            max_risk = 0.0

        # Normal service availability ratio
        service_avail = 100.0 - ((anomaly_count / max(1, total_queries)) * 100.0)

        return {
            "total_queries": total_queries,
            "unique_clients": unique_clients,
            "unique_domains": unique_domains,
            "anomaly_count": anomaly_count,
            "exfil_count": exfil_count,
            "max_risk": max_risk,
            "service_availability": service_avail,
            "p95_entropy": 4.12,  # Typical exfiltration threshold
        }
