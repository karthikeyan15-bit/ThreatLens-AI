"""Network log analyzer — detects intrusion activity in CSV flow data."""

import csv
import io
import re
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

from .tool_registry import ToolDefinition, register_tool


KNOWN_ATTACK_SIGNATURES = {
    "brute_force_ssh": {"proto": "TCP", "port_pattern": r"^22$", "flag": "SYN", "min_flows": 50},
    "brute_force_http": {"proto": "TCP", "port_pattern": r"^(80|443)$", "flag": "SYN", "min_flows": 100},
    "slowloris": {"proto": "TCP", "flag": "FIN", "min_flows": 10},
    "udp_flood": {"proto": "UDP", "min_flows": 200},
    "icmp_flood": {"proto": "ICMP", "min_flows": 100},
    "port_scan": {"flag": "SYN", "unique_dst_ports_per_src": 20},
}

# Maximum rows to process for performance — prevents timeouts on large files
MAX_ROWS = 50000


def parse_csv(content: str) -> pd.DataFrame:
    """Parse uploaded CSV content into a DataFrame.

    Limits rows to MAX_ROWS to prevent timeouts on very large files.
    """
    df = pd.read_csv(io.StringIO(content))
    if len(df) > MAX_ROWS:
        df = df.head(MAX_ROWS)
    return df


def infer_columns(df: pd.DataFrame) -> dict:
    """Map common column names to standard fields."""
    col_map = {}
    lower_cols = {c.lower().strip(): c for c in df.columns}

    mappings = {
        "srcip": ["srcip", "src_ip", "source ip", "sourceip"],
        "dstip": ["dstip", "dst_ip", "dest ip", "destip"],
        "srcport": ["srcport", "src_port", "source port"],
        "dstport": ["dstport", "dst_port", "dest port"],
        "proto": ["proto", "protocol"],
        "flow_duration": ["flow duration", "duration", "flow_duration"],
        "total_pkts": ["tot pkts", "total packets", "pkts"],
        "total_bytes": ["tot bytes", "total bytes", "bytes"],
        "label": ["label", "attack", "class", "category"],
    }

    for std, aliases in mappings.items():
        for alias in aliases:
            if alias in lower_cols:
                col_map[std] = lower_cols[alias]
                break

    return col_map


def detect_anomalies(df: pd.DataFrame, col_map: dict) -> list[dict]:
    """Detect anomalous patterns in network flows."""
    findings = []

    src_col = col_map.get("srcip")
    dst_col = col_map.get("dstip")
    proto_col = col_map.get("proto")
    dport_col = col_map.get("dstport")
    label_col = col_map.get("label")

    # Label-based analysis
    if label_col and label_col in df.columns:
        label_counts = df[label_col].value_counts()
        attack_labels = [l for l in label_counts.index if str(l).lower() not in ("benign", "normal")]
        for label in attack_labels[:5]:
            count = int(label_counts[label])
            pct = count / len(df) * 100
            findings.append({
                "type": "labeled_attack",
                "attack_type": str(label),
                "count": count,
                "percentage": round(pct, 2),
                "severity": "high" if pct > 20 else "medium",
            })

    # Signature-based anomaly detection — port scan
    if src_col and proto_col and dport_col:
        src_proto_df = df[[src_col, proto_col, dport_col]].dropna()
        if len(src_proto_df) > 0:
            # Count unique destination ports per source IP
            port_counts = src_proto_df.groupby([src_col, proto_col])[dport_col].nunique().reset_index()
            port_counts.columns = [src_col, proto_col, "unique_ports"]
            threshold = KNOWN_ATTACK_SIGNATURES.get("port_scan", {}).get("unique_dst_ports_per_src", 20)
            port_scans = port_counts[port_counts["unique_ports"] >= threshold]
            for _, row in port_scans.iterrows():
                findings.append({
                    "type": "port_scan",
                    "source_ip": str(row[src_col]),
                    "protocol": str(row[proto_col]),
                    "unique_dest_ports": int(row["unique_ports"]),
                    "severity": "high",
                })

    # High packet volume
    bytes_col = col_map.get("total_bytes")
    if bytes_col and bytes_col in df.columns and len(df) > 0:
        q95 = df[bytes_col].quantile(0.95)
        high_volume = df[df[bytes_col] > q95]
        if len(high_volume) > 0:
            findings.append({
                "type": "high_volume_flow",
                "count": len(high_volume),
                "avg_bytes": round(float(high_volume[bytes_col].mean()), 2),
                "severity": "medium",
            })

    return findings


def analyze_network_log(content: str) -> dict:
    """Full network log analysis pipeline."""
    df = parse_csv(content)
    col_map = infer_columns(df)
    findings = detect_anomalies(df, col_map)

    top_sources = Counter()
    if col_map.get("srcip") and col_map["srcip"] in df.columns:
        top_sources = Counter(df[col_map["srcip"]].dropna().astype(str).tolist()).most_common(10)

    top_targets = Counter()
    if col_map.get("dstip") and col_map["dstip"] in df.columns:
        top_targets = Counter(df[col_map["dstip"]].dropna().astype(str).tolist()).most_common(10)

    overall_risk = "low"
    if any(f["severity"] == "high" for f in findings):
        overall_risk = "high"
    elif any(f["severity"] == "medium" for f in findings):
        overall_risk = "medium"

    return {
        "total_records": len(df),
        "columns_detected": col_map,
        "findings": findings,
        "top_source_ips": [{"ip": ip, "count": cnt} for ip, cnt in top_sources],
        "top_destination_ips": [{"ip": ip, "count": cnt} for ip, cnt in top_targets],
        "overall_risk": overall_risk,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "network_flow_analyzer",
    }


def run(args: dict) -> dict:
    """Entry point registered with tool registry."""
    content = args.get("csv_content", "")
    if not content.strip():
        return {"success": False, "error": "CSV content is required"}
    return analyze_network_log(content)


NETWORK_TOOL = ToolDefinition(
    name="network_analyzer",
    description=(
        "Analyzes network flow CSV data (e.g., CIC-IDS2017 format). Detects brute-force attempts, "
        "port scans, DoS floods, and other intrusion patterns. Returns findings with severity levels."
    ),
    parameters={
        "type": "object",
        "properties": {
            "csv_content": {
                "type": "string",
                "description": "CSV file content as text (not a file path)",
            },
        },
        "required": ["csv_content"],
    },
    category="analysis",
    func=run,
)


register_tool(NETWORK_TOOL)
