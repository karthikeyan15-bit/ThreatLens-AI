"""IP and domain intelligence tool — reputation checks, WHOIS, MX/DNS, and enrichment."""

import re
import socket
import subprocess
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from .tool_registry import ToolDefinition, register_tool


def is_private_ip(ip: str) -> bool:
    """Check if an IP is in a private/reserved range."""
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        octets = [int(p) for p in parts]
    except ValueError:
        return False
    return (
        octets[0] == 10
        or (octets[0] == 172 and 16 <= octets[1] <= 31)
        or (octets[0] == 192 and octets[1] == 168)
        or octets[0] == 127
        or octets[0] == 0
    )


def resolve_domain(domain: str) -> dict:
    """Resolve a domain to IPs and gather basic DNS info."""
    result = {"domain": domain, "resolved_ips": [], "errors": []}
    try:
        answers = socket.getaddrinfo(domain, None, socket.AF_INET)
        result["resolved_ips"] = list({r[4][0] for r in answers})
    except socket.gaierror as e:
        result["errors"].append(f"DNS resolution failed: {e}")
    return result


def get_mx_records(domain: str) -> list[str]:
    """Get MX records for a domain."""
    try:
        import dns.resolver
        answers = dns.resolver.resolve(domain, 'MX')
        return [str(r.exchange).rstrip('.') for r in answers]
    except Exception:
        return []


def get_ns_records(domain: str) -> list[str]:
    """Get NS records for a domain."""
    try:
        import dns.resolver
        answers = dns.resolver.resolve(domain, 'NS')
        return [str(r).rstrip('.') for r in answers]
    except Exception:
        return []


def get_whois_info(domain: str) -> Dict[str, Any]:
    """Get WHOIS information for a domain."""
    try:
        import whois
        w = whois.whois(domain)
        return {
            "registrar": w.registrar,
            "creation_date": str(w.creation_date) if w.creation_date else None,
            "expiration_date": str(w.expiration_date) if w.expiration_date else None,
            "name_servers": w.name_servers if w.name_servers else [],
            "status": w.status if w.status else [],
        }
    except Exception as e:
        return {"error": str(e)}


def parse_indicator(args: dict) -> tuple[str, str]:
    """Determine if input is an IP or domain and return (type, value)."""
    value = args.get("indicator", "").strip()

    ipv4_pattern = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    if ipv4_pattern.match(value):
        return "ip", value

    # Basic domain check
    if re.match(r"^[a-zA-Z0-9][\w.-]*\.[a-zA-Z]{2,}$", value):
        return "domain", value

    return "unknown", value


def analyze_ip(ip: str) -> dict:
    """Analyze an IP address structurally."""
    result = {
        "ip": ip,
        "is_private": is_private_ip(ip),
        "is_public": not is_private_ip(ip),
        "risk_notes": [],
    }

    if is_private_ip(ip):
        result["risk_notes"].append("Private/internal IP address")
    elif ip.startswith("169.254."):
        result["risk_notes"].append("Link-local address (APIPA)")
    elif ip.startswith("224.") or ip.startswith("240."):
        result["risk_notes"].append("Multicast or reserved range")

    return result


def analyze_domain(domain: str) -> dict:
    """Analyze a domain using DNS resolution, MX/NS records, and WHOIS."""
    resolved = resolve_domain(domain)
    mx_records = get_mx_records(domain)
    ns_records = get_ns_records(domain)
    whois_info = get_whois_info(domain)

    risk_notes = []

    if registrars := domain.count("."):
        if registrars > 3:
            risk_notes.append("Unusually deep subdomain hierarchy")

    if not mx_records:
        risk_notes.append("No MX records found")

    return {
        "domain": domain,
        **resolved,
        "mx_records": mx_records,
        "ns_records": ns_records,
        "whois": whois_info,
        "risk_notes": risk_notes,
        "age_estimate": whois_info.get("creation_date", "unknown"),
    }


def is_private_ip(ip: str) -> bool:
    """Check if an IP is in a private/reserved range."""
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        octets = [int(p) for p in parts]
    except ValueError:
        return False
    return (
        octets[0] == 10
        or (octets[0] == 172 and 16 <= octets[1] <= 31)
        or (octets[0] == 192 and octets[1] == 168)
        or octets[0] == 127
        or octets[0] == 0
    )


def parse_indicator(args: dict) -> tuple[str, str]:
    """Determine if input is an IP or domain and return (type, value)."""
    value = args.get("indicator", "").strip()

    ipv4_pattern = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    if ipv4_pattern.match(value):
        return "ip", value

    # Basic domain check
    if re.match(r"^[a-zA-Z0-9][\w.-]*\.[a-zA-Z]{2,}$", value):
        return "domain", value

    return "unknown", value


def run(args: dict) -> dict:
    """Entry point registered with tool registry."""
    ind_type, value = parse_indicator(args)

    if ind_type == "ip":
        result = analyze_ip(value)
    elif ind_type == "domain":
        result = analyze_domain(value)
    else:
        result = {"error": f"Could not parse indicator: {value}", "type": "unknown"}

    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    result["source"] = "local_reputation_check"
    result["threat_intel_needed"] = ind_type in ("ip", "domain")
    return result


IP_TOOL = ToolDefinition(
    name="ip_analyzer",
    description=(
        "Analyzes an IP address or domain for reputation indicators. "
        "Resolves domains via DNS (A, MX, NS records), performs WHOIS lookup, "
        "detects private/reserved IPs, and flags structural anomalies. "
        "For external threat intel APIs, use `threat_intel_lookup`."
    ),
    parameters={
        "type": "object",
        "properties": {
            "indicator": {
                "type": "string",
                "description": "An IPv4 address or domain name to investigate",
            },
        },
        "required": ["indicator"],
    },
    category="analysis",
    func=run,
)


register_tool(IP_TOOL)