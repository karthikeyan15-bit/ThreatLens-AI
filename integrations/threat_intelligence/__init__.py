"""Threat intelligence integration registry."""

from .abuseipdb import AbuseIPDBClient
from .base import ThreatIntelClient
from .phishtank import PhishTankClient
from .urlhaus import URLHausClient
from .virustotal import VirusTotalClient

_CLIENTS = {
    "urlhaus": URLHausClient,
    "phishtank": PhishTankClient,
    "abuseipdb": AbuseIPDBClient,
    "virustotal": VirusTotalClient,
}


def get_client(service_name: str) -> ThreatIntelClient | None:
    cls = _CLIENTS.get(service_name.lower())
    if not cls:
        return None
    return cls()


def search_all(indicator: str, indicator_type: str) -> list[dict]:
    """Query all configured sources for an indicator."""
    results = []
    for name, cls in _CLIENTS.items():
        client = cls()
        if not client.is_configured():
            results.append({
                "service": name,
                "configured": False,
                "skipped": True,
                "reason": "API key not configured",
            })
            continue
        try:
            result = client.search(indicator, indicator_type)
            results.append(result)
        except Exception as e:
            results.append({"service": name, "error": str(e), "skipped": True})
    return results


__all__ = ["get_client", "search_all", "ThreatIntelClient"]
