"""VirusTotal threat intelligence client with caching, retries, and safety enforcement."""

import requests
from urllib.parse import urlparse

from .base import ThreatIntelClient, Verdict


class VirusTotalClient(ThreatIntelClient):
    service_name = "virustotal"
    base_url = "https://www.virustotal.com/api/v3"

    def _determine_verdict(self, stats: dict) -> str:
        """Determine verdict from VT last_analysis_stats."""
        malicious = stats.get("malicious", 0)
        suspicious = stats.get("suspicious", 0)
        undetected = stats.get("undetected", 0)
        harmless = stats.get("harmless", 0)

        total = malicious + suspicious + undetected + harmless
        if total == 0:
            return Verdict.INCONCLUSIVE

        malicious_ratio = (malicious + suspicious) / total

        if malicious_ratio >= 0.3 or malicious >= 3:
            return Verdict.MALICIOUS
        elif malicious_ratio > 0 or malicious >= 1:
            return Verdict.INCONCLUSIVE
        else:
            return Verdict.INCONCLUSIVE

    def _search_impl(self, indicator: str, indicator_type: str) -> dict:
        if indicator_type not in ("url", "domain", "ip"):
            return self._build_result({}, errors=[f"VT unsupported indicator type: {indicator_type}"],
                                     verdict=Verdict.INCONCLUSIVE)

        headers = {"x-apikey": self._get_api_key()}

        try:
            if indicator_type == "url":
                return self._lookup_url(headers, indicator)
            elif indicator_type == "domain":
                return self._lookup_domain(headers, indicator)
            elif indicator_type == "ip":
                return self._lookup_ip(headers, indicator)
        except requests.Timeout:
            return self._build_result({}, errors=["Request timed out"], verdict=Verdict.ERROR)
        except requests.RequestException as e:
            return self._build_result({}, errors=[f"HTTP error: {e}"], verdict=Verdict.ERROR)
        except Exception as e:
            return self._build_result({}, errors=[str(e)], verdict=Verdict.ERROR)

        return self._build_result({}, verdict=Verdict.INCONCLUSIVE)

    def _lookup_url(self, headers: dict, url: str) -> dict:
        encoded = url.replace("/", "%2F").replace(":", "%3A")
        resp = requests.get(f"{self.base_url}/urls/{encoded}", headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        attrs = data.get("attributes", {})
        stats = attrs.get("last_analysis_stats", {})
        verdict = self._determine_verdict(stats)

        return self._build_result({
            "type": "url",
            "reputation": attrs.get("reputation", "unknown"),
            "last_analysis_stats": stats,
            "categories": attrs.get("categories", {}),
            "times_submitted": attrs.get("times_submitted", 0),
            "last_submission_date": attrs.get("last_submission_date"),
        }, verdict=verdict)

    def _lookup_domain(self, headers: dict, domain: str) -> dict:
        resp = requests.get(f"{self.base_url}/domains/{domain}", headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        attrs = data.get("attributes", {})
        stats = attrs.get("last_analysis_stats", {})
        verdict = self._determine_verdict(stats)

        return self._build_result({
            "type": "domain",
            "reputation": attrs.get("reputation", "unknown"),
            "last_analysis_stats": stats,
            "whois": attrs.get("whois", "")[:200] if attrs.get("whois") else "",
        }, verdict=verdict)

    def _lookup_ip(self, headers: dict, ip: str) -> dict:
        resp = requests.get(f"{self.base_url}/ip_addresses/{ip}", headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        attrs = data.get("attributes", {})
        stats = attrs.get("last_analysis_stats", {})
        verdict = self._determine_verdict(stats)

        return self._build_result({
            "type": "ip",
            "reputation": attrs.get("reputation", "unknown"),
            "last_analysis_stats": stats,
            "country": attrs.get("country", "unknown"),
            "asn": attrs.get("asn", "unknown"),
        }, verdict=verdict)

    def search(self, indicator: str, indicator_type: str) -> dict:
        return self._safe_search(indicator, indicator_type)