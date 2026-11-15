"""AbuseIPDB threat intelligence client with caching, retries, and safety enforcement."""

import requests

from .base import ThreatIntelClient, Verdict


class AbuseIPDBClient(ThreatIntelClient):
    service_name = "abuseipdb"
    api_url = "https://api.abuseipdb.com/api/v2/check"

    def _search_impl(self, indicator: str, indicator_type: str) -> dict:
        if indicator_type != "ip" or not indicator:
            return self._build_result({}, errors=[f"AbuseIPDB only supports IP indicators, got {indicator_type}"],
                                     verdict=Verdict.INCONCLUSIVE)

        headers = {"Accept": "application/json", "Key": self._get_api_key()}
        resp = requests.get(
            self.api_url,
            headers=headers,
            params={"ipAddress": indicator, "maxAgeInDays": "90"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        report = data.get("data", {})

        confidence = report.get("confidenceScore", 0)
        report_count = report.get("totalReports", 0)

        # Determine verdict based on confidence and report count
        if confidence >= 75 or report_count >= 10:
            verdict = Verdict.MALICIOUS
        elif confidence >= 25 or report_count >= 1:
            verdict = Verdict.INCONCLUSIVE  # Suspicious but not definitive
        else:
            verdict = Verdict.INCONCLUSIVE

        return self._build_result({
            "ip": indicator,
            "is_public": not report.get("isPrivate"),
            "confidence": confidence,
            "report_count": report_count,
            "recent_reports": report.get("recentReports", []),
            "usage_type": report.get("usageType", "unknown"),
            "country": report.get("countryCode", "unknown"),
            "violated_terms": report.get("violatesTermsOfService", False),
        }, verdict=verdict)

    def search(self, indicator: str, indicator_type: str) -> dict:
        return self._safe_search(indicator, indicator_type)